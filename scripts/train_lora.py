#!/usr/bin/env python3
"""LoRA fine-tuning entrypoint for the MusicGen language model.

This script fine-tunes the LM part of AudioCraft MusicGen on 30s GTZAN clips.
The model is trained to predict EnCodec token sequences from the dataset. At
inference time the adapted LM is still used through MusicGen continuation.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ConditioningAttributes
from peft import LoraConfig, get_peft_model


SAMPLE_RATE = 32000
DURATION_SEC = 30
MODEL_NAME = "facebook/musicgen-melody"


class GTZANDataset(Dataset):
    """Load processed GTZAN wav files as fixed 30s mono tensors [1, T]."""

    def __init__(self, data_dir: Path, max_samples: int | None = None):
        self.files = [
            p for p in sorted(data_dir.rglob("*.wav"))
            if not p.name.startswith("._")
        ]
        if max_samples is not None:
            self.files = self.files[:max_samples]

        if not self.files:
            raise FileNotFoundError(f"没有找到训练音频: {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        wav, sr = torchaudio.load(str(self.files[idx]))
        if sr != SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        total_samples = DURATION_SEC * SAMPLE_RATE
        if wav.shape[1] < total_samples:
            wav = torch.nn.functional.pad(wav, (0, total_samples - wav.shape[1]))
        else:
            wav = wav[:, :total_samples]

        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak * 0.95
        return wav


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def linear_module_suffixes(module: torch.nn.Module) -> set[str]:
    return {
        name.split(".")[-1]
        for name, child in module.named_modules()
        if isinstance(child, torch.nn.Linear) and name
    }


def choose_lora_targets(lm: torch.nn.Module, requested: str) -> list[str]:
    """Return PEFT target module suffixes present in the MusicGen LM."""
    available = linear_module_suffixes(lm)
    if not available:
        raise RuntimeError("没有在 MusicGen LM 中找到 torch.nn.Linear 模块，无法应用 LoRA")

    if requested == "auto":
        preferred = [
            "q_proj", "k_proj", "v_proj", "out_proj",
            "linear1", "linear2", "fc1", "fc2",
        ]
        targets = [name for name in preferred if name in available]
        if not targets:
            # Conservative fallback: adapt feed-forward/output projections only.
            targets = sorted(
                name for name in available
                if any(key in name.lower() for key in ("proj", "linear", "fc"))
            )
    else:
        targets = [item.strip() for item in requested.split(",") if item.strip()]
        missing = [name for name in targets if name not in available]
        if missing:
            print(f"警告: 以下 LoRA target_modules 未在 LM 中作为 Linear 后缀出现: {missing}")

    if not targets:
        preview = ", ".join(sorted(available)[:80])
        raise RuntimeError(f"无法确定 LoRA target_modules。可用 Linear 后缀示例: {preview}")

    print(f"LoRA target_modules: {targets}")
    return targets


def apply_lora(lm: torch.nn.Module, rank: int, alpha: int, dropout: float, target_modules: str):
    targets = choose_lora_targets(lm, target_modules)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=targets,
        lora_dropout=dropout,
        bias="none",
    )
    peft_lm = get_peft_model(lm, config)
    expose_audiocraft_methods(peft_lm)
    peft_lm.print_trainable_parameters()
    return peft_lm


def expose_audiocraft_methods(peft_lm: torch.nn.Module) -> torch.nn.Module:
    """Expose AudioCraft LM methods that PEFT may hide behind its wrapper."""
    base_model = getattr(peft_lm, "base_model", None)
    inner = getattr(base_model, "model", None)
    if inner is None:
        return peft_lm

    for method_name in ("generate", "compute_predictions"):
        if hasattr(inner, method_name):
            setattr(peft_lm, method_name, getattr(inner, method_name))
    return peft_lm


def build_conditions(model: MusicGen, batch_size: int, description: str) -> list[ConditioningAttributes]:
    descriptions = [description] * batch_size
    if hasattr(model, "_prepare_tokens_and_attributes"):
        attributes, _ = model._prepare_tokens_and_attributes(descriptions, None)
        return attributes
    return [ConditioningAttributes(text={"description": description}) for _ in range(batch_size)]


def encode_audio_tokens(model: MusicGen, wav: torch.Tensor) -> torch.Tensor:
    """Encode waveform [B, C, T] to EnCodec codes [B, K, T_codes]."""
    with torch.no_grad():
        encoded = model.compression_model.encode(wav)

    if isinstance(encoded, tuple):
        codes = encoded[0]
    elif hasattr(encoded, "codes"):
        codes = encoded.codes
    else:
        codes = encoded

    return codes.detach().long()


def compute_lm_predictions(lm: torch.nn.Module, codes: torch.Tensor, conditions):
    if hasattr(lm, "compute_predictions"):
        return lm.compute_predictions(codes, conditions)

    # PeftModel usually delegates attributes, but keep a defensive fallback.
    base_model = getattr(lm, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        inner = base_model.model
        if hasattr(inner, "compute_predictions"):
            return inner.compute_predictions(codes, conditions)

    raise AttributeError("当前 LM 对象没有 compute_predictions 方法，无法计算训练 loss")


def normalize_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalize LM logits to [B, K, T, V]."""
    if logits.dim() != 4:
        raise RuntimeError(f"期望 logits 为 4 维，实际 shape={tuple(logits.shape)}")

    bsz, codebooks, steps = target.shape
    if logits.shape[0] != bsz or logits.shape[1] != codebooks:
        raise RuntimeError(
            f"logits 与 target batch/codebook 不匹配: logits={tuple(logits.shape)}, "
            f"target={tuple(target.shape)}"
        )

    if logits.shape[2] == steps:
        return logits
    if logits.shape[3] == steps:
        return logits.permute(0, 1, 3, 2).contiguous()

    raise RuntimeError(
        f"无法判断 logits 时间维: logits={tuple(logits.shape)}, target={tuple(target.shape)}"
    )


def compute_loss(lm_output, codes: torch.Tensor) -> torch.Tensor:
    logits = normalize_logits(lm_output.logits, codes)
    mask = getattr(lm_output, "mask", None)

    min_steps = min(logits.shape[2], codes.shape[2])
    logits = logits[:, :, :min_steps, :]
    labels = codes[:, :, :min_steps]

    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        mask = mask[:, :, :min_steps].bool()

    valid_logits = logits[mask]
    valid_labels = labels[mask]
    if valid_labels.numel() == 0:
        raise RuntimeError("LM mask 中没有有效 token，无法计算 loss")

    return F.cross_entropy(valid_logits, valid_labels)


def run_epoch(
    model: MusicGen,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    description: str,
    amp: bool,
    max_batches: int | None = None,
) -> float:
    training = optimizer is not None
    model.lm.train(training)
    total_loss = 0.0
    count = 0

    for batch_idx, wav in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        wav = wav.to(device)
        codes = encode_audio_tokens(model, wav)
        conditions = build_conditions(model, codes.shape[0], description)

        if training:
            optimizer.zero_grad(set_to_none=True)

        autocast_enabled = amp and device == "cuda"
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            lm_output = compute_lm_predictions(model.lm, codes, conditions)
            loss = compute_loss(lm_output, codes)

        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.lm.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        count += 1

    return total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description="LoRA 微调 MusicGen LM")
    parser.add_argument("--train-dir", type=Path, default=Path("data/processed/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/processed/val"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lora"))
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=20)
    parser.add_argument("--amp", action="store_true", help="启用 CUDA mixed precision")
    args = parser.parse_args()

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载模型 {args.model_name} 到 {device} ...")
    model = MusicGen.get_pretrained(args.model_name)
    model.to(device)
    model.compression_model.eval()
    for param in model.compression_model.parameters():
        param.requires_grad = False

    print(f"应用 LoRA: rank={args.rank}, alpha={args.alpha}, dropout={args.dropout}")
    model.lm = apply_lora(
        model.lm,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.target_modules,
    )
    model.lm.to(device)

    train_dataset = GTZANDataset(args.train_dir, max_samples=args.max_train_samples)
    val_dataset = GTZANDataset(args.val_dir, max_samples=args.max_val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")
    optimizer = torch.optim.AdamW(
        [p for p in model.lm.parameters() if p.requires_grad],
        lr=args.lr,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            description=args.description,
            amp=args.amp,
        )
        with torch.no_grad():
            val_loss = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                description=args.description,
                amp=args.amp,
                max_batches=args.max_val_batches,
            )

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print(f"Epoch [{epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        with open(args.output_dir / "training_log.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    adapter_path = args.output_dir / "lora_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(str(adapter_path))

    config = {
        "model_name": args.model_name,
        "sample_rate": SAMPLE_RATE,
        "duration_sec": DURATION_SEC,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": args.target_modules,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "description": args.description,
    }
    with open(args.output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"LoRA adapter 已保存 → {adapter_path}")
    print(f"训练日志 → {args.output_dir / 'training_log.json'}")


if __name__ == "__main__":
    main()
