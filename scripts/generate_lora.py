#!/usr/bin/env python3
"""Generate continuations with a MusicGen LoRA adapter."""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from audiocraft.models import MusicGen
from peft import PeftModel


INPUT_SEC = 20
OUTPUT_SEC = 10
MODEL_NAME = "facebook/musicgen-melody"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_lora_model(model_name: str, adapter_path: Path, device: str = "auto") -> MusicGen:
    """Load base MusicGen and attach a PEFT LoRA adapter to its LM."""
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter 不存在: {adapter_path}")

    device = resolve_device(device)
    print(f"加载基座模型 {model_name} 到 {device} ...")
    model = MusicGen.get_pretrained(model_name)
    model.to(device)

    print(f"加载 LoRA adapter ← {adapter_path}")
    model.lm = PeftModel.from_pretrained(model.lm, str(adapter_path))
    expose_audiocraft_methods(model.lm)
    model.lm.to(device)
    model.lm.eval()
    return model


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


def load_prompt(audio_path: Path, model_sr: int, input_sec: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(audio_path))
    if sr != model_sr:
        wav = torchaudio.transforms.Resample(sr, model_sr)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    input_samples = input_sec * model_sr
    if wav.shape[1] < input_samples:
        wav = torch.nn.functional.pad(wav, (0, input_samples - wav.shape[1]))
    else:
        wav = wav[:, :input_samples]
    return wav.unsqueeze(0)


def extract_continuation(
    generated: torch.Tensor,
    model_sr: int,
    input_sec: int,
    output_sec: int,
) -> torch.Tensor:
    if generated.dim() == 3:
        generated = generated[0]
    elif generated.dim() == 1:
        generated = generated.unsqueeze(0)

    input_samples = input_sec * model_sr
    output_samples = output_sec * model_sr
    total_samples = input_samples + output_samples

    if generated.shape[-1] >= total_samples:
        continuation = generated[..., input_samples:total_samples]
    elif generated.shape[-1] >= output_samples:
        continuation = generated[..., -output_samples:]
    else:
        continuation = torch.nn.functional.pad(
            generated,
            (0, output_samples - generated.shape[-1]),
        )
    return continuation.detach().cpu()


def generate(
    model: MusicGen,
    audio_path: Path,
    output_path: Path,
    input_sec: int = INPUT_SEC,
    output_sec: int = OUTPUT_SEC,
    description: str = "",
    top_k: int = 250,
    top_p: float = 0.0,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
):
    """Generate continuation audio and save it."""
    model_sr = model.sample_rate
    prompt = load_prompt(audio_path, model_sr, input_sec)
    duration = input_sec + output_sec
    model.set_generation_params(
        duration=duration,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        cfg_coef=cfg_coef,
    )

    with torch.no_grad():
        generated = model.generate_continuation(
            prompt,
            model_sr,
            descriptions=[description],
            progress=True,
        )

    continuation = extract_continuation(generated, model_sr, input_sec, output_sec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), continuation, model.sample_rate)


def collect_wavs(data_dir: Path, samples_per_genre: int) -> list[tuple[str, Path]]:
    items = []
    genres = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    for genre_dir in genres:
        wavs = [
            p for p in sorted(genre_dir.glob("*.wav"))
            if not p.name.startswith("._")
        ][:samples_per_genre]
        for wav in wavs:
            items.append((genre_dir.name, wav))
    return items


def main():
    parser = argparse.ArgumentParser(description="LoRA MusicGen 续写生成")
    parser.add_argument("--adapter-path", type=Path, default=Path("outputs/lora/lora_adapter"))
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/test"))
    parser.add_argument("--external-dir", type=Path, default=Path("data/raw/external"))
    parser.add_argument("--output-lora", type=Path, default=Path("outputs/lora"))
    parser.add_argument("--output-external", type=Path, default=Path("outputs/external"))
    parser.add_argument("--samples-per-genre", type=int, default=10)
    parser.add_argument("--input-sec", type=int, default=INPUT_SEC)
    parser.add_argument("--output-sec", type=int, default=OUTPUT_SEC)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg-coef", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    model = load_lora_model(args.model_name, args.adapter_path, args.device)

    manifest = []
    total = 0

    if args.data_dir.exists():
        print(f"GTZAN 测试集: {args.data_dir}")
        for genre, wav_path in collect_wavs(args.data_dir, args.samples_per_genre):
            out_path = args.output_lora / genre / f"{wav_path.stem}_lora.wav"
            if args.skip_existing and out_path.exists():
                print(f"  [GTZAN/{genre}] {wav_path.name} 跳过，已存在")
                continue
            print(f"  [GTZAN/{genre}] {wav_path.name} ...", end=" ")
            generate(
                model,
                wav_path,
                out_path,
                input_sec=args.input_sec,
                output_sec=args.output_sec,
                description=args.description,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                cfg_coef=args.cfg_coef,
            )
            manifest.append({
                "split": "gtzan_test",
                "genre": genre,
                "source": str(wav_path),
                "generated": str(out_path),
                "adapter": str(args.adapter_path),
            })
            total += 1
            print("✓")

    if args.external_dir.exists():
        external_wavs = [
            p for p in sorted(args.external_dir.glob("*.wav"))
            if not p.name.startswith("._")
        ]
        print(f"\n外部音频: {len(external_wavs)} 条")
        for wav_path in external_wavs:
            out_path = args.output_external / f"{wav_path.stem}_lora.wav"
            if args.skip_existing and out_path.exists():
                print(f"  [external] {wav_path.name} 跳过，已存在")
                continue
            print(f"  [external] {wav_path.name} ...", end=" ")
            generate(
                model,
                wav_path,
                out_path,
                input_sec=args.input_sec,
                output_sec=args.output_sec,
                description=args.description,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                cfg_coef=args.cfg_coef,
            )
            manifest.append({
                "split": "external",
                "genre": None,
                "source": str(wav_path),
                "generated": str(out_path),
                "adapter": str(args.adapter_path),
            })
            total += 1
            print("✓")

    args.output_lora.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_lora / "manifest_lora.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nLoRA 生成完成，共 {total} 条")
    print(f"manifest → {manifest_path}")


if __name__ == "__main__":
    main()
