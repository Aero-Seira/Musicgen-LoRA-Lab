#!/usr/bin/env python3
"""Generate MusicGen continuation baselines.

Input files are 30s GTZAN clips. The first 20s are used as prompt and the
generated continuation is saved as a standalone 10s wav.
"""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from audiocraft.models import MusicGen


SAMPLE_RATE = 32000
INPUT_SEC = 20
OUTPUT_SEC = 10
MODEL_NAME = "facebook/musicgen-melody"


def resolve_device(device: str) -> str:
    """Resolve "auto" to the best available torch device."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str = "auto") -> MusicGen:
    device = resolve_device(device)
    print(f"加载预训练模型 {model_name} 到 {device} ...")
    try:
        model = MusicGen.get_pretrained(model_name, device=device)
    except TypeError:
        model = MusicGen.get_pretrained(model_name)
        for attr in ("lm", "compression_model"):
            module = getattr(model, attr, None)
            if module is not None and hasattr(module, "to"):
                module.to(device)

    for attr in ("lm", "compression_model"):
        module = getattr(model, attr, None)
        if module is not None and hasattr(module, "eval"):
            module.eval()

    if hasattr(model, "eval"):
        model.eval()
    return model


def load_prompt(audio_path: Path, model_sr: int, input_sec: int) -> torch.Tensor:
    """Load the first input_sec seconds as a mono prompt tensor [1, 1, T]."""
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
    """Return a [channels, output_samples] continuation from MusicGen output."""
    if generated.dim() == 3:
        generated = generated[0]
    elif generated.dim() == 1:
        generated = generated.unsqueeze(0)

    input_samples = input_sec * model_sr
    output_samples = output_sec * model_sr
    total_samples = input_samples + output_samples

    # AudioCraft continuation usually returns the full prompt+continuation.
    # Some versions return only newly generated audio, so handle both shapes.
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


def generate_continuation(
    model: MusicGen,
    audio_path: Path,
    input_sec: int = INPUT_SEC,
    output_sec: int = OUTPUT_SEC,
    description: str = "",
    top_k: int = 250,
    top_p: float = 0.0,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
) -> torch.Tensor:
    """Generate a continuation conditioned on the first input_sec seconds."""
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

    return extract_continuation(generated, model_sr, input_sec, output_sec)


def main():
    parser = argparse.ArgumentParser(description="预训练 MusicGen 基线生成")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
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

    if not args.data_dir.exists():
        raise FileNotFoundError(f"测试集目录不存在: {args.data_dir}")

    model = load_model(args.model_name, args.device)

    genres = sorted([d.name for d in args.data_dir.iterdir() if d.is_dir()])
    print(f"流派: {genres}")
    print(f"每个流派采样 {args.samples_per_genre} 条")

    total = 0
    manifest = []
    for genre in genres:
        genre_dir = args.data_dir / genre
        out_dir = args.output_dir / genre
        out_dir.mkdir(parents=True, exist_ok=True)

        wav_files = [
            p for p in sorted(genre_dir.glob("*.wav"))
            if not p.name.startswith("._")
        ][:args.samples_per_genre]

        for wav_path in wav_files:
            out_path = out_dir / f"{wav_path.stem}_baseline.wav"
            if args.skip_existing and out_path.exists():
                print(f"  [{genre}] {wav_path.name} 跳过，已存在")
                continue

            print(f"  [{genre}] {wav_path.name} ...", end=" ")
            generated = generate_continuation(
                model,
                wav_path,
                input_sec=args.input_sec,
                output_sec=args.output_sec,
                description=args.description,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                cfg_coef=args.cfg_coef,
            )

            torchaudio.save(str(out_path), generated, model.sample_rate)
            manifest.append({
                "genre": genre,
                "source": str(wav_path),
                "generated": str(out_path),
                "model": args.model_name,
                "input_sec": args.input_sec,
                "output_sec": args.output_sec,
            })
            total += 1
            print("✓")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n基线生成完成，共 {total} 条 → {args.output_dir}")
    print(f"manifest → {manifest_path}")


if __name__ == "__main__":
    main()
