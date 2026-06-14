#!/usr/bin/env python3
"""Evaluate generated 10s continuations against real GTZAN targets."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
import librosa


SAMPLE_RATE = 32000
INPUT_SEC = 20
TARGET_SEC = 10


def load_mono(path: Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load audio as mono [time] tensor at target_sr."""
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0)


def crop_or_pad(wav: torch.Tensor, samples: int) -> torch.Tensor:
    if wav.numel() >= samples:
        return wav[:samples]
    return torch.nn.functional.pad(wav, (0, samples - wav.numel()))


def split_reference(
    wav: torch.Tensor,
    input_sec: int = INPUT_SEC,
    target_sec: int = TARGET_SEC,
    sr: int = SAMPLE_RATE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 30s reference clip into condition 0-20s and target 20-30s."""
    input_samples = input_sec * sr
    target_samples = target_sec * sr
    wav = crop_or_pad(wav, input_samples + target_samples)
    condition = wav[:input_samples]
    target = wav[input_samples:input_samples + target_samples]
    return condition, target


def mel_spectrogram_mse(gen: torch.Tensor, real: torch.Tensor, sr: int = SAMPLE_RATE) -> float:
    """计算生成音频与真实音频的 Mel 频谱 MSE"""
    gen_np = gen.detach().cpu().numpy().astype(np.float32)
    real_np = real.detach().cpu().numpy().astype(np.float32)

    mel_gen = librosa.feature.melspectrogram(y=gen_np, sr=sr, n_mels=80)
    mel_real = librosa.feature.melspectrogram(y=real_np, sr=sr, n_mels=80)

    # 转为 dB
    mel_gen_db = librosa.power_to_db(mel_gen, ref=np.max)
    mel_real_db = librosa.power_to_db(mel_real, ref=np.max)

    # 对齐长度
    min_len = min(mel_gen_db.shape[1], mel_real_db.shape[1])
    return float(np.mean((mel_gen_db[:, :min_len] - mel_real_db[:, :min_len]) ** 2))


def chroma_similarity(gen: torch.Tensor, real: torch.Tensor, sr: int = SAMPLE_RATE) -> float:
    """计算 Chroma 特征余弦相似度"""
    gen_np = gen.detach().cpu().numpy().astype(np.float32)
    real_np = real.detach().cpu().numpy().astype(np.float32)

    chroma_gen = librosa.feature.chroma_stft(y=gen_np, sr=sr)
    chroma_real = librosa.feature.chroma_stft(y=real_np, sr=sr)

    min_len = min(chroma_gen.shape[1], chroma_real.shape[1])
    chroma_gen = chroma_gen[:, :min_len]
    chroma_real = chroma_real[:, :min_len]

    # 余弦相似度
    dot = np.sum(chroma_gen * chroma_real)
    norm_gen = np.sqrt(np.sum(chroma_gen ** 2))
    norm_real = np.sqrt(np.sum(chroma_real ** 2))

    if norm_gen * norm_real == 0:
        return 0.0
    return float(dot / (norm_gen * norm_real))


def transition_smoothness(condition: torch.Tensor, generated: torch.Tensor, sr: int = SAMPLE_RATE) -> float:
    """衡量输入结尾与生成开头的频谱变化（越小越平滑）"""
    cond_np = condition.detach().cpu().numpy().astype(np.float32)
    gen_np = generated.detach().cpu().numpy().astype(np.float32)

    # 取最后 1s 和开头 1s
    hop = sr
    tail = cond_np[-hop:] if cond_np.shape[0] >= hop else np.pad(cond_np, (hop - cond_np.shape[0], 0))
    head = gen_np[:hop] if gen_np.shape[0] >= hop else np.pad(gen_np, (0, hop - gen_np.shape[0]))

    mel_tail = librosa.feature.melspectrogram(y=tail, sr=sr, n_mels=80)
    mel_head = librosa.feature.melspectrogram(y=head, sr=sr, n_mels=80)

    mel_tail_db = librosa.power_to_db(mel_tail, ref=np.max)
    mel_head_db = librosa.power_to_db(mel_head, ref=np.max)

    return float(np.mean((mel_tail_db - mel_head_db) ** 2))


def evaluate_pair(
    generated_path: Path,
    real_path: Path,
    input_sec: int = INPUT_SEC,
    target_sec: int = TARGET_SEC,
    sr: int = SAMPLE_RATE,
) -> dict:
    """评估一对生成音频"""
    reference = load_mono(real_path, sr)
    generated = load_mono(generated_path, sr)

    condition, real_target = split_reference(reference, input_sec, target_sec, sr)
    generated = crop_or_pad(generated, target_sec * sr)
    return {
        "mel_mse": mel_spectrogram_mse(generated, real_target, sr),
        "chroma_sim": chroma_similarity(generated, real_target, sr),
        "smoothness": transition_smoothness(condition, generated, sr),
    }


def strip_generation_suffix(stem: str) -> str:
    """Map generated file stems back to original GTZAN stems."""
    for suffix in ("_baseline", "_lora", "_generated", "_continuation"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_reference(real_dir: Path, genre: str, generated_file: Path) -> Path | None:
    stem = strip_generation_suffix(generated_file.stem)
    genre_dir = real_dir / genre

    exact = genre_dir / f"{stem}.wav"
    if exact.exists():
        return exact

    candidates = [
        p for p in genre_dir.glob(f"{stem}*.wav")
        if not p.name.startswith("._")
    ]
    return candidates[0] if candidates else None


def summarize(scores: list[dict]) -> dict:
    return {
        k: float(np.mean([s[k] for s in scores]))
        for k in scores[0].keys()
    }


def main():
    parser = argparse.ArgumentParser(description="客观指标评估")
    parser.add_argument("--real-dir", type=Path, default=Path("data/processed/test"), help="真实音频目录")
    parser.add_argument("--generated-dirs", nargs="+", type=Path, help="生成音频目录列表")
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics/results.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("outputs/metrics/results.csv"))
    parser.add_argument("--input-sec", type=int, default=INPUT_SEC)
    parser.add_argument("--target-sec", type=int, default=TARGET_SEC)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    args = parser.parse_args()

    if not args.generated_dirs:
        raise SystemExit("请提供 --generated-dirs，例如 outputs/baseline outputs/lora")
    if not args.real_dir.exists():
        raise FileNotFoundError(f"真实音频目录不存在: {args.real_dir}")

    results = {}
    rows = []

    for gen_dir in args.generated_dirs:
        if not gen_dir.exists():
            print(f"\n跳过不存在的目录: {gen_dir}")
            continue

        print(f"\n评估 {gen_dir.name} ...")
        genre_results = {}
        all_scores = []

        for genre_dir in sorted(gen_dir.iterdir()):
            if not genre_dir.is_dir():
                continue

            genre_scores = []
            for gen_file in sorted(genre_dir.glob("*.wav")):
                if gen_file.name.startswith("._"):
                    continue

                real_file = find_reference(args.real_dir, genre_dir.name, gen_file)
                if real_file is None:
                    print(f"  未找到参考音频，跳过: {gen_file}")
                    continue

                scores = evaluate_pair(
                    generated_path=gen_file,
                    real_path=real_file,
                    input_sec=args.input_sec,
                    target_sec=args.target_sec,
                    sr=args.sample_rate,
                )
                scores_with_meta = {
                    "set": gen_dir.name,
                    "genre": genre_dir.name,
                    "generated": str(gen_file),
                    "reference": str(real_file),
                    **scores,
                }
                rows.append(scores_with_meta)
                genre_scores.append(scores)
                all_scores.append(scores)

            if genre_scores:
                avg_scores = summarize(genre_scores)
                genre_results[genre_dir.name] = avg_scores
                print(f"  {genre_dir.name}: Mel MSE={avg_scores['mel_mse']:.4f} | "
                      f"Chroma={avg_scores['chroma_sim']:.4f} | "
                      f"Smooth={avg_scores['smoothness']:.4f}")

        if all_scores:
            overall = summarize(all_scores)
            genre_results["_overall"] = overall
            print(f"  overall: Mel MSE={overall['mel_mse']:.4f} | "
                  f"Chroma={overall['chroma_sim']:.4f} | "
                  f"Smooth={overall['smoothness']:.4f}")

        results[gen_dir.name] = genre_results

    # 保存结果
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存 → {args.output}")

    if rows:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["set", "genre", "generated", "reference", "mel_mse", "chroma_sim", "smoothness"]
        with open(args.csv_output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"逐样本 CSV → {args.csv_output}")


if __name__ == "__main__":
    main()
