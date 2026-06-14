#!/usr/bin/env python3
"""
preprocess_gtzan.py
GTZAN 数据预处理脚本
- 扫描 data/raw/gtzan/ 下所有 .wav（排除 ._* macOS 伴生文件）
- 统一：单声道、采样率、30s 长度、音量归一化
- 按 70/15/15 划分 train/val/test
- 输出到 data/processed/{train,val,test}/
"""

import argparse
import csv
import os
import random
import shutil
from pathlib import Path

import torch
import torchaudio
import numpy as np


SAMPLE_RATE = 32000  # MusicGen 期望的采样率
DURATION = 30        # 统一长度（秒）
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
GTZAN_GENRES = {
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
}


def looks_like_gtzan_dir(path: Path) -> bool:
    """Return True if path directly contains GTZAN genre directories."""
    if not path.exists() or not path.is_dir():
        return False
    child_names = {p.name for p in path.iterdir() if p.is_dir()}
    return len(child_names & GTZAN_GENRES) >= 5


def resolve_gtzan_dir(raw_dir: Path) -> Path:
    """Resolve common local/OpenBayes GTZAN directory layouts."""
    candidates = [
        raw_dir,
        raw_dir / "gtzan",
        raw_dir / "GTZAN",
        raw_dir / "genres_original",
        Path("/openbayes/input/input0"),
        Path("/openbayes/input/input0/gtzan"),
        Path("/openbayes/input/input0/GTZAN"),
        Path("/openbayes/input/input0/genres_original"),
    ]

    for candidate in candidates:
        if looks_like_gtzan_dir(candidate):
            return candidate

    openbayes_root = Path("/openbayes/input/input0")
    if openbayes_root.exists():
        for candidate in openbayes_root.rglob("*"):
            if candidate.is_dir() and looks_like_gtzan_dir(candidate):
                return candidate

    return raw_dir


def scan_gtzan(raw_dir: Path) -> dict[str, list[Path]]:
    """扫描 GTZAN 目录，返回 {genre: [wav_paths]}"""
    genres = {}
    for genre_dir in sorted(raw_dir.iterdir()):
        if not genre_dir.is_dir():
            continue
        wavs = [
            p for p in sorted(genre_dir.glob("*.wav"))
            if not p.name.startswith("._")
        ]
        if wavs:
            genres[genre_dir.name] = wavs
    return genres


def is_readable_audio(path: Path) -> tuple[bool, str | None]:
    """Return whether an audio file can be opened by torchaudio."""
    try:
        torchaudio.info(str(path))
        return True, None
    except Exception as exc:
        return False, str(exc)


def filter_readable_audio(
    genres: dict[str, list[Path]],
    output_dir: Path,
    strict: bool = False,
) -> dict[str, list[Path]]:
    """Filter out corrupt/unreadable audio files and write a skip report."""
    filtered: dict[str, list[Path]] = {}
    skipped = []

    for genre, paths in genres.items():
        valid_paths = []
        for path in paths:
            ok, error = is_readable_audio(path)
            if ok:
                valid_paths.append(path)
            else:
                skipped.append({
                    "genre": genre,
                    "path": str(path),
                    "error": error or "unknown error",
                })
        if valid_paths:
            filtered[genre] = valid_paths

    if skipped:
        output_dir.mkdir(parents=True, exist_ok=True)
        skipped_path = output_dir / "skipped_bad_audio.csv"
        with open(skipped_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["genre", "path", "error"])
            writer.writeheader()
            writer.writerows(skipped)

        print(f"跳过 {len(skipped)} 个无法读取的音频，详情 → {skipped_path}")
        for item in skipped[:10]:
            print(f"  跳过: {item['path']} ({item['error']})")

        if strict:
            raise RuntimeError(f"发现 {len(skipped)} 个坏音频，已按 --strict 停止")

    return filtered


def process_audio(
    path: Path,
    target_sr: int = SAMPLE_RATE,
    target_len: int = DURATION,
) -> torch.Tensor:
    """加载并预处理单条音频：重采样、单声道、裁剪/填充到固定长度"""
    waveform, sr = torchaudio.load(str(path))

    # 重采样
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # 单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 裁剪或填充到固定长度
    target_samples = target_len * target_sr
    if waveform.shape[1] > target_samples:
        waveform = waveform[:, :target_samples]
    elif waveform.shape[1] < target_samples:
        padding = target_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    # 音量归一化
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak * 0.95

    return waveform.squeeze(0)  # (samples,)


def split_and_save(
    processed_dir: Path,
    train_files: list[tuple[Path, str]],
    val_files: list[tuple[Path, str]],
    test_files: list[tuple[Path, str]],
) -> list[dict]:
    """处理并保存划分后的音频"""
    records = []
    for split_name, files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        out_dir = processed_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for src_path, genre in files:
            genre_dir = out_dir / genre
            genre_dir.mkdir(exist_ok=True)

            try:
                waveform = process_audio(src_path)
            except Exception as exc:
                print(f"警告: 处理失败，跳过 {src_path}: {exc}")
                continue

            out_path = genre_dir / src_path.name
            torchaudio.save(str(out_path), waveform.unsqueeze(0), SAMPLE_RATE)
            records.append({
                "split": split_name,
                "genre": genre,
                "source": str(src_path),
                "processed": str(out_path),
                "sample_rate": SAMPLE_RATE,
                "duration_sec": DURATION,
            })

    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    return records


def main():
    parser = argparse.ArgumentParser(description="GTZAN 数据预处理")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(os.environ.get("RAW_GTZAN_DIR", "data/raw/gtzan")),
        help="GTZAN 原始音频目录；OpenBayes 可用 /openbayes/input/input0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="处理后输出目录",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子")
    parser.add_argument("--limit-per-genre", type=int, default=None, help="调试用：每个流派最多处理 N 条")
    parser.add_argument("--strict", action="store_true", help="遇到坏音频时直接失败")
    parser.add_argument("--no-clean", action="store_true", help="不清理已有 processed 输出")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    raw_dir = resolve_gtzan_dir(args.raw_dir)
    print(f"扫描 {raw_dir} ...")
    if raw_dir != args.raw_dir:
        print(f"自动解析 GTZAN 路径: {args.raw_dir} → {raw_dir}")

    if args.output_dir.exists() and not args.no_clean:
        print(f"清理旧输出: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    genres = scan_gtzan(raw_dir)
    genres = filter_readable_audio(genres, args.output_dir, strict=args.strict)
    total = sum(len(v) for v in genres.values())
    print(f"发现 {len(genres)} 个流派，共 {total} 条音频")
    if total == 0:
        raise FileNotFoundError(
            f"没有在 {raw_dir} 找到 GTZAN wav 文件。"
            "请确认目录直接包含 blues/classical/.../rock 等流派子目录，"
            "或设置 RAW_GTZAN_DIR=/openbayes/input/input0。"
        )

    # 按流派分层划分，保证 train/val/test 都覆盖所有流派。
    train_files = []
    val_files = []
    test_files = []
    for genre, paths in genres.items():
        genre_paths = list(paths)
        random.shuffle(genre_paths)
        if args.limit_per_genre is not None:
            genre_paths = genre_paths[:args.limit_per_genre]

        n = len(genre_paths)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_files.extend((p, genre) for p in genre_paths[:n_train])
        val_files.extend((p, genre) for p in genre_paths[n_train:n_train + n_val])
        test_files.extend((p, genre) for p in genre_paths[n_train + n_val:])

    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    print(f"划分: train={len(train_files)} val={len(val_files)} test={len(test_files)}")
    print(f"采样率: {SAMPLE_RATE} Hz | 长度: {DURATION}s")

    records = split_and_save(args.output_dir, train_files, val_files, test_files)

    metadata_path = args.output_dir / "metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "genre", "source", "processed", "sample_rate", "duration_sec"],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"metadata → {metadata_path}")
    print("预处理完成 ✓")


if __name__ == "__main__":
    main()
