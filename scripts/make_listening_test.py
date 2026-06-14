#!/usr/bin/env python3
"""Create a simple randomized A/B listening-test HTML page."""

import argparse
import random
import json
from pathlib import Path
from datetime import datetime


def collect_samples(
    baseline_dir: Path,
    lora_dir: Path,
    n_samples: int = 10,
    seed: int = 42,
) -> list[dict]:
    """收集 baseline 和 lora 的对比样本对"""
    random.seed(seed)

    samples = []
    bl_genres = set(d.name for d in baseline_dir.iterdir() if d.is_dir())
    lr_genres = set(d.name for d in lora_dir.iterdir() if d.is_dir())
    genres = sorted(bl_genres & lr_genres)

    for genre in genres:
        bl_files = sorted((baseline_dir / genre).glob("*.wav"))
        lr_files = sorted((lora_dir / genre).glob("*.wav"))

        # 按 stem 匹配
        bl_map = {f.stem.replace("_baseline", ""): f for f in bl_files}
        lr_map = {f.stem.replace("_lora", ""): f for f in lr_files}

        common_stems = sorted(set(bl_map.keys()) & set(lr_map.keys()))
        for stem in common_stems[:max(n_samples // len(genres), 1)]:
            samples.append({
                "genre": genre,
                "stem": stem,
                "baseline": str(bl_map[stem]),
                "lora": str(lr_map[stem]),
            })

    random.shuffle(samples)
    return samples[:n_samples]


def audio_src(path: str) -> str:
    return Path(path).resolve().as_uri()


def randomize_ab(samples: list[dict], seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    randomized = []
    mapping = []

    for idx, sample in enumerate(samples, start=1):
        pair = [
            ("baseline", sample["baseline"]),
            ("lora", sample["lora"]),
        ]
        rng.shuffle(pair)
        a_model, a_path = pair[0]
        b_model, b_path = pair[1]

        randomized.append({
            "id": idx,
            "genre": sample["genre"],
            "stem": sample["stem"],
            "audio_a": a_path,
            "audio_b": b_path,
        })
        mapping.append({
            "id": idx,
            "genre": sample["genre"],
            "stem": sample["stem"],
            "A": a_model,
            "B": b_model,
            "audio_a": a_path,
            "audio_b": b_path,
        })

    return randomized, mapping


def generate_html_table(samples: list[dict], output_path: Path):
    """生成 HTML 听测表格"""
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>MusicGen LoRA 主观音频听测</title>
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; }
    h1 { font-size: 1.5em; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { border: 1px solid #ddd; padding: 10px; text-align: center; }
    th { background: #f5f5f5; font-weight: 600; }
    .instruction { background: #f9f9f9; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    audio { width: 230px; }
    .rating { width: 60px; }
</style>
</head>
<body>
<h1>MusicGen LoRA 主观音频听测</h1>

<div class="instruction">
<strong>评分说明：</strong><br>
请听每对音频（A 和 B），在各维度打 1-5 分（1=差，5=好）。<br>
- <strong>连贯性</strong>：生成片段与前 20 秒是否自然衔接<br>
- <strong>风格一致性</strong>：是否保持原音乐风格<br>
- <strong>音质</strong>：是否有明显噪声/断裂/失真<br>
- <strong>偏好</strong>：你更喜欢 A 还是 B？（填 A/B）
</div>

<table>
<thead>
<tr>
    <th>#</th>
    <th>流派</th>
    <th>音频 A</th>
    <th>连贯性 A</th>
    <th>风格 A</th>
    <th>音质 A</th>
    <th>音频 B</th>
    <th>连贯性 B</th>
    <th>风格 B</th>
    <th>音质 B</th>
    <th>偏好</th>
</tr>
</thead>
<tbody>
"""
    for s in samples:
        html += f"""<tr>
    <td>{s['id']}</td>
    <td>{s['genre']}</td>
    <td><audio controls src="{audio_src(s['audio_a'])}"></audio></td>
    <td class="rating"><input type="number" min="1" max="5"></td>
    <td class="rating"><input type="number" min="1" max="5"></td>
    <td class="rating"><input type="number" min="1" max="5"></td>
    <td><audio controls src="{audio_src(s['audio_b'])}"></audio></td>
    <td class="rating"><input type="number" min="1" max="5"></td>
    <td class="rating"><input type="number" min="1" max="5"></td>
    <td class="rating"><input type="number" min="1" max="5"></td>
    <td><select><option value="">--</option><option value="A">A</option><option value="B">B</option></select></td>
</tr>
"""
    html += """</tbody>
</table>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"听测表格已生成 → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成主观听测表格")
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--lora-dir", type=Path, default=Path("outputs/lora"))
    parser.add_argument("--output", type=Path, default=Path("reports/listening_test.html"))
    parser.add_argument("--mapping-output", type=Path, default=Path("reports/listening_test_mapping.json"))
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = collect_samples(args.baseline_dir, args.lora_dir, args.n_samples, args.seed)
    print(f"采样 {len(samples)} 对音频")
    randomized, mapping = randomize_ab(samples, args.seed)
    generate_html_table(randomized, args.output)

    args.mapping_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.mapping_output, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"盲听映射已保存 → {args.mapping_output}")


if __name__ == "__main__":
    main()
