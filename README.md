# MusicGen-LoRA-Lab

基于 MusicGen-Melody 的音乐音频续写与 LoRA 泛化能力研究

## Quick Start

### 安装 uv

本项目现在使用 `uv` 管理 Python 环境和依赖。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果平台已经有 `uv`，可以直接跳过。

### 一键启动（推荐）

先跑小样本 smoke test，确认环境、模型下载、生成和 LoRA 训练接口都能工作：

```bash
./start.sh smoke
```

正式实验再放大：

```bash
./start.sh full
```

如果只想先跑预训练基线：

```bash
./start.sh baseline
```

可以通过环境变量覆盖默认配置：

```bash
MODEL_NAME=facebook/musicgen-small SAMPLES_PER_GENRE=5 EPOCHS=2 ./start.sh full
```

### 手动分步运行（uv）

```bash
# 1. 同步依赖
uv sync

# 2. 预处理 GTZAN 数据
uv run python scripts/preprocess_gtzan.py

# 3. 跑预训练基线
uv run python scripts/generate_baseline.py

# 4. LoRA 微调
uv run python scripts/train_lora.py --epochs 3 --batch-size 1 --amp

# 5. LoRA 模型生成
uv run python scripts/generate_lora.py --adapter-path outputs/lora/lora_adapter

# 6. 评估
uv run python scripts/evaluate_audio.py \
    --generated-dirs outputs/baseline outputs/lora

# 7. 生成听测表格
uv run python scripts/make_listening_test.py
```

## 项目结构

```
Musicgen-LoRA-Lab/
├── MusicGen-LoRA-完整方案.md    # 完整研究方案
├── 作业思路.docx
├── README.md
├── pyproject.toml                 # uv 项目配置
├── uv.lock                        # uv 锁文件
├── .python-version                # uv 默认 Python 版本
├── start.sh                       # 一键启动脚本
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── gtzan/              # GTZAN 原始音频（10 流派 × 100 条）
│   │   └── external/           # 外部测试音频
│   └── processed/
│       ├── train/              # 训练集 (70%)
│       ├── val/                # 验证集 (15%)
│       └── test/               # 测试集 (15%)
│
├── scripts/
│   ├── preprocess_gtzan.py     # 数据预处理
│   ├── generate_baseline.py    # 预训练基线生成
│   ├── train_lora.py           # LoRA 微调
│   ├── generate_lora.py        # LoRA 模型生成
│   ├── evaluate_audio.py       # 客观指标评估
│   ├── make_listening_test.py  # 主观听测表格
│   └── run_platform.sh         # 实验流水线脚本
│
├── outputs/
│   ├── baseline/               # 基线生成结果
│   ├── lora/                   # LoRA 生成结果 + adapter 权重
│   ├── external/               # 外部音频生成结果
│   └── metrics/                # 评估指标 JSON
│
├── reports/
│   ├── figures/                # 图表
│   └── final_report.md
│
└── archived/                   # 旧项目（仅参考，不参与新流程）
```

## 实验设计

| 实验 | 目的 | 输入 → 输出 |
|------|------|-------------|
| 预训练基线 | 建立基线性能 | 前 20s → 后 10s |
| LoRA 微调 | 验证 LoRA 增强效果 | 前 20s → 后 10s |
| 外部泛化 | 测试鲁棒性 | 非 GTZAN 音频前 20s → 后 10s |

## 评估指标

**客观：** Mel-spectrogram MSE、Chroma Similarity、Transition Smoothness、FAD（可选）

**主观：** 5-10 人盲听，连贯性 / 风格一致性 / 音质 / 偏好，1-5 分

## 平台运行注意事项

- 依赖以 `pyproject.toml` 为准，`requirements.txt` 仅作为传统 pip 环境备用。
- 默认 Python 版本为 `3.10`，可用 `PYTHON_VERSION=3.11 ./start.sh smoke` 覆盖。
- `data/raw/gtzan/` 已按 GTZAN 流派结构放置，脚本会自动排除 macOS 的 `._*` 文件。
- `generate_baseline.py` 和 `generate_lora.py` 使用前 `20s` 作为 prompt，MusicGen 总生成时长设为 `30s`，最终只保存后 `10s` 续写片段。
- `evaluate_audio.py` 会把原始音频的 `20s-30s` 作为真实目标，不会误用前 `10s`。
- `train_lora.py` 对 MusicGen 的 LM 部分应用 PEFT LoRA，并保存到 `outputs/lora/lora_adapter/`。
- 显存不足时优先用 `MODEL_NAME=facebook/musicgen-small`、`BATCH_SIZE=1`、`MAX_TRAIN_SAMPLES=20` 跑通闭环。
