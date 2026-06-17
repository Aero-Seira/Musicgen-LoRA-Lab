# MusicGen-LoRA-Lab

基于 MusicGen-Melody 的音乐音频续写与 LoRA 泛化能力研究

如果是第一次阅读项目，建议先看：[项目通俗说明.md](项目通俗说明.md)。

## Quick Start

### 安装 uv

本项目现在使用 `uv` 管理 Python 环境和依赖。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果平台已经有 `uv`，可以直接跳过。

### 一键启动（推荐）

5090 32G 平台建议直接使用项目目标模型 `facebook/musicgen-melody`。先跑小样本 smoke test，确认环境、模型下载、生成和 LoRA 训练接口都能工作：

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

OpenBayes 数据集绑定到 `/openbayes/input/input0` 时无需手动复制数据，脚本会自动优先识别该路径。也可以显式指定：

```bash
RAW_GTZAN_DIR=/openbayes/input/input0 ./start.sh full
```

### 一键推理与测评

已有 LoRA adapter 后，可以不重新训练，直接运行推理与测评脚本：

```bash
./infer_eval.sh compare
```

默认会执行：

1. 同步 `uv` 环境。
2. 若 `data/processed/test` 不存在，自动预处理 GTZAN。
3. 生成 baseline 续写到 `outputs/infer/baseline/`。
4. 加载 `outputs/lora/lora_adapter/` 生成 LoRA 续写到 `outputs/infer/lora/`。
5. 生成测评结果 `outputs/metrics/infer_eval.json` 和 `outputs/metrics/infer_eval.csv`。
6. 生成盲听页 `reports/infer_listening_test.html`。

常用模式：

```bash
./infer_eval.sh lora       # 只跑 LoRA 推理和测评
./infer_eval.sh baseline   # 只跑预训练基线推理和测评
./infer_eval.sh evaluate   # 只评估已有 outputs/infer 结果
./infer_eval.sh external   # 对 data/raw/external 下的 wav 做 LoRA 推理
```

常用覆盖参数：

```bash
DEVICE=cuda SAMPLES_PER_GENRE=3 ./infer_eval.sh compare
ADAPTER_PATH=outputs/lora/lora_adapter ./infer_eval.sh lora
EXTERNAL_DIR=data/raw/external EXTERNAL_OUTPUT_DIR=outputs/demo ./infer_eval.sh external
BASELINE_OUTPUT_DIR=outputs/baseline LORA_OUTPUT_DIR=outputs/lora ./infer_eval.sh evaluate
MAKE_LISTENING=0 ./infer_eval.sh compare
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
├── infer_eval.sh                  # 一键推理与测评脚本
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
- 针对 5090 32G，`uv.lock` 已解析到 PyTorch 2.8 / CUDA 12.8 相关 wheel，避免旧 PyTorch 2.1 对 Blackwell 支持不足的问题。
- `xformers` 已固定为 `0.0.32.post2`，匹配 PyTorch 2.8；`start.sh` 会强制按锁文件重装 `xformers`，避免误装到面向 PyTorch 2.10 的 wheel。
- OpenBayes 上会优先从 `/openbayes/input/input0` 读取 GTZAN；本地则默认读取 `data/raw/gtzan/`。
- 数据目录可以直接包含 `blues/classical/.../rock`，也可以包一层 `gtzan/` 或 `genres_original/`。
- 脚本会自动排除 macOS 的 `._*` 文件。
- GTZAN 常见损坏样本如 `jazz.00054.wav` 会在预处理阶段自动跳过，并记录到 `data/processed/skipped_bad_audio.csv`。
- `generate_baseline.py` 和 `generate_lora.py` 使用前 `20s` 作为 prompt，MusicGen 总生成时长设为 `30s`，最终只保存后 `10s` 续写片段。
- `evaluate_audio.py` 会把原始音频的 `20s-30s` 作为真实目标，不会误用前 `10s`。
- `train_lora.py` 对 MusicGen 的 LM 部分应用 PEFT LoRA，并保存到 `outputs/lora/lora_adapter/`。
- 5090 32G 默认使用 `facebook/musicgen-melody`；显存不足时再降级为 `MODEL_NAME=facebook/musicgen-small`、`BATCH_SIZE=1`、`MAX_TRAIN_SAMPLES=20`。
