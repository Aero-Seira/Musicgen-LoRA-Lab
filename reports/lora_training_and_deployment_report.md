# MusicGen-LoRA 训练结果与部署使用报告

## 1. 本部分任务说明

本部分围绕 MusicGen-Melody 音乐续写任务完成实验。任务形式为：给定一段音乐的前 20 秒作为条件输入，模型生成后续内容，最终保存 10 秒续写片段。实验比较两类模型输出：

- 预训练基线模型：直接使用 `facebook/musicgen-melody` 生成续写结果。
- LoRA 微调模型：在 `facebook/musicgen-melody` 的语言模型部分加载 LoRA adapter 后生成续写结果。

本部分的主要目标不是重新训练一个完整 MusicGen，而是在有限算力和 GTZAN 数据集规模下，验证 LoRA 参数高效微调是否能改善音乐续写结果，并整理可复现的训练、评估和部署推理方法。

## 2. 环境与项目管理

项目使用 `uv` 管理 Python 环境和依赖，核心配置文件如下：

| 文件 | 作用 |
| --- | --- |
| `pyproject.toml` | Python 项目配置、依赖声明、PyTorch CUDA 版本约束 |
| `uv.lock` | 锁定后的依赖版本 |
| `.python-version` | 默认 Python 版本 |
| `start.sh` | 一键实验启动脚本 |
| `scripts/run_platform.sh` | 平台端实验流水线 |

本实验在算力平台上使用 5090 32G 显存配置，优先采用完整的 `facebook/musicgen-melody`，而不是降级到 `facebook/musicgen-small`。项目依赖中将 PyTorch 约束到 `>=2.7,<2.9`，并在 Linux CUDA 环境中使用 CUDA 12.8 wheel，以适配新一代显卡环境。

在 OpenBayes 平台上，GTZAN 数据集绑定到：

```bash
/openbayes/input/input0
```

脚本会自动识别常见 GTZAN 目录结构，例如：

```text
/openbayes/input/input0/genres_original/
/openbayes/input/input0/gtzan/genres_original/
/openbayes/input/input0/blues/
```

## 3. 数据与实验流程

### 3.1 数据预处理

预处理脚本为：

```bash
uv run python scripts/preprocess_gtzan.py
```

预处理逻辑包括：

- 自动扫描 GTZAN 原始音频目录。
- 排除 macOS 元数据文件，例如 `._*`。
- 将音频统一处理为 32000 Hz。
- 将每条音频统一到 30 秒长度。
- 按流派划分训练集、验证集和测试集。
- 跳过 GTZAN 中无法被解码的损坏音频文件。

本次实验划分比例为：

| Split | 比例 | 用途 |
| --- | ---: | --- |
| Train | 70% | LoRA 微调 |
| Val | 15% | 训练期间验证 |
| Test | 15% | 基线与 LoRA 生成、客观评估 |

### 3.2 完整实验流水线

一键运行方式：

```bash
./start.sh full
```

该命令会依次执行：

1. 同步 `uv` 依赖环境。
2. 预处理 GTZAN 数据。
3. 使用预训练 MusicGen-Melody 生成 baseline 结果。
4. 对 MusicGen-Melody 的 LM 部分进行 LoRA 微调。
5. 加载 LoRA adapter 生成续写结果。
6. 计算客观指标。
7. 生成可供听测整理的辅助文件。

调试环境时可先运行小样本：

```bash
./start.sh smoke
```

只生成预训练基线时可运行：

```bash
./start.sh baseline
```

## 4. 训练配置

本次 LoRA 微调配置记录在 `outputs/lora/train_config.json`：

| 配置项 | 值 |
| --- | --- |
| Base model | `facebook/musicgen-melody` |
| Sample rate | 32000 Hz |
| Audio duration | 30 s |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `linear1`, `linear2`, `out_proj` |
| Epochs | 3 |
| Learning rate | 1e-4 |
| Batch size | 1 |

LoRA adapter 的配置记录在 `outputs/lora/lora_adapter/adapter_config.json`。本次微调只保存 adapter 参数，而不是保存完整 MusicGen 模型，因此权重体积较小，便于提交、迁移和部署。

## 5. 训练过程与损失变化

训练日志保存在 `outputs/lora/training_log.json`：

| Epoch | Train Loss | Val Loss |
| ---: | ---: | ---: |
| 1 | 4.2983 | 4.3452 |
| 2 | 4.2716 | 4.3499 |
| 3 | 4.2417 | 4.3618 |

从日志可以看出，训练集 loss 从 4.2983 下降到 4.2417，说明 LoRA adapter 确实学习到了训练集中的一部分分布特征。但验证集 loss 从 4.3452 小幅上升到 4.3618，说明模型在 3 个 epoch 后已经出现轻微过拟合或泛化收益不足。

因此，本次结果应表述为：LoRA 微调对部分客观指标和部分流派有改善，但不能证明 LoRA 在所有音乐类型上都优于预训练基线。

## 6. `outputs/` 目录内容说明

本次实验输出位于 `outputs/`，总体大小约 311 MB。

| 路径 | 大小 | 内容 |
| --- | ---: | --- |
| `outputs/baseline/` | 约 140 MB | 预训练模型生成的续写音频 |
| `outputs/lora/` | 约 169 MB | LoRA 模型生成音频、训练日志、adapter |
| `outputs/lora/lora_adapter/` | 约 28 MB | 可部署的 LoRA adapter 权重 |
| `outputs/metrics/` | 约 1.1 MB | 客观评估指标 JSON/CSV |
| `outputs/external/` | 很小 | 外部音频泛化测试输出目录，本次基本为空 |

### 6.1 Baseline 输出

`outputs/baseline/` 中包含 100 条 baseline 续写音频，覆盖 10 个流派，每个流派 10 条样本。文件命名格式为：

```text
{原始样本名}_baseline.wav
```

示例：

```text
outputs/baseline/blues/blues.00003_baseline.wav
outputs/baseline/jazz/jazz.00015_baseline.wav
```

对应清单文件为：

```text
outputs/baseline/manifest.jsonl
```

该文件共 100 行，每行记录一个生成样本的来源和输出路径。

### 6.2 LoRA 输出

`outputs/lora/` 中包含 100 条 LoRA 续写音频，同样覆盖 10 个流派，每个流派 10 条样本。文件命名格式为：

```text
{原始样本名}_lora.wav
```

示例：

```text
outputs/lora/blues/blues.00003_lora.wav
outputs/lora/jazz/jazz.00015_lora.wav
```

对应清单文件为：

```text
outputs/lora/manifest_lora.jsonl
```

该文件共 100 行，每行记录生成样本、原始测试音频和 adapter 路径。

### 6.3 LoRA adapter

可用于部署推理的核心文件位于：

```text
outputs/lora/lora_adapter/
```

主要文件包括：

| 文件 | 作用 |
| --- | --- |
| `adapter_model.safetensors` | LoRA adapter 权重 |
| `adapter_config.json` | PEFT LoRA 配置 |
| `README.md` | adapter 元信息 |

部署时必须同时具备：

1. 基座模型：`facebook/musicgen-melody`
2. LoRA adapter：`outputs/lora/lora_adapter/`

单独的 adapter 不能独立推理，因为它只包含相对于基座模型的低秩增量参数。

### 6.4 评估指标文件

`outputs/metrics/` 中包含：

| 文件 | 作用 |
| --- | --- |
| `baseline_results.json` | baseline 单独评估结果 |
| `baseline_results.csv` | baseline 单独评估结果表格版 |
| `baseline_vs_lora.json` | baseline 与 LoRA 对比结果 |
| `baseline_vs_lora.csv` | baseline 与 LoRA 对比结果表格版 |

### 6.5 本地元数据文件

`outputs/` 中可见 `.DS_Store` 和 `._*` 文件。这些是 macOS 产生的本地元数据或资源叉文件，不属于实验结果。打包作业材料时建议排除这些文件。

可使用如下命令清理：

```bash
find outputs -name '.DS_Store' -delete
find outputs -name '._*' -delete
```

## 7. 生成音频格式检查

本次审计中检查了 baseline 与 LoRA 共 200 条生成音频：

| 项目 | 结果 |
| --- | --- |
| 音频数量 | baseline 100 条，LoRA 100 条 |
| 声道 | mono |
| 采样率 | 32000 Hz |
| 时长 | 10.0 s |
| 技术格式 | 均可正常读取 |

因此，从文件格式和可读取性角度看，本次生成结果可用于报告、展示和后续听测。

## 8. 客观指标结果

评估脚本为：

```bash
uv run python scripts/evaluate_audio.py \
  --generated-dirs outputs/baseline outputs/lora
```

评估时将原始测试音频的 20-30 秒片段作为参考目标，将模型生成的 10 秒续写片段与其对比。

本次使用的主要指标为：

| 指标 | 含义 | 趋势 |
| --- | --- | --- |
| Mel MSE | 生成音频与真实后续片段在 Mel 频谱上的均方误差 | 越低越好 |
| Chroma Similarity | 生成音频与真实后续片段的和声/音高类别相似度 | 越高越好 |
| Smoothness | 续写片段内部的过渡平滑度指标 | 越低越好 |

整体结果如下：

| Metric | Baseline | LoRA | 更优方向 | 结果 |
| --- | ---: | ---: | --- | --- |
| Mel MSE | 142.6694 | 130.8578 | 越低越好 | LoRA 改善约 8.28% |
| Chroma Similarity | 0.7061 | 0.7002 | 越高越好 | Baseline 略优 |
| Smoothness | 136.1970 | 139.1970 | 越低越好 | Baseline 略优 |

按 100 个匹配样本进行成对比较：

| Metric | LoRA 更优样本数 | 平均差值 LoRA - Baseline | 中位数差值 |
| --- | ---: | ---: | ---: |
| Mel MSE | 52 / 100 | -11.8116 | -0.4306 |
| Chroma Similarity | 42 / 100 | -0.0059 | -0.0114 |
| Smoothness | 49 / 100 | 2.9999 | 1.3141 |

该结果说明：LoRA 在 Mel MSE 的整体均值上有改进，但优势并不稳定；在 Chroma Similarity 和 Smoothness 上没有体现出整体优势。

## 9. 分流派分析

分流派结果如下：

| Genre | Baseline Mel MSE | LoRA Mel MSE | Mel MSE 变化 | 结论 |
| --- | ---: | ---: | ---: | --- |
| blues | 141.7770 | 189.9530 | +48.1760 | LoRA 变差 |
| classical | 166.7952 | 196.5751 | +29.7800 | LoRA 变差 |
| country | 92.3243 | 84.1891 | -8.1353 | LoRA 改善 |
| disco | 192.0507 | 82.6290 | -109.4217 | LoRA 改善明显 |
| hiphop | 138.3186 | 148.1262 | +9.8076 | LoRA 变差 |
| jazz | 154.7252 | 137.4150 | -17.3101 | LoRA 改善 |
| metal | 68.5502 | 99.2561 | +30.7059 | LoRA 变差 |
| pop | 96.5271 | 86.2215 | -10.3056 | LoRA 改善 |
| reggae | 250.6638 | 139.6748 | -110.9890 | LoRA 改善明显 |
| rock | 124.9618 | 144.5377 | +19.5759 | LoRA 变差 |

从流派角度看，LoRA 在以下流派的 Mel MSE 上有改善：

- country
- disco
- jazz
- pop
- reggae

LoRA 在以下流派的 Mel MSE 上变差：

- blues
- classical
- hiphop
- metal
- rock

其中 `jazz` 和 `reggae` 是较好的正向案例，因为它们不仅 Mel MSE 改善，Chroma Similarity 和 Smoothness 也同步改善。`classical` 和 `rock` 是较弱案例，三个指标整体表现都不理想。

## 10. 典型异常样本

部分样本的误差明显偏高，可能影响均值指标，应在报告中单独说明，避免只看平均值造成误判。

Baseline 中 Mel MSE 较高的样本：

| 文件 | Mel MSE |
| --- | ---: |
| `disco.00023_baseline.wav` | 972.4471 |
| `reggae.00051_baseline.wav` | 852.4286 |
| `reggae.00057_baseline.wav` | 599.9603 |

LoRA 中 Mel MSE 较高的样本：

| 文件 | Mel MSE |
| --- | ---: |
| `hiphop.00014_lora.wav` | 476.6928 |
| `blues.00031_lora.wav` | 428.8418 |
| `rock.00022_lora.wav` | 393.7738 |

Smoothness 最差样本：

| 模型 | 文件 | Smoothness |
| --- | --- | ---: |
| Baseline | `hiphop.00004_baseline.wav` | 392.6338 |
| LoRA | `rock.00009_lora.wav` | 439.1406 |

这些样本建议在小组展示或报告答辩前进行人工听感复核。

## 11. 实验结论

本次实验可以得到如下结论：

1. LoRA 微调流程已成功跑通，能够在 MusicGen-Melody 上保存并重新加载 adapter。
2. 训练 loss 持续下降，说明 adapter 学到了训练集分布。
3. LoRA 在整体 Mel MSE 上从 142.6694 降至 130.8578，约改善 8.28%。
4. LoRA 的改善具有流派依赖性，在 `jazz`、`reggae`、`disco` 等流派上更明显。
5. LoRA 没有在 Chroma Similarity 和 Smoothness 上取得整体优势。
6. 验证集 loss 小幅上升，说明当前 3 epoch 设置可能已经出现轻微过拟合。

因此，较严谨的结论是：

> LoRA 微调在本次 GTZAN 音乐续写任务中改善了整体 Mel 频谱距离，说明其对局部音色或频谱分布有一定适配作用；但该收益在不同流派间不均衡，且未稳定提升和声相似度与过渡平滑度。因此，本次 LoRA 结果可作为有效的参数高效微调实验结果，但不能宣称其全面优于预训练 MusicGen-Melody。

## 12. 局限性与后续改进

本次实验仍存在以下局限：

- GTZAN 数据集规模较小，每个流派样本有限。
- 训练只进行了 3 个 epoch，且 batch size 为 1，训练稳定性有限。
- 验证集 loss 上升，说明当前超参数仍需调整。
- 只做了客观指标评估，尚未完成正式多人盲听。
- 外部音频泛化测试目录基本为空，暂不能支撑跨数据集泛化结论。
- MusicGen 的音乐生成质量很难只靠 Mel MSE、Chroma 和 Smoothness 完整衡量。

后续可改进方向：

- 减少 epoch 或引入 early stopping。
- 降低学习率，例如从 `1e-4` 调整为 `5e-5`。
- 提高 LoRA dropout，例如从 `0.05` 调整到 `0.1`。
- 针对表现较弱的 `classical`、`rock` 等流派单独分析。
- 增加外部音乐样本测试。
- 组织 5-10 人盲听，比较连贯性、风格一致性、音质和主观偏好。

## 13. LoRA 模型部署推理方法

### 13.1 部署所需文件

部署 LoRA 推理时需要保留以下内容：

```text
pyproject.toml
uv.lock
.python-version
scripts/generate_lora.py
outputs/lora/lora_adapter/
```

如果需要使用项目内测试集批量生成，还需要：

```text
data/processed/test/
```

如果只对外部音频做推理，可以不带 GTZAN 测试集，只准备待续写的 wav 文件。

### 13.2 一键推理与测评脚本

项目提供了面向部署复验的脚本：

```bash
./infer_eval.sh compare
```

该脚本不会重新训练 LoRA，而是直接复用已有 adapter。默认流程包括：

1. 使用 `uv` 同步依赖环境。
2. 若 `data/processed/test` 不存在，自动从 GTZAN 原始目录预处理测试集。
3. 使用预训练 MusicGen-Melody 生成 baseline 到 `outputs/infer/baseline/`。
4. 加载 `outputs/lora/lora_adapter/` 生成 LoRA 结果到 `outputs/infer/lora/`。
5. 计算客观指标，输出到 `outputs/metrics/infer_eval.json` 和 `outputs/metrics/infer_eval.csv`。
6. 生成 A/B 听测页 `reports/infer_listening_test.html`。

常用模式如下：

```bash
./infer_eval.sh lora       # 只跑 LoRA 推理和测评
./infer_eval.sh baseline   # 只跑预训练基线推理和测评
./infer_eval.sh evaluate   # 只评估已有 outputs/infer 结果
./infer_eval.sh external   # 对 data/raw/external 下的 wav 做 LoRA 推理
```

常用参数覆盖方式：

```bash
DEVICE=cuda SAMPLES_PER_GENRE=3 ./infer_eval.sh compare
ADAPTER_PATH=outputs/lora/lora_adapter ./infer_eval.sh lora
EXTERNAL_DIR=data/raw/external EXTERNAL_OUTPUT_DIR=outputs/demo ./infer_eval.sh external
BASELINE_OUTPUT_DIR=outputs/baseline LORA_OUTPUT_DIR=outputs/lora ./infer_eval.sh evaluate
MAKE_LISTENING=0 ./infer_eval.sh compare
```

### 13.3 安装环境

在项目根目录运行：

```bash
uv sync --locked
```

如果在 OpenBayes 或类似 Linux GPU 平台上运行，并且需要强制使用指定 Python 版本：

```bash
uv sync --python 3.10 --locked
```

### 13.4 使用已训练 LoRA 对 GTZAN 测试集推理

运行：

```bash
uv run python scripts/generate_lora.py \
  --adapter-path outputs/lora/lora_adapter \
  --model-name facebook/musicgen-melody \
  --data-dir data/processed/test \
  --output-lora outputs/lora_infer \
  --samples-per-genre 10 \
  --device cuda
```

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--adapter-path` | LoRA adapter 路径 |
| `--model-name` | 基座模型名称，必须与训练时一致 |
| `--data-dir` | 待推理的测试集目录 |
| `--output-lora` | LoRA 推理输出目录 |
| `--samples-per-genre` | 每个流派生成样本数 |
| `--device` | 推理设备，可设为 `cuda`、`cpu` 或 `auto` |

输出文件会保存到：

```text
outputs/lora_infer/{genre}/{sample_name}_lora.wav
```

### 13.5 对外部音频推理

准备外部 wav 文件：

```bash
mkdir -p data/raw/external
```

将待续写音频放入：

```text
data/raw/external/
```

建议每条音频不少于 20 秒。若音频不足 20 秒，脚本会自动补零，但生成质量通常会下降。

运行：

```bash
uv run python scripts/generate_lora.py \
  --adapter-path outputs/lora/lora_adapter \
  --model-name facebook/musicgen-melody \
  --external-dir data/raw/external \
  --output-external outputs/external \
  --device cuda
```

输出文件会保存到：

```text
outputs/external/{source_stem}_lora.wav
```

### 13.6 控制生成长度和采样参数

默认设置为：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--input-sec` | 20 | 输入 prompt 秒数 |
| `--output-sec` | 10 | 保存的续写秒数 |
| `--top-k` | 250 | top-k 采样 |
| `--top-p` | 0.0 | nucleus sampling，0 表示不启用 |
| `--temperature` | 1.0 | 采样温度 |
| `--cfg-coef` | 3.0 | classifier-free guidance 强度 |

示例：

```bash
uv run python scripts/generate_lora.py \
  --adapter-path outputs/lora/lora_adapter \
  --external-dir data/raw/external \
  --output-external outputs/external_temp08 \
  --input-sec 20 \
  --output-sec 10 \
  --temperature 0.8 \
  --cfg-coef 3.0 \
  --device cuda
```

脚本内部会将 MusicGen 的生成总时长设置为：

```text
input_sec + output_sec
```

然后只截取最后 `output_sec` 秒作为最终续写结果。因此默认情况下，模型会根据前 20 秒条件生成总共 30 秒音频，最终保存后 10 秒。

### 13.7 Python 代码方式加载 LoRA 模型

如果不使用命令行脚本，也可以在 Python 中直接加载：

```python
from pathlib import Path

from audiocraft.models import MusicGen
from peft import PeftModel

device = "cuda"
model_name = "facebook/musicgen-melody"
adapter_path = Path("outputs/lora/lora_adapter")

model = MusicGen.get_pretrained(model_name, device=device)
model.lm = PeftModel.from_pretrained(model.lm, str(adapter_path))
model.lm.to(device)
model.lm.eval()
```

本项目的 `scripts/generate_lora.py` 已经封装了完整兼容逻辑，包括：

- 自动选择 `cuda`、`mps` 或 `cpu`。
- 加载 MusicGen 基座模型。
- 将 PEFT LoRA adapter 挂载到 `model.lm`。
- 暴露 AudioCraft 在 PEFT 包装后可能被隐藏的 `generate` 和 `compute_predictions` 方法。
- 自动重采样输入音频。
- 截取前 20 秒 prompt。
- 保存后 10 秒续写结果。

因此，实际部署时优先建议直接复用 `scripts/generate_lora.py`。

## 14. 提交材料建议

作为小组作业提交时，建议至少包含：

```text
reports/lora_training_and_deployment_report.md
reports/output_audit.md
pyproject.toml
uv.lock
start.sh
scripts/
outputs/lora/lora_adapter/
outputs/metrics/
```

如需展示音频结果，可额外打包：

```text
outputs/baseline/
outputs/lora/
```

但打包前建议删除 macOS 元数据文件：

```bash
find outputs -name '.DS_Store' -delete
find outputs -name '._*' -delete
```

如果提交平台限制文件大小，最小可提交版本应保留：

```text
outputs/lora/lora_adapter/
outputs/metrics/
reports/
```

音频样例可以只选择每个流派 1-2 组 baseline 与 LoRA 对照样本。

## 15. 可复现命令汇总

完整复现实验：

```bash
./start.sh full
```

只重新评估已有输出：

```bash
uv run python scripts/evaluate_audio.py \
  --generated-dirs outputs/baseline outputs/lora
```

使用 LoRA adapter 重新生成测试集结果：

```bash
uv run python scripts/generate_lora.py \
  --adapter-path outputs/lora/lora_adapter \
  --model-name facebook/musicgen-melody \
  --data-dir data/processed/test \
  --output-lora outputs/lora_infer \
  --samples-per-genre 10 \
  --device cuda
```

使用 LoRA adapter 对外部音频续写：

```bash
uv run python scripts/generate_lora.py \
  --adapter-path outputs/lora/lora_adapter \
  --model-name facebook/musicgen-melody \
  --external-dir data/raw/external \
  --output-external outputs/external \
  --device cuda
```

## 16. 最终表述建议

在小组报告中可以采用如下表述：

> 本部分基于 `facebook/musicgen-melody` 完成了 GTZAN 音乐续写实验，并使用 PEFT LoRA 对 MusicGen 的语言模型部分进行参数高效微调。实验生成了 100 条预训练基线续写音频和 100 条 LoRA 续写音频，所有输出均为 32000 Hz、mono、10 秒 wav 文件。客观评估显示，LoRA 将整体 Mel MSE 从 142.6694 降至 130.8578，约改善 8.28%；但 Chroma Similarity 和 Smoothness 未取得整体优势。分流派结果表明，LoRA 在 jazz、reggae、disco 等流派上效果较好，在 classical、rock 等流派上仍存在不足。因此，本实验验证了 LoRA 微调 MusicGen 的可行性和部分有效性，但仍需要更多数据、超参数调优和主观听测来支撑更强结论。
