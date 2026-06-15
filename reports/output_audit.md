# Output Audit

## Summary

Audit target: `outputs/`

Status: usable, with caveats.

- Baseline generations: 100 wav files, 10 genres x 10 samples.
- LoRA generations: 100 wav files, 10 genres x 10 samples.
- Audio format check: all 200 generated wav files are mono, 32000 Hz, exactly 10.0 seconds.
- LoRA adapter exists: `outputs/lora/lora_adapter/adapter_model.safetensors`, about 27 MB.
- Metrics exist:
  - `outputs/metrics/baseline_results.json`
  - `outputs/metrics/baseline_results.csv`
  - `outputs/metrics/baseline_vs_lora.json`
  - `outputs/metrics/baseline_vs_lora.csv`

## Training Audit

LoRA training config:

- Base model: `facebook/musicgen-melody`
- LoRA rank: 8
- LoRA alpha: 16
- Dropout: 0.05
- Epochs: 3
- Learning rate: 1e-4
- Batch size: 1
- Target modules: `linear1`, `linear2`, `out_proj`

Training log:

| Epoch | Train Loss | Val Loss |
| ---: | ---: | ---: |
| 1 | 4.2983 | 4.3452 |
| 2 | 4.2716 | 4.3499 |
| 3 | 4.2417 | 4.3618 |

Interpretation: training loss decreases, but validation loss rises slightly. This suggests mild overfitting or limited generalization after 3 epochs.

## Metric Audit

Overall comparison:

| Metric | Baseline | LoRA | Direction | Result |
| --- | ---: | ---: | --- | --- |
| Mel MSE | 142.6694 | 130.8578 | lower is better | LoRA better by 8.28% |
| Chroma Similarity | 0.7061 | 0.7002 | higher is better | Baseline slightly better |
| Smoothness | 136.1970 | 139.1970 | lower is better | Baseline slightly better |

Paired sample comparison, 100 matched pairs:

| Metric | LoRA Better Pairs | Mean Delta (LoRA - Baseline) | Median Delta |
| --- | ---: | ---: | ---: |
| Mel MSE | 52 / 100 | -11.8116 | -0.4306 |
| Chroma Similarity | 42 / 100 | -0.0059 | -0.0114 |
| Smoothness | 49 / 100 | 2.9999 | 1.3141 |

Interpretation: LoRA improves Mel MSE overall, but the paired gain is modest and not consistent across all samples. It does not improve Chroma Similarity or transition smoothness overall.

## Genre-Level Findings

LoRA improves Mel MSE for:

- country
- disco
- jazz
- pop
- reggae

LoRA worsens Mel MSE for:

- blues
- classical
- hiphop
- metal
- rock

LoRA improves all three tracked metrics only for:

- jazz
- reggae

LoRA is clearly weaker for:

- classical
- rock

## Notable Outliers

Highest baseline Mel MSE:

- `disco.00023_baseline.wav`: 972.4471
- `reggae.00051_baseline.wav`: 852.4286
- `reggae.00057_baseline.wav`: 599.9603

Highest LoRA Mel MSE:

- `hiphop.00014_lora.wav`: 476.6928
- `blues.00031_lora.wav`: 428.8418
- `rock.00022_lora.wav`: 393.7738

Worst transition smoothness:

- Baseline: `hiphop.00004_baseline.wav`: 392.6338
- LoRA: `rock.00009_lora.wav`: 439.1406

These files should be reviewed by listening before drawing strong conclusions.

## Deliverability Assessment

The outputs are sufficient for the project report:

1. The baseline and LoRA experiments both completed.
2. There are matched 100-sample outputs for objective comparison.
3. The LoRA adapter was saved successfully.
4. Metrics are available in both JSON and CSV.
5. Audio files have correct technical format.

However, the current result should be framed carefully:

- Do not claim LoRA is universally better.
- A defensible conclusion is: LoRA reduced overall Mel-spectrogram distance, but gains are genre-dependent and do not clearly improve chroma similarity or transition smoothness.
- The validation loss trend suggests additional tuning is needed before claiming strong generalization.

## Recommended Next Steps

1. Generate `reports/listening_test.html` if not already produced.
2. Conduct a small blind listening test on 10-20 matched pairs.
3. Include median metrics in the report, not only mean metrics, because several outliers strongly affect the averages.
4. Highlight jazz and reggae as positive cases; discuss classical and rock as weaker cases.
5. If rerunning training, try fewer epochs, lower learning rate, or stronger dropout.
