# Complete logistic-regression vs random-forest policy comparison

The primary matched comparison uses the same high-recall Confirm progression features and the same `light_safety` guardrail for both base models. Model and guardrail fitting use training/validation data; test metrics are reported for evaluation and are not used to choose a winner.

## Matched final-pipeline comparison

| Model | Split | Unsafe continuation | Over-escalation | Macro-F1 | Accuracy | Urgent precision | Human-defer precision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | test | 0.1872 | 0.1603 | 0.3769 | 0.6097 | 0.7637 | 0.3621 |
| random_forest | test | 0.2298 | 0.1224 | 0.3870 | 0.6477 | 0.7735 | 0.3946 |
| logistic_regression | validation | 0.0070 | 0.1225 | 0.4045 | 0.6336 | 0.8248 | 0.5119 |
| random_forest | validation | 0.0210 | 0.1202 | 0.4087 | 0.6444 | 0.8400 | 0.5174 |

## Interpretation rule

The validation split remains the basis for model selection. The test split is used only to characterize held-out generalization. A lower unsafe-continuation rate is preferable, while over-escalation quantifies the corresponding burden.

## Reproducibility

- Five policy seeds were run. Logistic regression is deterministic here; random-forest variability is summarized separately.
- Both old-Confirm and high-recall-Confirm feature sets were evaluated.
- Both unconstrained models, the matched `light_safety` guardrail, and every existing guardrail variant were evaluated.
- Prediction exports intentionally omit raw dialogue text.
