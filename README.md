# Mental-Health Chatbot Safety Policy Research

Research code and aggregate results for a modular mental-health conversation safety pipeline. The project combines Alert, Confirm, and risk-category predictions with causal progression features and a four-level escalation policy.

> **Research use only.** This repository is not a clinical device, crisis service, diagnostic system, or production-ready safety system. Its policy targets are label-derived research proxies rather than clinical escalation ground truth.

## What is included

- Model-training and evaluation code in [`src/`](src/)
- Aggregate results in [`results/`](results/)
- Paper-supporting methods and audits in [`paper/`](paper/)
- Publication tables in [`tables/`](tables/)
- Publication figures in [`figures/`](figures/)
- Reproduction notes and the presentation in [`docs/`](docs/)
- Dataset access and privacy guidance in [`data/README.md`](data/README.md)

Raw conversations, model checkpoints, embeddings, detailed prediction exports, and identifiable error-analysis examples are deliberately excluded.

## Main research pipeline

1. A DeBERTa-v3-large encoder produces current-turn risk representations.
2. A temporal Alert model uses an eight-turn history window.
3. The validation-selected `B_pos_weight_x2` Confirm model emphasizes recall.
4. A current-turn category model estimates seven risk-category probabilities.
5. Causal progression features summarize changes over the dialogue.
6. A learned four-level policy is combined with explicit safety guardrails.

The four output levels are:

- Level 0: continue the conversation
- Level 1: offer supportive resources
- Level 2: defer to human support
- Level 3: urgent escalation

## Key findings

### Encoder selection

DeBERTa-v3-large was selected from the severity benchmark because it had the strongest safety-oriented five-seed profile:

- mean test Macro-F1: **0.9670**
- mean high-risk recall: **0.9994**
- one estimated high-risk miss across five seeds

This was a three-class severity benchmark, not an Alert/Confirm encoder ablation.

### High-recall Confirm model

The validation-selected `B_pos_weight_x2` Confirm model reduced held-out unsafe continuation in the original selected policy from **29.79% to 22.98%**, with over-escalation changing from **11.90% to 12.24%**.

### Logistic-regression reassessment

A later matched comparison used the same high-recall Confirm features and the same `light_safety` guardrail for both policy models:

| Test metric | Logistic regression | Random forest |
| --- | ---: | ---: |
| High-risk unsafe continuation | **18.72% (88/470)** | 22.98% (108/470) |
| Over-escalation | 16.03% | **12.24%** |
| Macro-F1 | 0.3769 | **0.3870** |
| Accuracy | 0.6097 | **0.6477** |

Logistic regression is therefore the stronger **safety-first research candidate**, while random forest produces fewer unnecessary high-level escalations and slightly better overall classification metrics. See the [complete decision note](results/logistic_vs_random_forest/final_decision.md).

## Installation

Python 3.10 was used during the final audit environment. Create a virtual environment and install the approximate dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Exact historical dependency versions were not preserved, so `requirements.txt` describes the required packages without claiming an exact environment reconstruction.

## Running the project

The original scripts use paths relative to the former `bert-pretrained/` working directory. From this public repository, run scripts from `src/` or adapt the paths through your own data configuration.

The reconstructed experiment sequence is documented in [`docs/run_commands.md`](docs/run_commands.md). The policy-only reassessment can be reproduced after preparing the required progression-feature CSVs:

```bash
cd src
python compare_logistic_random_forest_complete.py
python analyze_logistic_random_forest_paired.py
```

The repository does not include those feature CSVs because they may contain sensitive or redistribution-restricted source material.

## Data

No raw mental-health conversations are distributed here. Before using CRADLE-Dialogue, AnnoMI, or another dataset, independently verify:

- its license and redistribution terms;
- consent and intended-use restrictions;
- privacy and de-identification requirements; and
- whether your institution requires ethics review.

See [`data/README.md`](data/README.md) for the expected inputs.

## Important limitations

- Train and validation data are described as synthetic/structured; the test set is described as clinician annotated.
- Most final CRADLE components were evaluated with one upstream model seed.
- The broader exploratory project inspected test reports, so results should be described as held-out evaluations rather than a pristine one-time test.
- The final policy still produced substantial unsafe continuation.
- Aggregate dataset results do not demonstrate clinical effectiveness or real-world safety.
- External validation, subgroup analysis, calibration, human review, and prospective studies remain necessary.

## License

No open-source license has been selected yet. Until a license is added, the code is publicly viewable but normal copyright restrictions remain. See [`LICENSE-CHOICES.md`](LICENSE-CHOICES.md) before inviting others to reuse or modify the code.

## Responsible disclosure

Please do not open a public issue containing private conversations, personally identifying information, or crisis disclosures. Use a private contact channel supplied by the repository owner instead.

