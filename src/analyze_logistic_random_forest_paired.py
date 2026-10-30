import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


RESULT_DIR = Path("cradle_dialogue/results/logistic_vs_random_forest_complete")
PREDICTION_DIR = RESULT_DIR / "predictions/high_recall_confirm"
MODELS = ["logistic_regression", "random_forest"]
SPLITS = ["validation", "test"]
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260729


def load_predictions(model, split):
    path = PREDICTION_DIR / f"{model}__light_safety__{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def metric_values(frame):
    category_columns = [
        column for column in frame.columns if column.startswith("target_category_")
    ]
    high_risk = frame["high_risk_target"].eq(1)
    low_risk = (
        frame["target_alert"].eq(0)
        & frame["target_confirm"].eq(0)
        & frame[category_columns].sum(axis=1).eq(0)
    )
    return {
        "unsafe_continuation": float(frame.loc[high_risk, "policy_level"].eq(0).mean()),
        "over_escalation": float(frame.loc[low_risk, "policy_level"].isin([2, 3]).mean()),
        "macro_f1": float(
            f1_score(
                frame["policy_target"],
                frame["policy_level"],
                labels=[0, 1, 2, 3],
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(frame["policy_target"].eq(frame["policy_level"]).mean()),
        "high_risk_count": int(high_risk.sum()),
        "unsafe_count": int((high_risk & frame["policy_level"].eq(0)).sum()),
        "low_risk_count": int(low_risk.sum()),
        "over_escalation_count": int(
            (low_risk & frame["policy_level"].isin([2, 3])).sum()
        ),
    }


def aligned_frames(split):
    frames = {model: load_predictions(model, split) for model in MODELS}
    key_columns = ["dialogue_id", "turn_id"]
    reference = frames[MODELS[0]][key_columns]
    for model in MODELS[1:]:
        if not reference.equals(frames[model][key_columns]):
            raise ValueError(f"Prediction rows are not aligned for {split}")
    return frames


def bootstrap_differences(frames):
    reference = frames[MODELS[0]]
    dialogue_ids = reference["dialogue_id"].drop_duplicates().to_numpy()
    group_indices = {
        dialogue_id: np.flatnonzero(reference["dialogue_id"].to_numpy() == dialogue_id)
        for dialogue_id in dialogue_ids
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = {
        "unsafe_continuation": [],
        "over_escalation": [],
        "macro_f1": [],
        "accuracy": [],
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_dialogues = rng.choice(dialogue_ids, size=len(dialogue_ids), replace=True)
        indices = np.concatenate([group_indices[dialogue_id] for dialogue_id in sampled_dialogues])
        sampled = {
            model: frame.iloc[indices].reset_index(drop=True)
            for model, frame in frames.items()
        }
        values = {model: metric_values(frame) for model, frame in sampled.items()}
        for metric in differences:
            differences[metric].append(
                values["logistic_regression"][metric] - values["random_forest"][metric]
            )
    return {
        metric: {
            "difference_logistic_minus_random_forest": float(np.mean(values)),
            "ci_95_lower": float(np.quantile(values, 0.025)),
            "ci_95_upper": float(np.quantile(values, 0.975)),
        }
        for metric, values in differences.items()
    }


def main():
    report = {
        "bootstrap_method": "paired cluster bootstrap over dialogue_id",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "splits": {},
    }
    for split in SPLITS:
        frames = aligned_frames(split)
        report["splits"][split] = {
            "point_estimates": {
                model: metric_values(frame) for model, frame in frames.items()
            },
            "paired_bootstrap": bootstrap_differences(frames),
        }

    (RESULT_DIR / "paired_statistical_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    test = report["splits"]["test"]
    validation = report["splits"]["validation"]
    test_lr = test["point_estimates"]["logistic_regression"]
    test_rf = test["point_estimates"]["random_forest"]
    test_boot = test["paired_bootstrap"]
    lines = [
        "# Paired statistical analysis",
        "",
        f"Uncertainty was estimated with {BOOTSTRAP_REPLICATES:,} paired cluster-bootstrap replicates over dialogues. Differences below are logistic regression minus random forest; negative unsafe-continuation differences favor logistic regression, while positive over-escalation differences indicate additional burden.",
        "",
        "## Test split",
        "",
        f"- Unsafe continuation: logistic regression {test_lr['unsafe_continuation']:.4f} ({test_lr['unsafe_count']}/{test_lr['high_risk_count']}) versus random forest {test_rf['unsafe_continuation']:.4f} ({test_rf['unsafe_count']}/{test_rf['high_risk_count']}). Difference {test_boot['unsafe_continuation']['difference_logistic_minus_random_forest']:.4f}, 95% bootstrap CI [{test_boot['unsafe_continuation']['ci_95_lower']:.4f}, {test_boot['unsafe_continuation']['ci_95_upper']:.4f}].",
        f"- Over-escalation: logistic regression {test_lr['over_escalation']:.4f} ({test_lr['over_escalation_count']}/{test_lr['low_risk_count']}) versus random forest {test_rf['over_escalation']:.4f} ({test_rf['over_escalation_count']}/{test_rf['low_risk_count']}). Difference {test_boot['over_escalation']['difference_logistic_minus_random_forest']:.4f}, 95% bootstrap CI [{test_boot['over_escalation']['ci_95_lower']:.4f}, {test_boot['over_escalation']['ci_95_upper']:.4f}].",
        f"- Macro-F1 difference: {test_boot['macro_f1']['difference_logistic_minus_random_forest']:.4f}, 95% bootstrap CI [{test_boot['macro_f1']['ci_95_lower']:.4f}, {test_boot['macro_f1']['ci_95_upper']:.4f}].",
        "",
        "## Validation split",
        "",
        f"- Unsafe continuation: logistic regression {validation['point_estimates']['logistic_regression']['unsafe_continuation']:.4f} versus random forest {validation['point_estimates']['random_forest']['unsafe_continuation']:.4f}.",
        f"- Over-escalation: logistic regression {validation['point_estimates']['logistic_regression']['over_escalation']:.4f} versus random forest {validation['point_estimates']['random_forest']['over_escalation']:.4f}.",
        "",
        "## Decision interpretation",
        "",
        "Logistic regression provides the lower unsafe-continuation rate on both validation and test under the matched final guardrail. Random forest provides lower over-escalation and slightly higher Macro-F1. Choosing between them therefore requires an explicit safety-versus-burden objective; the result does not support claiming that random forest is uniformly superior.",
    ]
    (RESULT_DIR / "paired_statistical_analysis.md").write_text("\n".join(lines) + "\n")
    print(f"Saved paired analysis to {RESULT_DIR}")


if __name__ == "__main__":
    main()
