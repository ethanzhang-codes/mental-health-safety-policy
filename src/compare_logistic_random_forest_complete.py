import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_cradle_learned_policy as learned_policy
import train_cradle_safety_constrained_policy_v2 as safety_policy


DATASETS = {
    "old_confirm": Path("cradle_dialogue/data/progression_features"),
    "high_recall_confirm": Path("cradle_dialogue/data/progression_features_high_recall_confirm"),
}
SPLIT_FILES = {
    "train": "cradle_train_progression_features.csv",
    "validation": "cradle_validation_progression_features.csv",
    "test": "cradle_test_progression_features.csv",
}
MODEL_NAMES = ["logistic_regression", "random_forest"]
GUARDRAIL_VARIANTS = [
    "strict_safety",
    "balanced_safety",
    "light_safety",
    "confirm_priority_safety",
    "user_custom_safety",
]
SEEDS = [1, 2, 3, 4, 5]
OUTPUT_ROOT = Path("cradle_dialogue/results/logistic_vs_random_forest_complete")


def set_model_seed(model, seed):
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "random_state"):
        classifier.set_params(random_state=seed)


def flatten_metrics(dataset, seed, model_name, policy_name, split, policy_metrics, classification_metrics):
    row = {
        "dataset": dataset,
        "seed": seed,
        "model": model_name,
        "policy": policy_name,
        "split": split,
        "rows": policy_metrics["rows"],
        "accuracy": classification_metrics["accuracy"],
        "macro_f1": classification_metrics["macro_f1"],
        "weighted_f1": classification_metrics["weighted_f1"],
        "unsafe_continuation_high_risk": policy_metrics["unsafe_continuation_rate_high_risk"],
        "unsafe_continuation_alert": policy_metrics["unsafe_continuation_rate_alert"],
        "unsafe_continuation_confirm": policy_metrics["unsafe_continuation_rate_confirm"],
        "over_escalation_burden": policy_metrics["over_escalation_burden"],
        "urgent_escalation_precision": policy_metrics["urgent_escalation_precision"],
        "human_defer_precision": policy_metrics["human_defer_precision"],
        "mean_time_to_escalation": policy_metrics["time_to_escalation"]["mean"],
        "median_time_to_escalation": policy_metrics["time_to_escalation"]["median"],
        "conversations_missing_escalation": policy_metrics["time_to_escalation"]["missing"],
        "mean_time_to_urgent_escalation": policy_metrics["time_to_urgent_escalation"]["mean"],
        "conversations_missing_urgent_escalation": policy_metrics["time_to_urgent_escalation"]["missing"],
    }
    for level in range(4):
        row[f"level_{level}_rate"] = policy_metrics["policy_level_distribution"].get(level, 0.0)
        per_class = classification_metrics["per_class"][str(level)]
        row[f"level_{level}_precision"] = per_class["precision"]
        row[f"level_{level}_recall"] = per_class["recall"]
        row[f"level_{level}_f1"] = per_class["f1-score"]
        row[f"level_{level}_support"] = per_class["support"]
    row["confusion_matrix"] = json.dumps(classification_metrics["confusion_matrix"])
    return row


def prediction_export(frame):
    preferred = [
        "split",
        "dialogue_id",
        "turn_id",
        "speaker_role",
        "policy_target",
        "policy_level",
        "policy_action",
        "learned_policy_level_before_guardrail",
        "guardrail_minimum_level",
        "guardrail_variant",
        "guardrail_triggered",
        "guardrail_reason",
        "high_risk_target",
        "target_alert",
        "target_confirm",
        *[f"target_category_{name}" for name in learned_policy.CATEGORY_TARGETS],
        "alert_prob",
        "confirm_prob",
        *[f"category_prob_{name}" for name in learned_policy.CATEGORY_TARGETS],
        "risk_accumulation_score",
        "risk_escalation_score",
        "risk_volatility_score",
        "policy_prob_level_0",
        "policy_prob_level_1",
        "policy_prob_level_2",
        "policy_prob_level_3",
    ]
    return frame[[column for column in preferred if column in frame.columns]]


def evaluate_frame(dataset, seed, model_name, policy_name, split, frame, y_true):
    policy_metrics = learned_policy.evaluate_policy(frame)
    classification_metrics = learned_policy.classification_metrics(
        y_true, frame["policy_level"].astype(int).to_numpy()
    )
    return flatten_metrics(
        dataset,
        seed,
        model_name,
        policy_name,
        split,
        policy_metrics,
        classification_metrics,
    )


def run_dataset(dataset_name, data_dir):
    frames = {}
    for split, filename in SPLIT_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {dataset_name} {split} data: {path}")
        frames[split] = learned_policy.create_policy_target(pd.read_csv(path))

    features = {}
    numeric_cols = None
    categorical_cols = None
    for split, frame in frames.items():
        features[split], split_numeric, split_categorical = learned_policy.prepare_features(frame)
        if numeric_cols is None:
            numeric_cols = split_numeric
            categorical_cols = split_categorical
        elif split_numeric != numeric_cols or split_categorical != categorical_cols:
            raise ValueError(f"Feature mismatch in {dataset_name} {split}")

    targets = {
        split: frame["policy_target"].astype(int).to_numpy()
        for split, frame in frames.items()
    }
    metric_rows = []
    export_dir = OUTPUT_ROOT / "predictions" / dataset_name
    export_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        for model_name in MODEL_NAMES:
            model = learned_policy.build_model(model_name, numeric_cols, categorical_cols)
            set_model_seed(model, seed)
            model.fit(features["train"], targets["train"])

            for split in SPLIT_FILES:
                predictions = model.predict(features[split])
                probabilities = model.predict_proba(features[split])
                unconstrained = learned_policy.attach_predictions(
                    frames[split], predictions, probabilities
                )
                metric_rows.append(
                    evaluate_frame(
                        dataset_name,
                        seed,
                        model_name,
                        "unconstrained",
                        split,
                        unconstrained,
                        targets[split],
                    )
                )
                if seed == 1:
                    prediction_export(unconstrained).to_csv(
                        export_dir / f"{model_name}__unconstrained__{split}.csv",
                        index=False,
                    )

                for variant in GUARDRAIL_VARIANTS:
                    guarded = safety_policy.apply_guardrail_variant(
                        frames[split], predictions, variant, probabilities
                    )
                    metric_rows.append(
                        evaluate_frame(
                            dataset_name,
                            seed,
                            model_name,
                            variant,
                            split,
                            guarded,
                            targets[split],
                        )
                    )
                    if seed == 1:
                        prediction_export(guarded).to_csv(
                            export_dir / f"{model_name}__{variant}__{split}.csv",
                            index=False,
                        )
    return metric_rows


def aggregate_metrics(metrics):
    numeric_columns = [
        column
        for column in metrics.columns
        if column not in {"dataset", "seed", "model", "policy", "split", "confusion_matrix"}
    ]
    grouped = metrics.groupby(["dataset", "model", "policy", "split"], as_index=False)
    means = grouped[numeric_columns].mean()
    stds = grouped[numeric_columns].std(ddof=1)
    stds = stds.rename(columns={column: f"{column}_std" for column in numeric_columns})
    return means.merge(stds, on=["dataset", "model", "policy", "split"])


def select_validation_configuration(metrics, dataset, model):
    candidates = metrics[
        (metrics["dataset"] == dataset)
        & (metrics["model"] == model)
        & (metrics["split"] == "validation")
        & (metrics["policy"] != "unconstrained")
    ].copy()
    candidates = candidates[
        (candidates["unsafe_continuation_high_risk"] <= 0.15)
        & (candidates["unsafe_continuation_confirm"] <= 0.15)
    ]
    if candidates.empty:
        candidates = metrics[
            (metrics["dataset"] == dataset)
            & (metrics["model"] == model)
            & (metrics["split"] == "validation")
            & (metrics["policy"] != "unconstrained")
        ].sort_values(
            [
                "unsafe_continuation_high_risk",
                "unsafe_continuation_confirm",
                "unsafe_continuation_alert",
                "over_escalation_burden",
            ]
        )
    else:
        candidates = candidates.sort_values(
            [
                "over_escalation_burden",
                "unsafe_continuation_confirm",
                "unsafe_continuation_alert",
                "urgent_escalation_precision",
                "human_defer_precision",
            ],
            ascending=[True, True, True, False, False],
        )
    return candidates.iloc[0]["policy"]


def build_key_comparison(metrics):
    rows = []
    for dataset in DATASETS:
        for model in MODEL_NAMES:
            selected_guardrail = select_validation_configuration(metrics, dataset, model)
            for policy in ["unconstrained", "light_safety", selected_guardrail]:
                label = (
                    "validation_selected_guardrail"
                    if policy == selected_guardrail and policy != "light_safety"
                    else ("matched_final_guardrail" if policy == "light_safety" else "unconstrained")
                )
                subset = metrics[
                    (metrics["dataset"] == dataset)
                    & (metrics["model"] == model)
                    & (metrics["policy"] == policy)
                    & (metrics["seed"] == 1)
                ].copy()
                subset.insert(3, "comparison_role", label)
                rows.append(subset)
    return pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["dataset", "model", "policy", "split"]
    )


def write_summary(key_comparison):
    high_recall = key_comparison[
        (key_comparison["dataset"] == "high_recall_confirm")
        & (key_comparison["policy"] == "light_safety")
        & (key_comparison["split"].isin(["validation", "test"]))
    ]
    lines = [
        "# Complete logistic-regression vs random-forest policy comparison",
        "",
        "The primary matched comparison uses the same high-recall Confirm progression features and the same `light_safety` guardrail for both base models. Model and guardrail fitting use training/validation data; test metrics are reported for evaluation and are not used to choose a winner.",
        "",
        "## Matched final-pipeline comparison",
        "",
        "| Model | Split | Unsafe continuation | Over-escalation | Macro-F1 | Accuracy | Urgent precision | Human-defer precision |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in high_recall.sort_values(["split", "model"]).iterrows():
        lines.append(
            f"| {row['model']} | {row['split']} | "
            f"{row['unsafe_continuation_high_risk']:.4f} | "
            f"{row['over_escalation_burden']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['accuracy']:.4f} | "
            f"{row['urgent_escalation_precision']:.4f} | "
            f"{row['human_defer_precision']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "The validation split remains the basis for model selection. The test split is used only to characterize held-out generalization. A lower unsafe-continuation rate is preferable, while over-escalation quantifies the corresponding burden.",
            "",
            "## Reproducibility",
            "",
            "- Five policy seeds were run. Logistic regression is deterministic here; random-forest variability is summarized separately.",
            "- Both old-Confirm and high-recall-Confirm feature sets were evaluated.",
            "- Both unconstrained models, the matched `light_safety` guardrail, and every existing guardrail variant were evaluated.",
            "- Prediction exports intentionally omit raw dialogue text.",
        ]
    )
    (OUTPUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name, data_dir in DATASETS.items():
        rows.extend(run_dataset(dataset_name, data_dir))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_ROOT / "all_metrics.csv", index=False)
    aggregate_metrics(metrics).to_csv(OUTPUT_ROOT / "five_seed_summary.csv", index=False)
    key_comparison = build_key_comparison(metrics)
    key_comparison.to_csv(OUTPUT_ROOT / "key_comparison.csv", index=False)
    write_summary(key_comparison)

    metadata = {
        "datasets": {name: str(path) for name, path in DATASETS.items()},
        "models": MODEL_NAMES,
        "guardrail_variants": GUARDRAIL_VARIANTS,
        "seeds": SEEDS,
        "selection_data": "validation",
        "test_used_for_selection": False,
        "raw_text_exported": False,
    }
    (OUTPUT_ROOT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved complete comparison to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
