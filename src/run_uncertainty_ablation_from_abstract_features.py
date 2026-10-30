import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import train_cradle_safety_constrained_policy_v2 as safety_policy


FEATURE_DIR = Path("cradle_dialogue/data/abstract_aligned_features")
BASELINE_POLICY_DIR = Path("cradle_dialogue/policy_outputs_high_recall_confirm/safety_constrained_policy")
RESULT_DIR = Path("cradle_dialogue/results/uncertainty_ablation")
OUTPUT_DIR = Path("cradle_dialogue/policy_outputs_uncertainty_ablation")

BASELINE_TEST_CONSTANTS = {
    "high_risk_unsafe_continuation": 0.2297872340425532,
    "confirm_unsafe_continuation": 0.2297872340425532,
    "alert_unsafe_continuation": 0.26495726495726496,
    "over_escalation_burden": 0.12238493723849372,
    "urgent_escalation_precision": 0.7727272727272727,
    "human_defer_precision": 0.3945666235446313,
    "macro_f1": 0.38786635857726703,
}

UNCERTAINTY_FEATURES = [
    "alert_entropy",
    "alert_confidence",
    "alert_margin",
    "confirm_entropy",
    "confirm_confidence",
    "confirm_margin",
    "category_mean_entropy",
    "category_max_entropy",
    "category_mean_confidence",
    "category_min_margin",
    "category_uncertain_count",
    "top_category_margin",
    "prev_alert_entropy",
    "alert_entropy_change_from_prev",
    "alert_entropy_rolling_mean_3",
    "alert_entropy_cumulative_max",
    "prev_confirm_entropy",
    "confirm_entropy_change_from_prev",
    "confirm_entropy_rolling_mean_3",
    "confirm_entropy_cumulative_max",
    "category_max_entropy_rolling_mean_3",
    "uncertainty_accumulation_score",
    "uncertainty_rising_flag",
]

SPECIAL_ZERO = set(safety_policy.SPECIAL_FILL_ZERO) | {
    "prev_alert_entropy",
    "alert_entropy_change_from_prev",
    "prev_confirm_entropy",
    "confirm_entropy_change_from_prev",
}
SPECIAL_NEG1 = set(safety_policy.SPECIAL_FILL_NEG1)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_abstract_split(split: str):
    path = FEATURE_DIR / f"cradle_{split}_abstract_aligned_features.csv"
    require_file(path)
    df = pd.read_csv(path)
    df = safety_policy.create_policy_target(df)
    return df


def load_baseline_policy(split: str):
    path = BASELINE_POLICY_DIR / f"cradle_{split}_safety_constrained_policy_outputs.csv"
    require_file(path)
    feature_df = load_abstract_split(split)
    policy_cols = [
        "dialogue_id",
        "turn_id",
        "policy_level",
        "policy_action",
        "policy_rationale",
        "learned_policy_level_before_guardrail",
        "guardrail_minimum_level",
        "guardrail_variant",
        "guardrail_triggered",
        "guardrail_reason",
        "policy_prob_level_0",
        "policy_prob_level_1",
        "policy_prob_level_2",
        "policy_prob_level_3",
    ]
    policy = pd.read_csv(path, usecols=lambda col: col in policy_cols)
    drop_cols = [col for col in policy_cols if col not in {"dialogue_id", "turn_id"} and col in feature_df.columns]
    merged = feature_df.drop(columns=drop_cols).merge(policy, on=["dialogue_id", "turn_id"], how="left")
    if merged["policy_level"].isna().any():
        raise ValueError(f"Baseline policy merge left missing policy rows for split={split}.")
    merged["policy_level"] = merged["policy_level"].astype(int)
    return merged


def feature_columns(use_uncertainty: bool):
    numeric = list(safety_policy.ALLOWED_NUMERIC)
    if use_uncertainty:
        numeric += [col for col in UNCERTAINTY_FEATURES if col not in numeric]
    categorical = list(safety_policy.ALLOWED_CATEGORICAL)
    return numeric, categorical


def prepare_features(df: pd.DataFrame, use_uncertainty: bool):
    numeric, categorical = feature_columns(use_uncertainty)
    missing = [col for col in numeric + categorical if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    x = df[numeric + categorical].copy()
    for col in numeric:
        if col in SPECIAL_ZERO:
            x[col] = x[col].fillna(0.0)
        elif col in SPECIAL_NEG1:
            x[col] = x[col].fillna(-1.0)
    for col in categorical:
        x[col] = x[col].fillna("NONE")
    if x[numeric].isna().any().any():
        missing_numeric = x[numeric].columns[x[numeric].isna().any()].tolist()
        raise ValueError(f"NaNs remain in numeric features: {missing_numeric}")
    return x, numeric, categorical


def build_random_forest(numeric, categorical):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="NONE")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=1,
        class_weight="balanced",
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def predict_probabilities(model, x):
    pred = model.predict(x).astype(int)
    raw = model.predict_proba(x)
    proba = np.zeros((len(x), 4), dtype=float)
    for idx, cls in enumerate(model.named_steps["classifier"].classes_):
        proba[:, int(cls)] = raw[:, idx]
    return pred, proba


def action_for_level(level):
    return safety_policy.LEVEL_TO_ACTION[int(level)]


def learned_rationale(level):
    level = int(level)
    if level == 3:
        return "Urgent escalation because the uncertainty-ablation learned policy predicted elevated crisis risk."
    if level == 2:
        return "Defer to human support because the uncertainty-ablation learned policy predicted elevated escalation need."
    if level == 1:
        return "Offer supportive resources because the uncertainty-ablation learned policy predicted moderate support need."
    return "Continue conversation because the uncertainty-ablation learned policy predicted low immediate risk."


def attach_learned_outputs(df: pd.DataFrame, pred, proba, label: str):
    out = df.copy()
    out["policy_level"] = pred.astype(int)
    out["policy_action"] = [action_for_level(level) for level in out["policy_level"]]
    out["policy_rationale"] = [learned_rationale(level) for level in out["policy_level"]]
    out["policy_model_label"] = label
    for idx in range(4):
        out[f"policy_prob_level_{idx}"] = proba[:, idx]
    return out


def train_policy(train_df, use_uncertainty: bool):
    x_train, numeric, categorical = prepare_features(train_df, use_uncertainty)
    y_train = train_df["policy_target"].astype(int).to_numpy()
    model = build_random_forest(numeric, categorical)
    model.fit(x_train, y_train)
    return model


def evaluate_with_classification(df: pd.DataFrame):
    policy_metrics = safety_policy.evaluate_policy(df)
    y_true = df["policy_target"].astype(int).to_numpy()
    y_pred = df["policy_level"].astype(int).to_numpy()
    policy_metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    policy_metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    policy_metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    policy_metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()
    return policy_metrics


def metric_value(metrics, name):
    if name.startswith("level_"):
        return metrics["policy_level_distribution"].get(int(name[-1]), 0.0)
    mapping = {
        "high_risk_unsafe_continuation": "unsafe_continuation_rate_high_risk",
        "confirm_unsafe_continuation": "unsafe_continuation_rate_confirm",
        "alert_unsafe_continuation": "unsafe_continuation_rate_alert",
        "over_escalation_burden": "over_escalation_burden",
        "urgent_escalation_precision": "urgent_escalation_precision",
        "human_defer_precision": "human_defer_precision",
        "macro_f1": "macro_f1",
        "weighted_f1": "weighted_f1",
    }
    if name in mapping:
        return metrics[mapping[name]]
    if name == "mean_time_to_escalation":
        return metrics["time_to_escalation"]["mean"]
    if name == "median_time_to_escalation":
        return metrics["time_to_escalation"]["median"]
    if name == "confusion_matrix":
        return json.dumps(metrics["confusion_matrix"])
    raise KeyError(name)


def flatten_metrics(policy_name, split, metrics):
    names = [
        "level_0_percentage",
        "level_1_percentage",
        "level_2_percentage",
        "level_3_percentage",
        "high_risk_unsafe_continuation",
        "confirm_unsafe_continuation",
        "alert_unsafe_continuation",
        "over_escalation_burden",
        "urgent_escalation_precision",
        "human_defer_precision",
        "mean_time_to_escalation",
        "median_time_to_escalation",
        "macro_f1",
        "weighted_f1",
        "confusion_matrix",
    ]
    row = {"policy_name": policy_name, "split": split}
    for name in names:
        key = name.replace("_percentage", "")
        row[name] = metric_value(metrics, key)
    return row


def run_model_outputs(model, frames, use_uncertainty: bool, label: str):
    outputs = {}
    for split, df in frames.items():
        x, _, _ = prepare_features(df, use_uncertainty)
        pred, proba = predict_probabilities(model, x)
        outputs[split] = attach_learned_outputs(df, pred, proba, label)
    return outputs


def save_policy_output(frame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_feature_set():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Uncertainty features used in the ablation",
        "========================================",
        "",
        "The uncertainty-augmented policy used the original progression/prediction feature set plus:",
        "",
    ]
    lines.extend(f"- {col}" for col in UNCERTAINTY_FEATURES)
    (RESULT_DIR / "uncertainty_feature_set.txt").write_text("\n".join(lines) + "\n")


def write_summary(validation_rows, test_rows, selected, baseline_val, uncertainty_val, baseline_test, uncertainty_test):
    selected_text = "selected" if selected else "not selected"
    interpretation = (
        "The framework includes probability-derived uncertainty estimation as an abstract-alignment layer, "
        "but uncertainty-augmented policy training did not produce a validation-supported improvement over "
        "the high-recall Confirm safety-constrained baseline. Therefore, the final selected policy remains "
        "the safety-constrained learned policy with high-recall Confirm B_pos_weight_x2, while richer "
        "uncertainty modeling is left for future work."
    )
    if selected:
        interpretation = (
            "The uncertainty-augmented safety-constrained policy met the validation selection rule. "
            "It should be treated as an uncertainty-aware candidate, with test metrics reported only after "
            "validation selection."
        )

    lines = [
        "# Uncertainty Ablation Summary",
        "",
        "## Setup",
        "",
        "This controlled ablation trained policy classifiers from the existing abstract-aligned features. Alert, Confirm, and Category models were not retrained. Progression features were not rebuilt. Frozen high-recall Confirm policy outputs were not overwritten.",
        "",
        "The uncertainty-augmented models used the original progression/prediction features plus probability-derived entropy, confidence, margin, and temporal uncertainty features.",
        "",
        "## Validation Selection",
        "",
        f"Uncertainty-augmented safety-constrained policy was **{selected_text}**.",
        "",
        f"Baseline validation high-risk unsafe continuation: `{baseline_val['unsafe_continuation_rate_high_risk']}`.",
        f"Uncertainty validation high-risk unsafe continuation: `{uncertainty_val['unsafe_continuation_rate_high_risk']}`.",
        f"Baseline validation over-escalation burden: `{baseline_val['over_escalation_burden']}`.",
        f"Uncertainty validation over-escalation burden: `{uncertainty_val['over_escalation_burden']}`.",
        "",
        "Selection rule: choose uncertainty only if validation high-risk unsafe continuation improves over baseline and validation over-escalation remains <= `0.20`.",
        "",
        "## Test Evaluation After Validation Decision",
        "",
        f"Baseline test high-risk unsafe continuation: `{baseline_test['unsafe_continuation_rate_high_risk']}`.",
        f"Uncertainty test high-risk unsafe continuation: `{uncertainty_test['unsafe_continuation_rate_high_risk']}`.",
        f"Baseline test over-escalation burden: `{baseline_test['over_escalation_burden']}`.",
        f"Uncertainty test over-escalation burden: `{uncertainty_test['over_escalation_burden']}`.",
        "",
        "## Interpretation",
        "",
        interpretation,
        "",
        "## Final Summary Questions",
        "",
        f"1. Which uncertainty features were used? See `uncertainty_feature_set.txt`; {len(UNCERTAINTY_FEATURES)} uncertainty features were added.",
        f"2. Did uncertainty features improve validation high-risk unsafe continuation? {'Yes' if uncertainty_val['unsafe_continuation_rate_high_risk'] < baseline_val['unsafe_continuation_rate_high_risk'] else 'No'}.",
        f"3. Did uncertainty features keep validation over-escalation burden <= 0.20? {'Yes' if uncertainty_val['over_escalation_burden'] <= 0.20 else 'No'}.",
        f"4. Was the uncertainty-augmented policy selected? {'Yes' if selected else 'No'}.",
        "5. What happened on test when the validation-selected decision was evaluated once? See `uncertainty_ablation_test_results.csv`.",
        f"6. Did uncertainty improve over the current high-recall Confirm safety-constrained baseline? {'Yes' if selected else 'No, not under the validation rule.'}",
        f"7. Should uncertainty features become part of the final policy pipeline? {'Yes, if accepting the selected validation result.' if selected else 'No; keep them as an abstract-alignment analysis layer.'}",
        f"8. Or should they be reported as an uncertainty-aware ablation/future-work direction? {'They can be reported as a selected ablation candidate.' if selected else 'Report them as an uncertainty-aware ablation and future-work direction.'}",
        "9. How should the paper describe the uncertainty-aware component? As probability-derived uncertainty estimation added to the framework, with explicit validation ablation results.",
        "10. What is the single next step? Write this as an ablation result in the Results/Discussion without changing the frozen final policy unless the validation rule selected uncertainty.",
    ]
    (RESULT_DIR / "uncertainty_ablation_summary.md").write_text("\n".join(lines) + "\n")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_feature_set()

    frames = {split: load_abstract_split(split) for split in ["train", "validation", "test"]}
    baseline = {split: load_baseline_policy(split) for split in ["validation", "test"]}

    original_model = train_policy(frames["train"], use_uncertainty=False)
    uncertainty_model = train_policy(frames["train"], use_uncertainty=True)

    original_outputs = run_model_outputs(original_model, frames, use_uncertainty=False, label="learned_original_features")
    uncertainty_outputs = run_model_outputs(
        uncertainty_model,
        frames,
        use_uncertainty=True,
        label="learned_uncertainty_augmented",
    )
    safety_uncertainty_outputs = {}
    for split, learned_out in uncertainty_outputs.items():
        probabilities = learned_out[[f"policy_prob_level_{idx}" for idx in range(4)]].to_numpy()
        safety_uncertainty_outputs[split] = safety_policy.apply_guardrail_variant(
            learned_out,
            learned_out["policy_level"].astype(int).to_numpy(),
            "light_safety",
            probabilities=probabilities,
        )
        safety_uncertainty_outputs[split]["policy_model_label"] = "safety_constrained_uncertainty_augmented"

    save_policy_output(
        uncertainty_outputs["validation"],
        OUTPUT_DIR / "learned_policy_uncertainty_validation_outputs.csv",
    )
    save_policy_output(
        uncertainty_outputs["test"],
        OUTPUT_DIR / "learned_policy_uncertainty_test_outputs.csv",
    )
    save_policy_output(
        safety_uncertainty_outputs["validation"],
        OUTPUT_DIR / "safety_constrained_uncertainty_validation_outputs.csv",
    )
    save_policy_output(
        safety_uncertainty_outputs["test"],
        OUTPUT_DIR / "safety_constrained_uncertainty_test_outputs.csv",
    )

    validation_metrics = {
        "baseline_high_recall_safety_constrained": evaluate_with_classification(baseline["validation"]),
        "learned_original_features": evaluate_with_classification(original_outputs["validation"]),
        "learned_uncertainty_augmented": evaluate_with_classification(uncertainty_outputs["validation"]),
        "safety_constrained_uncertainty_augmented": evaluate_with_classification(safety_uncertainty_outputs["validation"]),
    }
    test_metrics = {
        "baseline_high_recall_safety_constrained": evaluate_with_classification(baseline["test"]),
        "learned_original_features": evaluate_with_classification(original_outputs["test"]),
        "learned_uncertainty_augmented": evaluate_with_classification(uncertainty_outputs["test"]),
        "safety_constrained_uncertainty_augmented": evaluate_with_classification(safety_uncertainty_outputs["test"]),
    }

    validation_rows = [
        flatten_metrics(name, "validation", metrics)
        for name, metrics in validation_metrics.items()
    ]
    test_rows = [
        flatten_metrics(name, "test", metrics)
        for name, metrics in test_metrics.items()
    ]
    pd.DataFrame(validation_rows).to_csv(RESULT_DIR / "uncertainty_ablation_validation_results.csv", index=False)
    pd.DataFrame(test_rows).to_csv(RESULT_DIR / "uncertainty_ablation_test_results.csv", index=False)

    baseline_val = validation_metrics["baseline_high_recall_safety_constrained"]
    uncertainty_val = validation_metrics["safety_constrained_uncertainty_augmented"]
    baseline_test = test_metrics["baseline_high_recall_safety_constrained"]
    uncertainty_test = test_metrics["safety_constrained_uncertainty_augmented"]
    selected = (
        uncertainty_val["unsafe_continuation_rate_high_risk"] < baseline_val["unsafe_continuation_rate_high_risk"]
        and uncertainty_val["over_escalation_burden"] <= 0.20
    )
    write_summary(validation_rows, test_rows, selected, baseline_val, uncertainty_val, baseline_test, uncertainty_test)

    print("1. Script completed successfully.")
    print(f"2. Uncertainty selected: {selected}.")
    print(
        "3. Baseline validation unsafe and over-escalation: "
        f"{baseline_val['unsafe_continuation_rate_high_risk']}, {baseline_val['over_escalation_burden']}."
    )
    print(
        "4. Uncertainty validation unsafe and over-escalation: "
        f"{uncertainty_val['unsafe_continuation_rate_high_risk']}, {uncertainty_val['over_escalation_burden']}."
    )
    print(
        "5. Baseline test unsafe and over-escalation: "
        f"{baseline_test['unsafe_continuation_rate_high_risk']}, {baseline_test['over_escalation_burden']}."
    )
    print(
        "6. Uncertainty test unsafe and over-escalation: "
        f"{uncertainty_test['unsafe_continuation_rate_high_risk']}, {uncertainty_test['over_escalation_burden']}."
    )
    recommendation = (
        "Use uncertainty-augmented safety-constrained policy as a selected candidate."
        if selected
        else "Keep current high-recall Confirm safety-constrained policy final; report uncertainty as an ablation/future-work direction."
    )
    print(f"7. Final recommendation: {recommendation}")


if __name__ == "__main__":
    main()
