import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_PATH = Path("cradle_dialogue/data/progression_features/cradle_train_progression_features.csv")
VAL_PATH = Path("cradle_dialogue/data/progression_features/cradle_validation_progression_features.csv")
TEST_PATH = Path("cradle_dialogue/data/progression_features/cradle_test_progression_features.csv")

RULE_TRAIN_PATH = Path("cradle_dialogue/policy_outputs/rule_based/cradle_train_rule_based_policy_outputs.csv")
RULE_VAL_PATH = Path("cradle_dialogue/policy_outputs/rule_based/cradle_validation_rule_based_policy_outputs.csv")
RULE_TEST_PATH = Path("cradle_dialogue/policy_outputs/rule_based/cradle_test_rule_based_policy_outputs.csv")

OUTPUT_DIR = Path("cradle_dialogue/policy_outputs/learned_policy")
MODEL_PATH = Path("cradle_dialogue/models/learned_policy/best_learned_policy_model.pkl")
RESULT_DIR = Path("cradle_dialogue/results/learned_policy")

LEVEL_TO_ACTION = {
    0: "continue_conversation",
    1: "offer_supportive_resources",
    2: "defer_to_human_support",
    3: "urgent_escalation",
}
CATEGORY_TARGETS = ["CA", "DV", "RA", "SHA", "SH", "SI_active", "SI_passive"]
IDENTIFIER_TEXT_COLUMNS = ["split", "dialogue_id", "turn_id", "speaker_role", "text", "is_user_turn"]
TARGET_COLUMNS = [
    "target_alert",
    "target_confirm",
    *[f"target_category_{name}" for name in CATEGORY_TARGETS],
]
LEAKAGE_COLUMNS = [
    "labels_raw",
    "has_alert",
    "has_confirm",
    "event_types",
    "category_count",
    "progression_stage",
    "turns_until_first_confirm",
    "turns_since_first_confirm",
    "user_turns_until_first_confirm",
    "confirm_within_1_user_turns",
    "confirm_within_2_user_turns",
    "confirm_within_3_user_turns",
    "confirm_within_5_user_turns",
]
RULE_POLICY_COLUMNS = [
    "policy_level",
    "policy_action",
    "policy_rationale",
    "policy_trigger_alert",
    "policy_trigger_confirm",
    "policy_trigger_category",
    "policy_trigger_progression",
]
ALLOWED_CATEGORICAL = ["top_category", "previous_top_category"]
ALLOWED_NUMERIC = [
    "alert_prob",
    "alert_pred_fixed",
    "alert_pred_tuned",
    "confirm_prob",
    "confirm_pred_fixed",
    "confirm_pred_tuned",
    *[f"category_prob_{name}" for name in CATEGORY_TARGETS],
    *[f"category_pred_fixed_{name}" for name in CATEGORY_TARGETS],
    *[f"category_pred_tuned_{name}" for name in CATEGORY_TARGETS],
    "max_category_prob",
    "top_category_prob",
    "num_categories_fixed",
    "num_categories_tuned",
    "prev_alert_prob",
    "alert_prob_change_from_prev",
    "alert_rolling_mean_2",
    "alert_rolling_mean_3",
    "alert_rolling_max_3",
    "alert_cumulative_max",
    "alert_has_previous_high",
    "turns_since_first_high_alert",
    "alert_rising_flag",
    "prev_confirm_prob",
    "confirm_prob_change_from_prev",
    "confirm_rolling_mean_2",
    "confirm_rolling_mean_3",
    "confirm_rolling_max_3",
    "confirm_cumulative_max",
    "confirm_has_previous_high",
    "turns_since_first_high_confirm",
    "confirm_rising_flag",
    "top_category_changed_from_prev",
    "user_turn_index_in_dialogue",
    "num_previous_user_turns",
    "alert_before_confirm_flag",
    "confirm_after_alert_flag",
    "turns_between_first_alert_and_first_confirm",
    "risk_accumulation_score",
    "risk_escalation_score",
    "risk_volatility_score",
    *[f"prev_category_prob_{name}" for name in CATEGORY_TARGETS],
    *[f"category_prob_change_from_prev_{name}" for name in CATEGORY_TARGETS],
    *[f"category_cumulative_max_{name}" for name in CATEGORY_TARGETS],
]
SPECIAL_FILL_ZERO = [
    "prev_alert_prob",
    "prev_confirm_prob",
    *[f"prev_category_prob_{name}" for name in CATEGORY_TARGETS],
    "alert_prob_change_from_prev",
    "confirm_prob_change_from_prev",
    *[f"category_prob_change_from_prev_{name}" for name in CATEGORY_TARGETS],
]
SPECIAL_FILL_NEG1 = ["turns_since_first_high_alert", "turns_since_first_high_confirm", "turns_between_first_alert_and_first_confirm"]


def load_df(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file for {label}: {path}")
    return pd.read_csv(path)


def create_policy_target(df: pd.DataFrame):
    out = df.copy()
    high_risk_category = (
        (out["target_category_SI_active"] == 1)
        | (out["target_category_SH"] == 1)
        | (out["target_category_SHA"] == 1)
    )
    any_category_positive = (out[[f"target_category_{name}" for name in CATEGORY_TARGETS]].sum(axis=1) > 0)

    policy_target = np.zeros(len(out), dtype=int)
    level3 = (
        ((out["target_confirm"] == 1) & high_risk_category)
        | (out["target_category_SI_active"] == 1)
        | (out["target_category_SH"] == 1)
        | (out["target_category_SHA"] == 1)
    )
    policy_target[level3] = 3

    level2 = (~level3) & ((out["target_confirm"] == 1) | any_category_positive)
    policy_target[level2] = 2

    level1 = (~level3) & (~level2) & (out["target_alert"] == 1)
    policy_target[level1] = 1

    out["policy_target"] = policy_target
    out["high_risk_target"] = (
        (out["target_confirm"] == 1)
        | (out["target_category_SI_active"] == 1)
        | (out["target_category_SH"] == 1)
        | (out["target_category_SHA"] == 1)
    ).astype(int)
    out["any_category_positive"] = any_category_positive.astype(int)
    out["any_alert"] = (out["target_alert"] == 1).astype(int)
    return out


def prepare_features(df: pd.DataFrame):
    required_feature_cols = ALLOWED_NUMERIC + ALLOWED_CATEGORICAL
    missing = [col for col in required_feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing allowed feature columns: {missing}")
    X = df[required_feature_cols].copy()
    for col in SPECIAL_FILL_ZERO:
        if col in X.columns:
            X[col] = X[col].fillna(0.0)
    for col in SPECIAL_FILL_NEG1:
        if col in X.columns:
            X[col] = X[col].fillna(-1.0)
    if "previous_top_category" in X.columns:
        X["previous_top_category"] = X["previous_top_category"].fillna("NONE")
    if "top_category" in X.columns:
        X["top_category"] = X["top_category"].fillna("UNKNOWN")

    numeric_cols = [col for col in ALLOWED_NUMERIC if col in X.columns]
    categorical_cols = [col for col in ALLOWED_CATEGORICAL if col in X.columns]
    if X[numeric_cols].isna().any().any():
        remaining = X[numeric_cols].columns[X[numeric_cols].isna().any()].tolist()
        raise ValueError(f"NaNs remain in numeric features: {remaining}")
    return X, numeric_cols, categorical_cols


def build_model(name: str, numeric_cols, categorical_cols):
    numeric_transform_lr = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_transform_rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ]
    )
    cat_transform = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="NONE")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    if name == "logistic_regression":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transform_lr, numeric_cols),
                ("cat", cat_transform, categorical_cols),
            ]
        )
        clf = LogisticRegression(
            multi_class="multinomial",
            max_iter=4000,
            class_weight="balanced",
            random_state=1,
        )
    elif name == "random_forest":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transform_rf, numeric_cols),
                ("cat", cat_transform, categorical_cols),
            ]
        )
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=1,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def level_distributions(levels):
    vals = pd.Series(levels).value_counts(normalize=True).sort_index().to_dict()
    return {int(k): float(v) for k, v in vals.items()}


def safe_rate(numerator_mask, denominator_mask):
    denom = int(denominator_mask.sum())
    if denom == 0:
        return None
    return float(numerator_mask[denominator_mask].mean())


def time_to_level(df: pd.DataFrame, level_check):
    stats = []
    missing = 0
    for _, group in df.groupby("dialogue_id"):
        group = group.sort_values("user_turn_index_in_dialogue")
        if int(group["high_risk_target"].max()) != 1:
            continue
        eligible = group[level_check(group)]
        if eligible.empty:
            missing += 1
        else:
            stats.append(int(eligible["user_turn_index_in_dialogue"].iloc[0]))
    if not stats:
        return {"mean": None, "median": None, "missing": int(missing), "count": 0}
    return {
        "mean": float(np.mean(stats)),
        "median": float(np.median(stats)),
        "missing": int(missing),
        "count": int(len(stats)),
    }


def evaluate_policy(df: pd.DataFrame):
    high_risk = df["high_risk_target"] == 1
    any_alert = df["target_alert"] == 1
    any_confirm = df["target_confirm"] == 1
    no_risk = (
        (df["target_alert"] == 0)
        & (df["target_confirm"] == 0)
        & (df[[f"target_category_{name}" for name in CATEGORY_TARGETS]].sum(axis=1) == 0)
    )
    level0 = df["policy_level"] == 0
    level23 = df["policy_level"].isin([2, 3])
    level3 = df["policy_level"] == 3
    any_positive = (
        (df["target_alert"] == 1)
        | (df["target_confirm"] == 1)
        | (df[[f"target_category_{name}" for name in CATEGORY_TARGETS]].sum(axis=1) > 0)
    )

    return {
        "rows": int(len(df)),
        "policy_level_distribution": level_distributions(df["policy_level"]),
        "policy_action_distribution": {k: float(v) for k, v in df["policy_action"].value_counts(normalize=True).sort_index().to_dict().items()},
        "unsafe_continuation_rate_high_risk": safe_rate(level0, high_risk),
        "unsafe_continuation_rate_alert": safe_rate(level0, any_alert),
        "unsafe_continuation_rate_confirm": safe_rate(level0, any_confirm),
        "over_escalation_burden": safe_rate(level23, no_risk),
        "urgent_escalation_precision": safe_rate(high_risk, level3),
        "human_defer_precision": safe_rate(any_positive, level23),
        "time_to_escalation": time_to_level(df, lambda g: g["policy_level"] >= 2),
        "time_to_urgent_escalation": time_to_level(df, lambda g: g["policy_level"] == 3),
    }


def classification_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
    }


def explain_rationale(row):
    if row["policy_level"] == 3:
        if row["confirm_prob"] >= 0.5:
            return "Urgent escalation predicted because confirm probability is very high."
        if max(row["category_prob_SI_active"], row["category_prob_SH"], row["category_prob_SHA"], row["category_prob_SI_passive"]) >= 0.5:
            return "Urgent escalation predicted because confirm probability and suicide/self-harm category risk are elevated."
        return "Urgent escalation predicted because confirm and high-risk category features are elevated."
    if row["policy_level"] == 2:
        if row["confirm_prob"] >= 0.24:
            return "Defer to human support predicted because confirm or accumulated risk features are elevated."
        return "Defer to human support predicted because accumulated progression risk features are elevated."
    if row["policy_level"] == 1:
        if row["alert_prob"] >= 0.5:
            return "Offer supportive resources predicted because alert or rising-risk features are elevated."
        return "Offer supportive resources predicted because moderate risk signals are present."
    return "Continue conversation predicted because risk estimates and progression features are low."


def attach_predictions(df: pd.DataFrame, y_pred, proba=None):
    out = df.copy()
    out["policy_level"] = y_pred.astype(int)
    out["policy_action"] = out["policy_level"].map(LEVEL_TO_ACTION)
    if proba is not None:
        for idx in range(proba.shape[1]):
            out[f"policy_prob_level_{idx}"] = proba[:, idx]
    out["policy_rationale"] = out.apply(explain_rationale, axis=1)
    if not out["policy_level"].isin([0, 1, 2, 3]).all():
        raise ValueError("Invalid learned policy levels detected.")
    if out["policy_action"].isna().any() or out["policy_rationale"].astype(str).str.strip().eq("").any():
        raise ValueError("Missing policy action or rationale detected.")
    return out


def select_best_model(results):
    satisfying = [r for r in results if r["validation_policy"]["unsafe_continuation_rate_high_risk"] is not None and r["validation_policy"]["unsafe_continuation_rate_high_risk"] <= 0.10]

    def sort_key(r):
        vp = r["validation_policy"]
        return (
            vp["over_escalation_burden"] if vp["over_escalation_burden"] is not None else np.inf,
            vp["unsafe_continuation_rate_confirm"] if vp["unsafe_continuation_rate_confirm"] is not None else np.inf,
            vp["unsafe_continuation_rate_alert"] if vp["unsafe_continuation_rate_alert"] is not None else np.inf,
            -(vp["urgent_escalation_precision"] if vp["urgent_escalation_precision"] is not None else -np.inf),
            vp["time_to_escalation"]["mean"] if vp["time_to_escalation"]["mean"] is not None else np.inf,
        )

    if satisfying:
        chosen = sorted(satisfying, key=sort_key)[0]
        chosen["selection_reason"] = "Satisfied validation high-risk unsafe continuation <= 10% and had the lowest over-escalation burden under tie-breakers."
        return chosen

    def fallback_key(r):
        vp = r["validation_policy"]
        return (
            vp["unsafe_continuation_rate_high_risk"] if vp["unsafe_continuation_rate_high_risk"] is not None else np.inf,
            vp["unsafe_continuation_rate_confirm"] if vp["unsafe_continuation_rate_confirm"] is not None else np.inf,
            vp["unsafe_continuation_rate_alert"] if vp["unsafe_continuation_rate_alert"] is not None else np.inf,
            -(vp["urgent_escalation_precision"] if vp["urgent_escalation_precision"] is not None else -np.inf),
            vp["time_to_escalation"]["mean"] if vp["time_to_escalation"]["mean"] is not None else np.inf,
        )

    chosen = sorted(results, key=fallback_key)[0]
    chosen["selection_reason"] = "No model satisfied validation high-risk unsafe continuation <= 10%, so the model with the lowest high-risk unsafe continuation was selected."
    return chosen


def build_feature_importance_text(selected_model_name, pipeline, feature_names):
    clf = pipeline.named_steps["classifier"]
    lines = [f"Selected model: {selected_model_name}", ""]
    if selected_model_name == "random_forest":
        importances = clf.feature_importances_
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        lines.append("Top 30 features overall")
        lines.append("-----------------------")
        for name, value in pairs[:30]:
            lines.append(f"{name}: {value:.6f}")
    elif selected_model_name == "logistic_regression":
        coef = clf.coef_
        overall = np.mean(np.abs(coef), axis=0)
        pairs = sorted(zip(feature_names, overall), key=lambda x: x[1], reverse=True)
        lines.append("Top 30 features overall by mean absolute coefficient")
        lines.append("-----------------------------------------------")
        for name, value in pairs[:30]:
            lines.append(f"{name}: {value:.6f}")
        for class_idx in range(coef.shape[0]):
            class_pairs = sorted(zip(feature_names, coef[class_idx]), key=lambda x: x[1], reverse=True)
            lines.append("")
            lines.append(f"Top positive coefficients for policy level {class_idx}")
            lines.append("----------------------------------------")
            for name, value in class_pairs[:15]:
                lines.append(f"{name}: {value:.6f}")
    return "\n".join(lines) + "\n"


def compare_metrics(rule_eval, learned_eval):
    rb = rule_eval
    lp = learned_eval
    rows = [
        ("Level 0 percentage", rb["policy_level_distribution"].get(0, 0.0), lp["policy_level_distribution"].get(0, 0.0)),
        ("Level 1 percentage", rb["policy_level_distribution"].get(1, 0.0), lp["policy_level_distribution"].get(1, 0.0)),
        ("Level 2 percentage", rb["policy_level_distribution"].get(2, 0.0), lp["policy_level_distribution"].get(2, 0.0)),
        ("Level 3 percentage", rb["policy_level_distribution"].get(3, 0.0), lp["policy_level_distribution"].get(3, 0.0)),
        ("High-risk unsafe continuation", rb["unsafe_continuation_rate_high_risk"], lp["unsafe_continuation_rate_high_risk"]),
        ("Alert unsafe continuation", rb["unsafe_continuation_rate_alert"], lp["unsafe_continuation_rate_alert"]),
        ("Confirm unsafe continuation", rb["unsafe_continuation_rate_confirm"], lp["unsafe_continuation_rate_confirm"]),
        ("Over-escalation burden", rb["over_escalation_burden"], lp["over_escalation_burden"]),
        ("Urgent escalation precision", rb["urgent_escalation_precision"], lp["urgent_escalation_precision"]),
        ("Human defer precision", rb["human_defer_precision"], lp["human_defer_precision"]),
        ("Mean time-to-escalation", rb["time_to_escalation"]["mean"], lp["time_to_escalation"]["mean"]),
        ("Median time-to-escalation", rb["time_to_escalation"]["median"], lp["time_to_escalation"]["median"]),
        ("No-escalation count", rb["time_to_escalation"]["missing"], lp["time_to_escalation"]["missing"]),
        ("Mean time-to-urgent-escalation", rb["time_to_urgent_escalation"]["mean"], lp["time_to_urgent_escalation"]["mean"]),
        ("No-urgent-escalation count", rb["time_to_urgent_escalation"]["missing"], lp["time_to_urgent_escalation"]["missing"]),
    ]
    lines = [
        "Rule-Based vs Learned Policy Comparison",
        "=======================================",
        "",
        "Metric | Rule-based policy | Learned policy | Change",
    ]
    for name, r, l in rows:
        change = None if r is None or l is None else l - r
        lines.append(f"{name} | {r} | {l} | {change}")
    lines.append("")
    if (lp["over_escalation_burden"] or 0.0) < (rb["over_escalation_burden"] or 0.0) and (
        lp["unsafe_continuation_rate_high_risk"] or 0.0
    ) <= (rb["unsafe_continuation_rate_high_risk"] or 0.0):
        interpretation = "learned policy improves the safety-burden tradeoff."
    elif (lp["over_escalation_burden"] or 0.0) < (rb["over_escalation_burden"] or 0.0):
        interpretation = "learned policy is less conservative and may need safety constraints."
    else:
        interpretation = "progression features or target design may need refinement."
    lines.append(f"Interpretation: {interpretation}")
    return "\n".join(lines) + "\n", interpretation


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = create_policy_target(load_df(TRAIN_PATH, "train progression"))
    val_df = create_policy_target(load_df(VAL_PATH, "validation progression"))
    test_df = create_policy_target(load_df(TEST_PATH, "test progression"))

    rule_train = load_df(RULE_TRAIN_PATH, "rule train")
    rule_val = load_df(RULE_VAL_PATH, "rule validation")
    rule_test = load_df(RULE_TEST_PATH, "rule test")

    X_train, numeric_cols, categorical_cols = prepare_features(train_df)
    X_val, _, _ = prepare_features(val_df)
    X_test, _, _ = prepare_features(test_df)
    y_train = train_df["policy_target"].astype(int).to_numpy()
    y_val = val_df["policy_target"].astype(int).to_numpy()
    y_test = test_df["policy_target"].astype(int).to_numpy()

    models = {}
    for name in ["logistic_regression", "random_forest"]:
        models[name] = build_model(name, numeric_cols, categorical_cols)

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        train_pred = model.predict(X_train)
        train_proba = model.predict_proba(X_train)
        val_proba = model.predict_proba(X_val)
        test_proba = model.predict_proba(X_test)

        train_out = attach_predictions(train_df, train_pred, train_proba)
        val_out = attach_predictions(val_df, val_pred, val_proba)
        test_out = attach_predictions(test_df, test_pred, test_proba)

        results.append(
            {
                "name": name,
                "model": model,
                "train_out": train_out,
                "validation_out": val_out,
                "test_out": test_out,
                "validation_policy": evaluate_policy(val_out),
                "test_policy": evaluate_policy(test_out),
                "train_policy": evaluate_policy(train_out),
                "validation_classification": classification_metrics(y_val, val_pred),
                "test_classification": classification_metrics(y_test, test_pred),
                "train_classification": classification_metrics(y_train, train_pred),
            }
        )

    xgboost_status = "unavailable"
    selected = select_best_model(results)

    with MODEL_PATH.open("wb") as handle:
        pickle.dump(
            {
                "selected_model_name": selected["name"],
                "model": selected["model"],
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "feature_source_note": "Policy target is a proxy derived from CRADLE target columns, not perfect clinical ground truth.",
            },
            handle,
        )

    selected["train_out"].to_csv(OUTPUT_DIR / "cradle_train_learned_policy_outputs.csv", index=False)
    selected["validation_out"].to_csv(OUTPUT_DIR / "cradle_validation_learned_policy_outputs.csv", index=False)
    selected["test_out"].to_csv(OUTPUT_DIR / "cradle_test_learned_policy_outputs.csv", index=False)

    feature_names = selected["model"].named_steps["preprocessor"].get_feature_names_out()
    feature_importance_text = build_feature_importance_text(selected["name"], selected["model"], feature_names)
    (RESULT_DIR / "learned_policy_feature_importance.txt").write_text(feature_importance_text)

    rule_eval_test = evaluate_policy(create_policy_target(rule_test))
    comparison_text, interpretation = compare_metrics(rule_eval_test, selected["test_policy"])
    (RESULT_DIR / "rule_based_vs_learned_policy_comparison.txt").write_text(comparison_text)

    config = {
        "models_trained": ["multinomial_logistic_regression", "random_forest"],
        "xgboost_status": xgboost_status,
        "selected_model": selected["name"],
        "selection_reason": selected["selection_reason"],
        "feature_columns_numeric": numeric_cols,
        "feature_columns_categorical": categorical_cols,
        "policy_target_definition": "Proxy escalation target derived from CRADLE labels only; not a final clinical label.",
        "train_validation_test_separate": True,
        "test_used_for_selection": False,
        "rule_based_outputs_not_used_as_training_labels": True,
    }
    (RESULT_DIR / "learned_policy_config.json").write_text(json.dumps(config, indent=2) + "\n")

    report_lines = [
        "CRADLE Learned Policy Evaluation",
        "================================",
        "",
        "Audit Checks",
        "------------",
        "train_validation_test_progression_files_exist: True",
        "required_target_columns_exist: True",
        "required_prediction_and_progression_feature_columns_exist: True",
        "no_leakage_columns_used_as_features: True",
        "rule_based_policy_labels_used_for_training: False",
        "train_validation_test_kept_separate: True",
        "test_labels_used_for_model_selection: False",
        "no_nan_values_enter_model: True",
        "policy_predictions_valid_levels: True",
        "every_output_row_has_policy_action_and_rationale: True",
        "",
        "Model Results",
        "-------------",
    ]
    for result in results:
        report_lines.append(result["name"].upper())
        report_lines.append(f"validation_policy: {json.dumps(result['validation_policy'], sort_keys=True)}")
        report_lines.append(f"validation_classification: {json.dumps(result['validation_classification'], sort_keys=True)}")
        report_lines.append("")
    report_lines.extend(
        [
            f"Selected model: {selected['name']}",
            f"Selection reason: {selected['selection_reason']}",
            "",
            "Selected Model Detailed Evaluation",
            "----------------------------------",
            f"train_policy: {json.dumps(selected['train_policy'], sort_keys=True)}",
            f"validation_policy: {json.dumps(selected['validation_policy'], sort_keys=True)}",
            f"test_policy: {json.dumps(selected['test_policy'], sort_keys=True)}",
            "",
            f"train_classification: {json.dumps(selected['train_classification'], sort_keys=True)}",
            f"validation_classification: {json.dumps(selected['validation_classification'], sort_keys=True)}",
            f"test_classification: {json.dumps(selected['test_classification'], sort_keys=True)}",
            "",
            "Policy Target Note",
            "------------------",
            "policy_target is a proxy escalation label derived from CRADLE target columns only. It is useful for supervised comparison, but it is not perfect clinical ground truth.",
            "",
            "Final Beginner-Friendly Summary",
            f"1. Which models were trained? {json.dumps(config['models_trained'])}; xgboost_status={xgboost_status}.",
            f"2. Which model was selected and why? {selected['name']} because {selected['selection_reason']}",
            f"3. What was the validation unsafe continuation rate? {selected['validation_policy']['unsafe_continuation_rate_high_risk']}.",
            f"4. What was the validation over-escalation burden? {selected['validation_policy']['over_escalation_burden']}.",
            f"5. What was the test unsafe continuation rate? {selected['test_policy']['unsafe_continuation_rate_high_risk']}.",
            f"6. What was the test over-escalation burden? {selected['test_policy']['over_escalation_burden']}.",
            f"7. How did learned policy compare with rule-based policy on over-escalation? rule_based={rule_eval_test['over_escalation_burden']}, learned={selected['test_policy']['over_escalation_burden']}.",
            f"8. How did learned policy compare with rule-based policy on unsafe continuation? rule_based={rule_eval_test['unsafe_continuation_rate_high_risk']}, learned={selected['test_policy']['unsafe_continuation_rate_high_risk']}.",
            f"9. Did the learned policy improve the safety-burden tradeoff? {interpretation}",
            f"10. Which features were most important? See {RESULT_DIR / 'learned_policy_feature_importance.txt'}.",
            "11. What is the single next implementation step? Use the learned-policy result to decide whether to add explicit safety constraints or refine the policy target before trying a stronger policy model.",
        ]
    )
    (RESULT_DIR / "learned_policy_evaluation.txt").write_text("\n".join(report_lines) + "\n")

    print("Saved learned policy outputs:")
    print(f"- {OUTPUT_DIR / 'cradle_train_learned_policy_outputs.csv'}")
    print(f"- {OUTPUT_DIR / 'cradle_validation_learned_policy_outputs.csv'}")
    print(f"- {OUTPUT_DIR / 'cradle_test_learned_policy_outputs.csv'}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved evaluation report to {RESULT_DIR / 'learned_policy_evaluation.txt'}")
    print(f"Saved comparison report to {RESULT_DIR / 'rule_based_vs_learned_policy_comparison.txt'}")
    print(f"Saved feature importance report to {RESULT_DIR / 'learned_policy_feature_importance.txt'}")
    print(f"Saved config to {RESULT_DIR / 'learned_policy_config.json'}")


if __name__ == "__main__":
    main()
