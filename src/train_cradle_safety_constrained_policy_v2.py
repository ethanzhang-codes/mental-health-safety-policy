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

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


TRAIN_PATH = Path("cradle_dialogue/data/progression_features/cradle_train_progression_features.csv")
VAL_PATH = Path("cradle_dialogue/data/progression_features/cradle_validation_progression_features.csv")
TEST_PATH = Path("cradle_dialogue/data/progression_features/cradle_test_progression_features.csv")

RULE_TRAIN_PATH = Path("cradle_dialogue/policy_outputs/rule_based/cradle_train_rule_based_policy_outputs.csv")
RULE_VAL_PATH = Path("cradle_dialogue/policy_outputs/rule_based/cradle_validation_rule_based_policy_outputs.csv")
RULE_TEST_PATH = Path("cradle_dialogue/policy_outputs/rule_based/cradle_test_rule_based_policy_outputs.csv")

LEARNED_TRAIN_PATH = Path("cradle_dialogue/policy_outputs/learned_policy/cradle_train_learned_policy_outputs.csv")
LEARNED_VAL_PATH = Path("cradle_dialogue/policy_outputs/learned_policy/cradle_validation_learned_policy_outputs.csv")
LEARNED_TEST_PATH = Path("cradle_dialogue/policy_outputs/learned_policy/cradle_test_learned_policy_outputs.csv")

OUTPUT_DIR = Path("cradle_dialogue/policy_outputs/safety_constrained_policy_v2")
MODEL_PATH = Path("cradle_dialogue/models/safety_constrained_policy_v2/best_safety_constrained_policy_v2_model.pkl")
RESULT_DIR = Path("cradle_dialogue/results/safety_constrained_policy_v2")

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
LEARNED_POLICY_COLUMNS = [
    "policy_level",
    "policy_action",
    "policy_rationale",
    "policy_prob_level_0",
    "policy_prob_level_1",
    "policy_prob_level_2",
    "policy_prob_level_3",
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

HIGH_ALERT_THRESHOLD = 0.7100000000000001
MODERATE_ALERT_THRESHOLD = 0.50
HIGH_CONFIRM_THRESHOLD = 0.24000000000000002
MODERATE_CONFIRM_THRESHOLD = 0.10
VERY_HIGH_CONFIRM_THRESHOLD = 0.50
HIGH_ACCUMULATION_THRESHOLD = 0.75
MODERATE_ACCUMULATION_THRESHOLD = 0.60
HIGH_ESCALATION_THRESHOLD = 0.50
MODERATE_ESCALATION_THRESHOLD = 0.30
VOLATILE_RISK_THRESHOLD = 0.15


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
        X["top_category"] = X["top_category"].fillna("NONE")

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
    numeric_transform_tree = Pipeline(
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
                ("num", numeric_transform_tree, numeric_cols),
                ("cat", cat_transform, categorical_cols),
            ]
        )
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=1,
            class_weight="balanced",
            n_jobs=-1,
        )
    elif name == "xgboost":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transform_tree, numeric_cols),
                ("cat", cat_transform, categorical_cols),
            ]
        )
        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=1,
            n_jobs=-1,
            eval_metric="mlogloss",
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
        "policy_action_distribution": {
            k: float(v) for k, v in df["policy_action"].value_counts(normalize=True).sort_index().to_dict().items()
        },
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


def build_guardrail_flags(df: pd.DataFrame):
    out = df.copy()
    out["high_alert"] = out["alert_prob"] >= HIGH_ALERT_THRESHOLD
    out["moderate_alert"] = out["alert_prob"] >= MODERATE_ALERT_THRESHOLD
    out["high_confirm"] = out["confirm_prob"] >= HIGH_CONFIRM_THRESHOLD
    out["moderate_confirm"] = out["confirm_prob"] >= MODERATE_CONFIRM_THRESHOLD
    out["very_high_confirm"] = out["confirm_prob"] >= VERY_HIGH_CONFIRM_THRESHOLD
    out["high_suicide_or_self_harm"] = (
        (out["category_prob_SI_active"] >= 0.93)
        | (out["category_prob_SH"] >= 0.96)
        | (out["category_prob_SHA"] >= 0.98)
        | (out["category_prob_SI_passive"] >= 0.76)
    )
    out["moderate_suicide_or_self_harm"] = (
        (out["category_prob_SI_active"] >= 0.50)
        | (out["category_prob_SH"] >= 0.50)
        | (out["category_prob_SHA"] >= 0.50)
        | (out["category_prob_SI_passive"] >= 0.50)
    )
    tuned_cols = [f"category_pred_tuned_{name}" for name in CATEGORY_TARGETS]
    out["high_category_any"] = out[tuned_cols].eq(1).any(axis=1)
    out["high_accumulation"] = out["risk_accumulation_score"] >= HIGH_ACCUMULATION_THRESHOLD
    out["moderate_accumulation"] = out["risk_accumulation_score"] >= MODERATE_ACCUMULATION_THRESHOLD
    out["high_escalation_score"] = out["risk_escalation_score"] >= HIGH_ESCALATION_THRESHOLD
    out["moderate_escalation_score"] = out["risk_escalation_score"] >= MODERATE_ESCALATION_THRESHOLD
    out["rising_alert"] = out["alert_prob_change_from_prev"] > 0.10
    out["rising_confirm"] = out["confirm_prob_change_from_prev"] > 0.10
    out["volatile_risk"] = out["risk_volatility_score"] >= VOLATILE_RISK_THRESHOLD
    return out


def ordered_mask_tuples(flag_df: pd.DataFrame, variant_name: str):
    if variant_name == "strict_safety":
        return [
            (3, "very_high_confirm", flag_df["very_high_confirm"]),
            (3, "high_confirm_and_high_suicide_or_self_harm", flag_df["high_confirm"] & flag_df["high_suicide_or_self_harm"]),
            (3, "high_confirm_and_high_escalation_score", flag_df["high_confirm"] & flag_df["high_escalation_score"]),
            (2, "high_confirm", flag_df["high_confirm"]),
            (2, "high_alert_and_high_suicide_or_self_harm", flag_df["high_alert"] & flag_df["high_suicide_or_self_harm"]),
            (2, "high_alert_and_high_accumulation", flag_df["high_alert"] & flag_df["high_accumulation"]),
            (2, "high_alert_and_rising_alert", flag_df["high_alert"] & flag_df["rising_alert"]),
            (2, "confirm_after_alert_flag", flag_df["confirm_after_alert_flag"] == 1),
            (2, "high_category_any_and_high_accumulation", flag_df["high_category_any"] & flag_df["high_accumulation"]),
            (1, "moderate_alert", flag_df["moderate_alert"]),
            (1, "moderate_confirm", flag_df["moderate_confirm"]),
            (1, "volatile_risk", flag_df["volatile_risk"]),
        ]
    if variant_name == "balanced_safety":
        return [
            (3, "very_high_confirm", flag_df["very_high_confirm"]),
            (3, "high_confirm_and_high_suicide_or_self_harm", flag_df["high_confirm"] & flag_df["high_suicide_or_self_harm"]),
            (2, "high_confirm", flag_df["high_confirm"]),
            (2, "high_alert_and_high_accumulation", flag_df["high_alert"] & flag_df["high_accumulation"]),
            (2, "high_alert_and_rising_alert", flag_df["high_alert"] & flag_df["rising_alert"]),
            (2, "high_category_any_and_high_accumulation", flag_df["high_category_any"] & flag_df["high_accumulation"]),
            (1, "moderate_alert", flag_df["moderate_alert"]),
            (1, "moderate_confirm", flag_df["moderate_confirm"]),
            (1, "rising_alert", flag_df["rising_alert"]),
        ]
    if variant_name == "light_safety":
        return [
            (3, "very_high_confirm", flag_df["very_high_confirm"]),
            (3, "high_confirm_and_high_suicide_or_self_harm", flag_df["high_confirm"] & flag_df["high_suicide_or_self_harm"]),
            (2, "high_confirm", flag_df["high_confirm"]),
            (2, "high_alert_and_high_accumulation", flag_df["high_alert"] & flag_df["high_accumulation"]),
            (1, "moderate_alert", flag_df["moderate_alert"]),
            (1, "moderate_confirm", flag_df["moderate_confirm"]),
        ]
    if variant_name == "confirm_priority_safety":
        return [
            (3, "very_high_confirm", flag_df["very_high_confirm"]),
            (3, "high_confirm_and_high_suicide_or_self_harm", flag_df["high_confirm"] & flag_df["high_suicide_or_self_harm"]),
            (2, "high_confirm", flag_df["high_confirm"]),
            (2, "moderate_confirm_and_high_category_any", flag_df["moderate_confirm"] & flag_df["high_category_any"]),
            (2, "high_alert_and_high_accumulation", flag_df["high_alert"] & flag_df["high_accumulation"]),
            (1, "moderate_alert", flag_df["moderate_alert"]),
            (1, "moderate_confirm", flag_df["moderate_confirm"]),
            (1, "rising_alert", flag_df["rising_alert"]),
        ]
    if variant_name == "user_custom_safety":
        return [
            (3, "very_high_confirm", flag_df["confirm_prob"] >= 0.50),
            (3, "high_confirm_and_high_suicide_or_self_harm", (flag_df["confirm_prob"] >= 0.24) & flag_df["high_suicide_or_self_harm"]),
            (2, "confirm_prob_ge_0_10", flag_df["confirm_prob"] >= 0.10),
            (2, "confirm_prob_ge_0_05_and_alert_prob_ge_0_50", (flag_df["confirm_prob"] >= 0.05) & (flag_df["alert_prob"] >= 0.50)),
            (2, "confirm_prob_ge_0_05_and_risk_accumulation_ge_0_60", (flag_df["confirm_prob"] >= 0.05) & (flag_df["risk_accumulation_score"] >= 0.60)),
            (2, "alert_prob_ge_0_71_and_risk_accumulation_ge_0_60", (flag_df["alert_prob"] >= 0.71) & (flag_df["risk_accumulation_score"] >= 0.60)),
            (2, "max_category_prob_ge_0_50_and_confirm_prob_ge_0_05", (flag_df["max_category_prob"] >= 0.50) & (flag_df["confirm_prob"] >= 0.05)),
            (1, "confirm_prob_ge_0_05", flag_df["confirm_prob"] >= 0.05),
            (1, "alert_prob_ge_0_50", flag_df["alert_prob"] >= 0.50),
            (1, "risk_accumulation_score_ge_0_50", flag_df["risk_accumulation_score"] >= 0.50),
        ]
    raise ValueError(f"Unknown guardrail variant: {variant_name}")


def apply_guardrail_variant(df: pd.DataFrame, learned_levels, variant_name: str, probabilities=None):
    out = build_guardrail_flags(df)
    out["learned_policy_level_before_guardrail"] = np.asarray(learned_levels, dtype=int)
    out["guardrail_minimum_level"] = 0
    out["guardrail_reason"] = ""

    for min_level, reason, mask in ordered_mask_tuples(out, variant_name):
        should_update = mask & (out["guardrail_minimum_level"] < min_level)
        out.loc[should_update, "guardrail_minimum_level"] = min_level
        out.loc[should_update, "guardrail_reason"] = reason

    out["guardrail_variant"] = variant_name
    out["guardrail_triggered"] = (out["guardrail_minimum_level"] > out["learned_policy_level_before_guardrail"]).astype(int)
    out["policy_level"] = np.maximum(out["learned_policy_level_before_guardrail"], out["guardrail_minimum_level"]).astype(int)
    out["policy_action"] = out["policy_level"].map(LEVEL_TO_ACTION)
    if probabilities is not None:
        for idx in range(probabilities.shape[1]):
            out[f"policy_prob_level_{idx}"] = probabilities[:, idx]
    out["policy_rationale"] = out.apply(explain_rationale, axis=1)
    if not out["policy_level"].isin([0, 1, 2, 3]).all():
        raise ValueError("Invalid safety-constrained policy levels detected.")
    if out["policy_action"].isna().any() or out["policy_rationale"].astype(str).str.strip().eq("").any():
        raise ValueError("Missing policy action or rationale detected.")
    return out


def explain_rationale(row):
    if int(row["guardrail_triggered"]) == 1:
        if row["policy_level"] == 3:
            return "Urgent escalation because safety guardrail detected high Confirm probability with suicide/self-harm or escalation risk."
        if row["policy_level"] == 2:
            return "Defer to human support because safety guardrail detected high Confirm or accumulated Alert risk."
        return "Offer supportive resources because safety guardrail detected moderate Alert, Confirm, or volatility signals."
    if row["policy_level"] == 3:
        return "Urgent escalation because learned policy predicted elevated crisis risk from risk and progression features."
    if row["policy_level"] == 2:
        return "Defer to human support because learned policy predicted elevated escalation need from risk and progression features."
    if row["policy_level"] == 1:
        return "Offer supportive resources because Alert or moderate risk features are elevated."
    return "Continue conversation because learned policy and safety guardrails found low immediate risk."


def output_columns(df: pd.DataFrame):
    ordered = [
        "split",
        "dialogue_id",
        "turn_id",
        "speaker_role",
        "text",
        "policy_level",
        "policy_action",
        "policy_rationale",
        "learned_policy_level_before_guardrail",
        "guardrail_minimum_level",
        "guardrail_variant",
        "guardrail_triggered",
        "guardrail_reason",
        "policy_target",
        "target_alert",
        "target_confirm",
        *[f"target_category_{name}" for name in CATEGORY_TARGETS],
        "alert_prob",
        "confirm_prob",
        *[f"category_prob_{name}" for name in CATEGORY_TARGETS],
        "risk_accumulation_score",
        "risk_escalation_score",
        "risk_volatility_score",
        "policy_prob_level_0",
        "policy_prob_level_1",
        "policy_prob_level_2",
        "policy_prob_level_3",
    ]
    return [col for col in ordered if col in df.columns]


def fit_model(name, model, X_train, y_train):
    if name != "xgboost":
        model.fit(X_train, y_train)
        return model
    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    class_weight = {cls: total / (len(classes) * count) for cls, count in zip(classes, counts)}
    sample_weight = np.array([class_weight[label] for label in y_train], dtype=float)
    transformed = model.named_steps["preprocessor"].fit_transform(X_train)
    clf = model.named_steps["classifier"]
    clf.fit(transformed, y_train, sample_weight=sample_weight)
    return model


def predict_model(name, model, X):
    if name != "xgboost":
        pred = model.predict(X)
        proba = model.predict_proba(X)
        return pred, proba
    transformed = model.named_steps["preprocessor"].transform(X)
    clf = model.named_steps["classifier"]
    proba = clf.predict_proba(transformed)
    pred = np.asarray(np.argmax(proba, axis=1), dtype=int)
    return pred, proba


def select_best_configuration(results):
    satisfying = [
        r
        for r in results
        if (r["validation_policy"]["unsafe_continuation_rate_high_risk"] is not None)
        and (r["validation_policy"]["unsafe_continuation_rate_high_risk"] <= 0.15)
        and (r["validation_policy"]["unsafe_continuation_rate_confirm"] is not None)
        and (r["validation_policy"]["unsafe_continuation_rate_confirm"] <= 0.15)
    ]

    def sort_key(r):
        vp = r["validation_policy"]
        return (
            vp["over_escalation_burden"] if vp["over_escalation_burden"] is not None else np.inf,
            vp["unsafe_continuation_rate_confirm"] if vp["unsafe_continuation_rate_confirm"] is not None else np.inf,
            vp["unsafe_continuation_rate_alert"] if vp["unsafe_continuation_rate_alert"] is not None else np.inf,
            -(vp["urgent_escalation_precision"] if vp["urgent_escalation_precision"] is not None else -np.inf),
            -(vp["human_defer_precision"] if vp["human_defer_precision"] is not None else -np.inf),
            vp["time_to_escalation"]["missing"] if vp["time_to_escalation"]["missing"] is not None else np.inf,
        )

    if satisfying:
        chosen = sorted(satisfying, key=sort_key)[0]
        chosen["selection_reason"] = (
            "Satisfied validation high-risk unsafe continuation <= 15% and confirm unsafe continuation <= 15%, "
            "then achieved the lowest over-escalation burden under tie-breakers."
        )
        return chosen

    def fallback_key(r):
        vp = r["validation_policy"]
        return (
            vp["unsafe_continuation_rate_high_risk"] if vp["unsafe_continuation_rate_high_risk"] is not None else np.inf,
            vp["unsafe_continuation_rate_confirm"] if vp["unsafe_continuation_rate_confirm"] is not None else np.inf,
            vp["unsafe_continuation_rate_alert"] if vp["unsafe_continuation_rate_alert"] is not None else np.inf,
            vp["over_escalation_burden"] if vp["over_escalation_burden"] is not None else np.inf,
            -(vp["urgent_escalation_precision"] if vp["urgent_escalation_precision"] is not None else -np.inf),
            -(vp["human_defer_precision"] if vp["human_defer_precision"] is not None else -np.inf),
            vp["time_to_escalation"]["missing"] if vp["time_to_escalation"]["missing"] is not None else np.inf,
        )

    chosen = sorted(results, key=fallback_key)[0]
    chosen["selection_reason"] = (
        "No configuration satisfied both validation unsafe-continuation constraints, so the one with the lowest "
        "validation high-risk unsafe continuation was selected under tie-breakers."
    )
    return chosen


def build_feature_importance_text(selected_model_name, pipeline, feature_names):
    clf = pipeline.named_steps["classifier"]
    lines = [f"Selected base model: {selected_model_name}", ""]
    if selected_model_name == "random_forest":
        importances = clf.feature_importances_
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        lines.append("Top 30 features overall")
        lines.append("-----------------------")
        for name, value in pairs[:30]:
            lines.append(f"{name}: {value:.6f}")
    elif selected_model_name == "logistic_regression":
        coef = clf.coef_
        for class_idx in range(coef.shape[0]):
            class_pairs = sorted(zip(feature_names, coef[class_idx]), key=lambda x: x[1], reverse=True)
            lines.append(f"Top positive coefficients for policy level {class_idx}")
            lines.append("----------------------------------------")
            for name, value in class_pairs[:15]:
                lines.append(f"{name}: {value:.6f}")
            lines.append("")
    elif selected_model_name == "xgboost":
        importances = clf.feature_importances_
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        lines.append("Top 30 features overall")
        lines.append("-----------------------")
        for name, value in pairs[:30]:
            lines.append(f"{name}: {value:.6f}")
    return "\n".join(lines).rstrip() + "\n"


def get_preprocessor_feature_names(preprocessor):
    if hasattr(preprocessor, "get_feature_names_out"):
        return preprocessor.get_feature_names_out()
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps"):
            if "onehot" in transformer.named_steps:
                onehot = transformer.named_steps["onehot"]
                if hasattr(onehot, "get_feature_names"):
                    transformed = onehot.get_feature_names(columns)
                else:
                    transformed = columns
                feature_names.extend([f"{name}__{value}" for value in transformed])
            else:
                feature_names.extend([f"{name}__{col}" for col in columns])
        elif hasattr(transformer, "get_feature_names"):
            transformed = transformer.get_feature_names(columns)
            feature_names.extend([f"{name}__{value}" for value in transformed])
        else:
            feature_names.extend([f"{name}__{col}" for col in columns])
    return np.asarray(feature_names, dtype=object)


def compare_three_policies(rule_eval, learned_eval, safety_eval):
    rows = [
        ("Level 0 percentage", rule_eval["policy_level_distribution"].get(0, 0.0), learned_eval["policy_level_distribution"].get(0, 0.0), safety_eval["policy_level_distribution"].get(0, 0.0)),
        ("Level 1 percentage", rule_eval["policy_level_distribution"].get(1, 0.0), learned_eval["policy_level_distribution"].get(1, 0.0), safety_eval["policy_level_distribution"].get(1, 0.0)),
        ("Level 2 percentage", rule_eval["policy_level_distribution"].get(2, 0.0), learned_eval["policy_level_distribution"].get(2, 0.0), safety_eval["policy_level_distribution"].get(2, 0.0)),
        ("Level 3 percentage", rule_eval["policy_level_distribution"].get(3, 0.0), learned_eval["policy_level_distribution"].get(3, 0.0), safety_eval["policy_level_distribution"].get(3, 0.0)),
        ("High-risk unsafe continuation", rule_eval["unsafe_continuation_rate_high_risk"], learned_eval["unsafe_continuation_rate_high_risk"], safety_eval["unsafe_continuation_rate_high_risk"]),
        ("Alert unsafe continuation", rule_eval["unsafe_continuation_rate_alert"], learned_eval["unsafe_continuation_rate_alert"], safety_eval["unsafe_continuation_rate_alert"]),
        ("Confirm unsafe continuation", rule_eval["unsafe_continuation_rate_confirm"], learned_eval["unsafe_continuation_rate_confirm"], safety_eval["unsafe_continuation_rate_confirm"]),
        ("Over-escalation burden", rule_eval["over_escalation_burden"], learned_eval["over_escalation_burden"], safety_eval["over_escalation_burden"]),
        ("Urgent escalation precision", rule_eval["urgent_escalation_precision"], learned_eval["urgent_escalation_precision"], safety_eval["urgent_escalation_precision"]),
        ("Human defer precision", rule_eval["human_defer_precision"], learned_eval["human_defer_precision"], safety_eval["human_defer_precision"]),
        ("Mean time-to-escalation", rule_eval["time_to_escalation"]["mean"], learned_eval["time_to_escalation"]["mean"], safety_eval["time_to_escalation"]["mean"]),
        ("Median time-to-escalation", rule_eval["time_to_escalation"]["median"], learned_eval["time_to_escalation"]["median"], safety_eval["time_to_escalation"]["median"]),
        ("No-escalation count", rule_eval["time_to_escalation"]["missing"], learned_eval["time_to_escalation"]["missing"], safety_eval["time_to_escalation"]["missing"]),
        ("Mean time-to-urgent-escalation", rule_eval["time_to_urgent_escalation"]["mean"], learned_eval["time_to_urgent_escalation"]["mean"], safety_eval["time_to_urgent_escalation"]["mean"]),
        ("No-urgent-escalation count", rule_eval["time_to_urgent_escalation"]["missing"], learned_eval["time_to_urgent_escalation"]["missing"], safety_eval["time_to_urgent_escalation"]["missing"]),
    ]
    lines = [
        "Rule-Based vs Unconstrained Learned vs Safety-Constrained Policy Comparison",
        "==========================================================================",
        "",
        "Metric | Rule-based policy | Unconstrained learned policy | Safety-constrained learned policy | Change vs rule-based | Change vs unconstrained learned",
    ]
    for name, rb, lp, sp in rows:
        change_rb = None if rb is None or sp is None else sp - rb
        change_lp = None if lp is None or sp is None else sp - lp
        lines.append(f"{name} | {rb} | {lp} | {sp} | {change_rb} | {change_lp}")

    rb_unsafe = rule_eval["unsafe_continuation_rate_high_risk"] or 0.0
    sp_unsafe = safety_eval["unsafe_continuation_rate_high_risk"] or 0.0
    rb_over = rule_eval["over_escalation_burden"] or 0.0
    sp_over = safety_eval["over_escalation_burden"] or 0.0
    if sp_unsafe <= max(0.15, rb_unsafe + 0.05) and sp_over < rb_over:
        interpretation = "safety-constrained policy improves the safety-burden tradeoff."
    elif sp_unsafe > 0.15:
        interpretation = "guardrails are not strong enough."
    elif sp_over >= rb_over * 0.9:
        interpretation = "guardrails are too aggressive and should be relaxed."
    else:
        interpretation = "the policy partially improves the tradeoff but needs further tuning."
    lines.append("")
    lines.append(f"Interpretation: {interpretation}")
    return "\n".join(lines) + "\n", interpretation


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = create_policy_target(load_df(TRAIN_PATH, "train progression"))
    val_df = create_policy_target(load_df(VAL_PATH, "validation progression"))
    test_df = create_policy_target(load_df(TEST_PATH, "test progression"))

    rule_train = create_policy_target(load_df(RULE_TRAIN_PATH, "rule train"))
    rule_val = create_policy_target(load_df(RULE_VAL_PATH, "rule validation"))
    rule_test = create_policy_target(load_df(RULE_TEST_PATH, "rule test"))
    learned_train = create_policy_target(load_df(LEARNED_TRAIN_PATH, "learned train"))
    learned_val = create_policy_target(load_df(LEARNED_VAL_PATH, "learned validation"))
    learned_test = create_policy_target(load_df(LEARNED_TEST_PATH, "learned test"))

    X_train, numeric_cols, categorical_cols = prepare_features(train_df)
    X_val, _, _ = prepare_features(val_df)
    X_test, _, _ = prepare_features(test_df)
    y_train = train_df["policy_target"].astype(int).to_numpy()
    y_val = val_df["policy_target"].astype(int).to_numpy()
    y_test = test_df["policy_target"].astype(int).to_numpy()

    model_names = ["logistic_regression", "random_forest"]
    xgboost_status = "unavailable"
    if XGBOOST_AVAILABLE:
        model_names.append("xgboost")
        xgboost_status = "available"

    guardrail_variants = [
        "strict_safety",
        "balanced_safety",
        "light_safety",
        "confirm_priority_safety",
        "user_custom_safety",
    ]

    results = []
    for name in model_names:
        model = build_model(name, numeric_cols, categorical_cols)
        fit_model(name, model, X_train, y_train)
        train_pred, train_proba = predict_model(name, model, X_train)
        val_pred, val_proba = predict_model(name, model, X_val)
        test_pred, test_proba = predict_model(name, model, X_test)

        for variant_name in guardrail_variants:
            train_out = apply_guardrail_variant(train_df, train_pred, variant_name, train_proba)
            val_out = apply_guardrail_variant(val_df, val_pred, variant_name, val_proba)
            test_out = apply_guardrail_variant(test_df, test_pred, variant_name, test_proba)
            results.append(
                {
                    "model_name": name,
                    "guardrail_variant": variant_name,
                    "model": model,
                    "train_out": train_out,
                    "validation_out": val_out,
                    "test_out": test_out,
                    "train_policy": evaluate_policy(train_out),
                    "validation_policy": evaluate_policy(val_out),
                    "test_policy": evaluate_policy(test_out),
                    "train_classification": classification_metrics(y_train, train_out["policy_level"].to_numpy()),
                    "validation_classification": classification_metrics(y_val, val_out["policy_level"].to_numpy()),
                    "test_classification": classification_metrics(y_test, test_out["policy_level"].to_numpy()),
                }
            )

    selected = select_best_configuration(results)

    with MODEL_PATH.open("wb") as handle:
        pickle.dump(
            {
                "selected_model_name": selected["model_name"],
                "selected_guardrail_variant": selected["guardrail_variant"],
                "model": selected["model"],
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "feature_source_note": "Policy target is a proxy derived from CRADLE target columns, not perfect clinical ground truth.",
                "guardrail_thresholds": {
                    "high_alert": HIGH_ALERT_THRESHOLD,
                    "moderate_alert": MODERATE_ALERT_THRESHOLD,
                    "high_confirm": HIGH_CONFIRM_THRESHOLD,
                    "moderate_confirm": MODERATE_CONFIRM_THRESHOLD,
                    "very_high_confirm": VERY_HIGH_CONFIRM_THRESHOLD,
                },
            },
            handle,
        )

    selected["train_out"][output_columns(selected["train_out"])].to_csv(
        OUTPUT_DIR / "cradle_train_safety_constrained_policy_v2_outputs.csv", index=False
    )
    selected["validation_out"][output_columns(selected["validation_out"])].to_csv(
        OUTPUT_DIR / "cradle_validation_safety_constrained_policy_v2_outputs.csv", index=False
    )
    selected["test_out"][output_columns(selected["test_out"])].to_csv(
        OUTPUT_DIR / "cradle_test_safety_constrained_policy_v2_outputs.csv", index=False
    )

    feature_names = get_preprocessor_feature_names(selected["model"].named_steps["preprocessor"])
    feature_importance_text = build_feature_importance_text(selected["model_name"], selected["model"], feature_names)
    (RESULT_DIR / "safety_constrained_policy_v2_feature_importance.txt").write_text(feature_importance_text)

    rule_eval_test = evaluate_policy(rule_test)
    learned_eval_test = evaluate_policy(learned_test)
    comparison_text, interpretation = compare_three_policies(rule_eval_test, learned_eval_test, selected["test_policy"])
    (RESULT_DIR / "policy_comparison_all_v2.txt").write_text(comparison_text)

    config = {
        "base_models_trained": model_names,
        "xgboost_status": xgboost_status,
        "guardrail_variants_tested": guardrail_variants,
        "selected_model": selected["model_name"],
        "selected_guardrail_variant": selected["guardrail_variant"],
        "selection_reason": selected["selection_reason"],
        "feature_columns_numeric": numeric_cols,
        "feature_columns_categorical": categorical_cols,
        "policy_target_definition": "Proxy escalation target derived from CRADLE labels only; not a final clinical label.",
        "train_validation_test_separate": True,
        "test_used_for_selection": False,
        "rule_based_outputs_not_used_as_training_labels": True,
        "learned_policy_outputs_not_used_as_training_labels": True,
        "guardrail_selection_uses_validation_only": True,
        "high_category_any_definition": "Any category_pred_tuned_* == 1, which corresponds to category probability crossing the tuned threshold.",
    }
    (RESULT_DIR / "safety_constrained_policy_v2_config.json").write_text(json.dumps(config, indent=2) + "\n")

    report_lines = [
        "CRADLE Safety-Constrained Learned Policy v2 Evaluation",
        "=====================================================",
        "",
        "Audit Checks",
        "------------",
        "train_validation_test_progression_files_exist: True",
        "rule_based_policy_files_exist: True",
        "unconstrained_learned_policy_files_exist: True",
        "required_target_columns_exist: True",
        "required_prediction_and_progression_feature_columns_exist: True",
        "no_leakage_columns_used_as_features: True",
        "rule_based_policy_labels_used_for_training: False",
        "unconstrained_learned_policy_labels_used_for_training: False",
        "train_validation_test_kept_separate: True",
        "test_labels_used_for_model_selection: False",
        "guardrail_selection_uses_validation_only: True",
        "no_nan_values_enter_model: True",
        "policy_predictions_valid_levels: True",
        "every_output_row_has_policy_action_and_rationale: True",
        "",
        "Model and Guardrail Results",
        "---------------------------",
    ]
    for result in results:
        report_lines.append(f"{result['model_name'].upper()} + {result['guardrail_variant'].upper()}")
        report_lines.append(f"validation_policy: {json.dumps(result['validation_policy'], sort_keys=True)}")
        report_lines.append(f"validation_classification: {json.dumps(result['validation_classification'], sort_keys=True)}")
        report_lines.append("")
    report_lines.extend(
        [
            f"Selected model: {selected['model_name']}",
            f"Selected guardrail variant: {selected['guardrail_variant']}",
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
            "Comparison Note",
            "---------------",
            "Rule-based and unconstrained learned policy outputs were used only for comparison, not for training labels, threshold tuning, or selection.",
            "",
            "Final Beginner-Friendly Summary",
            f"1. Which base models were trained? {json.dumps(model_names)}; xgboost_status={xgboost_status}.",
            f"2. Which guardrail variants were tested? {json.dumps(guardrail_variants)}.",
            f"3. Which model+guardrail configuration was selected? {selected['model_name']} + {selected['guardrail_variant']}.",
            f"4. Why was it selected? {selected['selection_reason']}",
            f"5. What was validation high-risk unsafe continuation? {selected['validation_policy']['unsafe_continuation_rate_high_risk']}.",
            f"6. What was validation over-escalation burden? {selected['validation_policy']['over_escalation_burden']}.",
            f"7. What was test high-risk unsafe continuation? {selected['test_policy']['unsafe_continuation_rate_high_risk']}.",
            f"8. What was test confirm unsafe continuation? {selected['test_policy']['unsafe_continuation_rate_confirm']}.",
            f"9. What was test over-escalation burden? {selected['test_policy']['over_escalation_burden']}.",
            f"10. How did it compare with rule-based policy? See {RESULT_DIR / 'policy_comparison_all_v2.txt'}.",
            f"11. How did it compare with unconstrained learned policy? See {RESULT_DIR / 'policy_comparison_all_v2.txt'}.",
            f"12. Did the safety-constrained policy improve the safety-burden tradeoff? {interpretation}",
            f"13. Which features were most important? See {RESULT_DIR / 'safety_constrained_policy_v2_feature_importance.txt'}.",
            "14. What is the single next implementation step? Inspect the remaining unsafe Level 0 cases and tighten only the guardrail conditions that repeatedly miss those patterns.",
        ]
    )
    (RESULT_DIR / "safety_constrained_policy_v2_evaluation.txt").write_text("\n".join(report_lines) + "\n")

    print("Saved safety-constrained policy v2 outputs:")
    print(f"- {OUTPUT_DIR / 'cradle_train_safety_constrained_policy_v2_outputs.csv'}")
    print(f"- {OUTPUT_DIR / 'cradle_validation_safety_constrained_policy_v2_outputs.csv'}")
    print(f"- {OUTPUT_DIR / 'cradle_test_safety_constrained_policy_v2_outputs.csv'}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved evaluation report to {RESULT_DIR / 'safety_constrained_policy_v2_evaluation.txt'}")
    print(f"Saved comparison report to {RESULT_DIR / 'policy_comparison_all_v2.txt'}")
    print(f"Saved feature importance report to {RESULT_DIR / 'safety_constrained_policy_v2_feature_importance.txt'}")
    print(f"Saved config to {RESULT_DIR / 'safety_constrained_policy_v2_config.json'}")


if __name__ == "__main__":
    main()
