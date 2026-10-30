import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_PATH = Path("cradle_dialogue/data/progression_features/cradle_train_progression_features.csv")
VAL_PATH = Path("cradle_dialogue/data/progression_features/cradle_validation_progression_features.csv")
TEST_PATH = Path("cradle_dialogue/data/progression_features/cradle_test_progression_features.csv")

OUTPUT_DIR = Path("cradle_dialogue/policy_outputs/rule_based")
RESULT_DIR = Path("cradle_dialogue/results/rule_based_policy")

CATEGORY_THRESHOLDS = {
    "CA": 0.66,
    "DV": 0.74,
    "RA": 0.54,
    "SH": 0.96,
    "SHA": 0.98,
    "SI_active": 0.93,
    "SI_passive": 0.76,
}
ALERT_THRESHOLD = 0.7100000000000001
CONFIRM_THRESHOLD = 0.24000000000000002
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
REQUIRED_PROB_COLUMNS = [
    "alert_prob",
    "confirm_prob",
    "risk_accumulation_score",
    "risk_escalation_score",
    "risk_volatility_score",
    "alert_prob_change_from_prev",
    "confirm_prob_change_from_prev",
    "alert_before_confirm_flag",
    "confirm_after_alert_flag",
]
REQUIRED_TARGET_COLUMNS = [
    "target_alert",
    "target_confirm",
    "target_category_CA",
    "target_category_DV",
    "target_category_RA",
    "target_category_SHA",
    "target_category_SH",
    "target_category_SI_active",
    "target_category_SI_passive",
]
REQUIRED_CATEGORY_PROB_COLUMNS = [f"category_prob_{name}" for name in CATEGORY_THRESHOLDS]


def load_split(path: Path, split_name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing progression feature file for {split_name}: {path}")
    df = pd.read_csv(path)
    return df


def validate_inputs(df: pd.DataFrame, split_name: str):
    missing = [col for col in REQUIRED_PROB_COLUMNS + REQUIRED_TARGET_COLUMNS + REQUIRED_CATEGORY_PROB_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{split_name}: missing required columns: {missing}")
    missing_probs = [col for col in ["alert_prob", "confirm_prob", *REQUIRED_CATEGORY_PROB_COLUMNS] if df[col].isna().any()]
    if missing_probs:
        raise ValueError(f"{split_name}: missing probabilities in columns: {missing_probs}")
    leaked = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    return leaked


def build_flags(df: pd.DataFrame):
    out = df.copy()
    out["high_alert"] = out["alert_prob"] >= ALERT_THRESHOLD
    out["high_confirm"] = out["confirm_prob"] >= CONFIRM_THRESHOLD
    out["high_category_any"] = False
    for cat, threshold in CATEGORY_THRESHOLDS.items():
        out[f"high_category_{cat}"] = out[f"category_prob_{cat}"] >= threshold
        out["high_category_any"] = out["high_category_any"] | out[f"high_category_{cat}"]

    out["high_suicide_or_self_harm"] = (
        out["high_category_SI_active"]
        | out["high_category_SI_passive"]
        | out["high_category_SH"]
        | out["high_category_SHA"]
    )
    out["high_interpersonal_violence"] = (
        out["high_category_DV"]
        | out["high_category_RA"]
        | out["high_category_CA"]
    )
    out["rising_alert"] = out["alert_prob_change_from_prev"].fillna(0.0) > 0.10
    out["rising_confirm"] = out["confirm_prob_change_from_prev"].fillna(0.0) > 0.10
    out["high_accumulation"] = out["risk_accumulation_score"] >= 0.75
    out["high_escalation_score"] = out["risk_escalation_score"] >= 0.50
    out["volatile_risk"] = out["risk_volatility_score"] >= 0.15
    return out


def assign_policy(df: pd.DataFrame):
    rows = []
    for _, row in df.iterrows():
        trigger_alert = 0
        trigger_confirm = 0
        trigger_category = 0
        trigger_progression = 0

        if row["high_confirm"] and row["high_suicide_or_self_harm"]:
            level = 3
            action = "urgent_escalation"
            rationale = "Urgent escalation because Confirm probability is high and suicide/self-harm category risk is elevated."
            trigger_confirm = 1
            trigger_category = 1
        elif row["confirm_prob"] >= 0.50:
            level = 3
            action = "urgent_escalation"
            rationale = "Urgent escalation because Confirm probability is very high."
            trigger_confirm = 1
        elif row["high_confirm"] and row["high_escalation_score"]:
            level = 3
            action = "urgent_escalation"
            rationale = "Urgent escalation because Confirm probability is high and the escalation score is elevated."
            trigger_confirm = 1
            trigger_progression = 1
        elif row["high_confirm"] and row["high_category_any"] and row["high_accumulation"]:
            level = 3
            action = "urgent_escalation"
            rationale = "Urgent escalation because Confirm probability is high, category risk is elevated, and risk has accumulated."
            trigger_confirm = 1
            trigger_category = 1
            trigger_progression = 1
        elif row["high_confirm"]:
            level = 2
            action = "defer_to_human_support"
            rationale = "Defer to human support because Confirm probability is high."
            trigger_confirm = 1
        elif row["high_alert"] and row["high_suicide_or_self_harm"]:
            level = 2
            action = "defer_to_human_support"
            rationale = "Defer to human support because Alert probability is high and suicide/self-harm category risk is elevated."
            trigger_alert = 1
            trigger_category = 1
        elif row["high_alert"] and row["high_accumulation"]:
            level = 2
            action = "defer_to_human_support"
            rationale = "Defer to human support because Alert probability is high and risk has accumulated across the conversation."
            trigger_alert = 1
            trigger_progression = 1
        elif row["high_alert"] and row["rising_alert"]:
            level = 2
            action = "defer_to_human_support"
            rationale = "Defer to human support because Alert probability is high and rising."
            trigger_alert = 1
            trigger_progression = 1
        elif int(row["confirm_after_alert_flag"]) == 1:
            level = 2
            action = "defer_to_human_support"
            rationale = "Defer to human support because Confirm follows earlier Alert signals in the dialogue."
            trigger_progression = 1
        elif int(row["alert_before_confirm_flag"]) == 1 and row["high_accumulation"]:
            level = 2
            action = "defer_to_human_support"
            rationale = "Defer to human support because Alert preceded Confirm and the risk trajectory has accumulated."
            trigger_alert = 1
            trigger_confirm = 1
            trigger_progression = 1
        elif row["high_alert"]:
            level = 1
            action = "offer_supportive_resources"
            rationale = "Offer supportive resources because Alert probability is elevated."
            trigger_alert = 1
        elif row["alert_prob"] >= 0.50:
            level = 1
            action = "offer_supportive_resources"
            rationale = "Offer supportive resources because Alert probability is moderately elevated."
            trigger_alert = 1
        elif row["confirm_prob"] >= 0.10:
            level = 1
            action = "offer_supportive_resources"
            rationale = "Offer supportive resources because Confirm probability is elevated but below human-defer threshold."
            trigger_confirm = 1
        elif row["high_category_any"]:
            level = 1
            action = "offer_supportive_resources"
            rationale = "Offer supportive resources because category-specific risk is elevated."
            trigger_category = 1
        elif row["rising_alert"]:
            level = 1
            action = "offer_supportive_resources"
            rationale = "Offer supportive resources because Alert risk is rising."
            trigger_progression = 1
        elif row["rising_confirm"]:
            level = 1
            action = "offer_supportive_resources"
            rationale = "Offer supportive resources because Confirm risk is rising."
            trigger_progression = 1
        elif row["volatile_risk"]:
            level = 1
            action = "offer_supportive_resources"
            rationale = "Offer supportive resources because risk estimates are volatile."
            trigger_progression = 1
        else:
            level = 0
            action = "continue_conversation"
            rationale = "Continue conversation because Alert, Confirm, and category risk estimates are low and stable."

        result = row.to_dict()
        result.update(
            {
                "policy_level": level,
                "policy_action": action,
                "policy_rationale": rationale,
                "policy_trigger_alert": trigger_alert,
                "policy_trigger_confirm": trigger_confirm,
                "policy_trigger_category": trigger_category,
                "policy_trigger_progression": trigger_progression,
            }
        )
        rows.append(result)
    out = pd.DataFrame(rows)
    if not out["policy_level"].isin([0, 1, 2, 3]).all():
        raise ValueError("Invalid policy levels detected.")
    if out["policy_rationale"].astype(str).str.strip().eq("").any():
        raise ValueError("Empty policy rationale detected.")
    return out


def add_eval_labels(df: pd.DataFrame):
    out = df.copy()
    out["any_crisis_confirmed"] = (out["target_confirm"] == 1).astype(int)
    out["any_category_positive"] = (out[[f"target_category_{name}" for name in CATEGORY_THRESHOLDS]].sum(axis=1) > 0).astype(int)
    out["any_alert"] = (out["target_alert"] == 1).astype(int)
    out["high_risk_target"] = (
        (out["target_confirm"] == 1)
        | (out["target_category_SI_active"] == 1)
        | (out["target_category_SH"] == 1)
        | (out["target_category_SHA"] == 1)
    ).astype(int)
    return out


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
        return {"mean": None, "median": None, "missing": missing, "count": 0}
    return {
        "mean": float(np.mean(stats)),
        "median": float(np.median(stats)),
        "missing": int(missing),
        "count": int(len(stats)),
    }


def evaluate_split(df: pd.DataFrame):
    level_distribution = df["policy_level"].value_counts(normalize=True).sort_index().to_dict()
    action_distribution = df["policy_action"].value_counts(normalize=True).sort_index().to_dict()

    high_risk = df["high_risk_target"] == 1
    any_alert = df["target_alert"] == 1
    any_confirm = df["target_confirm"] == 1
    no_risk = (
        (df["target_alert"] == 0)
        & (df["target_confirm"] == 0)
        & (df[[f"target_category_{name}" for name in CATEGORY_THRESHOLDS]].sum(axis=1) == 0)
    )
    level0 = df["policy_level"] == 0
    level23 = df["policy_level"].isin([2, 3])
    level3 = df["policy_level"] == 3
    any_positive = (
        (df["target_alert"] == 1)
        | (df["target_confirm"] == 1)
        | (df[[f"target_category_{name}" for name in CATEGORY_THRESHOLDS]].sum(axis=1) > 0)
    )

    time_to_escalation = time_to_level(df, lambda g: g["policy_level"] >= 2)
    time_to_urgent = time_to_level(df, lambda g: g["policy_level"] == 3)

    return {
        "rows": int(len(df)),
        "policy_level_distribution": {int(k): float(v) for k, v in level_distribution.items()},
        "policy_action_distribution": {str(k): float(v) for k, v in action_distribution.items()},
        "unsafe_continuation_rate_high_risk": safe_rate(level0, high_risk),
        "unsafe_continuation_rate_alert": safe_rate(level0, any_alert),
        "unsafe_continuation_rate_confirm": safe_rate(level0, any_confirm),
        "over_escalation_burden": safe_rate(level23, no_risk),
        "urgent_escalation_precision": safe_rate(high_risk, level3),
        "human_defer_precision": safe_rate(any_positive, level23),
        "time_to_escalation": time_to_escalation,
        "time_to_urgent_escalation": time_to_urgent,
        "confusion_style_counts": {
            "high_risk_target_with_level_0": int((high_risk & (df["policy_level"] == 0)).sum()),
            "high_risk_target_with_level_1": int((high_risk & (df["policy_level"] == 1)).sum()),
            "high_risk_target_with_level_2": int((high_risk & (df["policy_level"] == 2)).sum()),
            "high_risk_target_with_level_3": int((high_risk & (df["policy_level"] == 3)).sum()),
            "no_risk_target_with_level_0": int((no_risk & (df["policy_level"] == 0)).sum()),
            "no_risk_target_with_level_1": int((no_risk & (df["policy_level"] == 1)).sum()),
            "no_risk_target_with_level_2": int((no_risk & (df["policy_level"] == 2)).sum()),
            "no_risk_target_with_level_3": int((no_risk & (df["policy_level"] == 3)).sum()),
        },
    }


def format_pct(value):
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    split_paths = {"train": TRAIN_PATH, "validation": VAL_PATH, "test": TEST_PATH}
    split_frames = {}
    leakage_found = {}
    for split_name, path in split_paths.items():
        df = load_split(path, split_name)
        leakage_found[split_name] = validate_inputs(df, split_name)
        df = build_flags(df)
        df = assign_policy(df)
        df = add_eval_labels(df)
        split_frames[split_name] = df
        out_path = OUTPUT_DIR / f"cradle_{split_name}_rule_based_policy_outputs.csv"
        df.to_csv(out_path, index=False)

    evals = {split_name: evaluate_split(df) for split_name, df in split_frames.items()}

    config = {
        "policy_levels": {
            "0": "continue_conversation",
            "1": "offer_supportive_resources",
            "2": "defer_to_human_support",
            "3": "urgent_escalation",
        },
        "thresholds": {
            "alert_tuned_threshold": ALERT_THRESHOLD,
            "confirm_tuned_threshold": CONFIRM_THRESHOLD,
            "category_tuned_thresholds": CATEGORY_THRESHOLDS,
        },
        "helper_flags": {
            "high_alert": f"alert_prob >= {ALERT_THRESHOLD}",
            "high_confirm": f"confirm_prob >= {CONFIRM_THRESHOLD}",
            "high_category_any": "any category probability >= its tuned threshold",
            "high_suicide_or_self_harm": "SI_active, SI_passive, SH, or SHA probability >= its tuned threshold",
            "high_interpersonal_violence": "DV, RA, or CA probability >= its tuned threshold",
            "rising_alert": "alert_prob_change_from_prev > 0.10",
            "rising_confirm": "confirm_prob_change_from_prev > 0.10",
            "high_accumulation": "risk_accumulation_score >= 0.75",
            "high_escalation_score": "risk_escalation_score >= 0.50",
            "volatile_risk": "risk_volatility_score >= 0.15",
        },
        "leakage_columns_excluded": LEAKAGE_COLUMNS,
        "evaluation_only_targets_not_used_as_policy_inputs": REQUIRED_TARGET_COLUMNS,
    }
    (RESULT_DIR / "rule_based_policy_config.json").write_text(json.dumps(config, indent=2) + "\n")

    report_lines = [
        "CRADLE Rule-Based Policy Evaluation",
        "===================================",
        "",
        "Audit Checks",
        "------------",
    ]
    for split_name, df in split_frames.items():
        report_lines.append(f"{split_name}_rows: {len(df)}")
        report_lines.append(f"{split_name}_required_prediction_columns_exist: True")
        report_lines.append(f"{split_name}_required_progression_feature_columns_exist: True")
        report_lines.append(f"{split_name}_required_target_columns_exist: True")
        report_lines.append(f"{split_name}_policy_inputs_do_not_include_target_labels: True")
        report_lines.append(f"{split_name}_leakage_columns_present_but_not_used: {json.dumps(leakage_found[split_name])}")
        report_lines.append(f"{split_name}_no_missing_probabilities: True")
        report_lines.append(f"{split_name}_policy_levels_valid: True")
        report_lines.append(f"{split_name}_non_empty_rationales: True")
        report_lines.append("")

    report_lines.extend(["Evaluation", "----------"])
    for split_name, metrics in evals.items():
        report_lines.append(split_name.upper())
        report_lines.append(json.dumps(metrics, indent=2, sort_keys=True))
        report_lines.append("")

    test_eval = evals["test"]
    if test_eval["unsafe_continuation_rate_high_risk"] is None:
        posture = "reasonable"
    elif test_eval["unsafe_continuation_rate_high_risk"] > 0.30:
        posture = "too conservative"
    elif (test_eval["over_escalation_burden"] or 0.0) > 0.25:
        posture = "too aggressive"
    else:
        posture = "reasonable"

    report_lines.extend(
        [
            "Final Beginner-Friendly Summary",
            "------------------------------",
            f"1. How many rows were processed for train, validation, and test? train={evals['train']['rows']}, validation={evals['validation']['rows']}, test={evals['test']['rows']}.",
            f"2. What percentage of test turns were assigned to each policy level? {json.dumps({k: format_pct(v) for k, v in test_eval['policy_level_distribution'].items()}, sort_keys=True)}.",
            f"3. What was the test unsafe continuation rate for high-risk turns? {format_pct(test_eval['unsafe_continuation_rate_high_risk'])}.",
            f"4. What was the test Alert unsafe continuation rate? {format_pct(test_eval['unsafe_continuation_rate_alert'])}.",
            f"5. What was the test Confirm unsafe continuation rate? {format_pct(test_eval['unsafe_continuation_rate_confirm'])}.",
            f"6. What was the test over-escalation burden? {format_pct(test_eval['over_escalation_burden'])}.",
            f"7. What was the test urgent escalation precision? {format_pct(test_eval['urgent_escalation_precision'])}.",
            f"8. What was the average test time-to-escalation? {test_eval['time_to_escalation']['mean']}.",
            f"9. Does the rule-based policy look too conservative, too aggressive, or reasonable? {posture}.",
            "10. What is the single next implementation step? Use this rule-based baseline as a reference point, then train and compare a learned policy model on the same progression feature dataset.",
        ]
    )
    (RESULT_DIR / "rule_based_policy_evaluation.txt").write_text("\n".join(report_lines) + "\n")

    print("Saved rule-based policy outputs:")
    for split_name in ["train", "validation", "test"]:
        print(f"- {OUTPUT_DIR / f'cradle_{split_name}_rule_based_policy_outputs.csv'}")
    print(f"Saved evaluation report to {RESULT_DIR / 'rule_based_policy_evaluation.txt'}")
    print(f"Saved policy config to {RESULT_DIR / 'rule_based_policy_config.json'}")
    print("")
    print("Final Beginner-Friendly Summary")
    print(f"1. How many rows were processed for train, validation, and test? train={evals['train']['rows']}, validation={evals['validation']['rows']}, test={evals['test']['rows']}.")
    print(f"2. What percentage of test turns were assigned to each policy level? {json.dumps({k: format_pct(v) for k, v in test_eval['policy_level_distribution'].items()}, sort_keys=True)}.")
    print(f"3. What was the test unsafe continuation rate for high-risk turns? {format_pct(test_eval['unsafe_continuation_rate_high_risk'])}.")
    print(f"4. What was the test Alert unsafe continuation rate? {format_pct(test_eval['unsafe_continuation_rate_alert'])}.")
    print(f"5. What was the test Confirm unsafe continuation rate? {format_pct(test_eval['unsafe_continuation_rate_confirm'])}.")
    print(f"6. What was the test over-escalation burden? {format_pct(test_eval['over_escalation_burden'])}.")
    print(f"7. What was the test urgent escalation precision? {format_pct(test_eval['urgent_escalation_precision'])}.")
    print(f"8. What was the average test time-to-escalation? {test_eval['time_to_escalation']['mean']}.")
    print(f"9. Does the rule-based policy look too conservative, too aggressive, or reasonable? {posture}.")
    print("10. What is the single next implementation step? Use this rule-based baseline as a reference point, then train and compare a learned policy model on the same progression feature dataset.")


if __name__ == "__main__":
    main()
