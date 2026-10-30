import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


CATEGORY_TARGETS = ["CA", "DV", "RA", "SHA", "SH", "SI_active", "SI_passive"]
SELF_HARM_CATEGORIES = {"SH", "SHA", "SI_active", "SI_passive"}

BASE_POLICY_DIR = Path("cradle_dialogue/policy_outputs_high_recall_confirm/safety_constrained_policy")
PROGRESSION_DIR = Path("cradle_dialogue/data/progression_features_high_recall_confirm")
OUTPUT_DIR = Path("cradle_dialogue/policy_outputs_high_recall_confirm_guardrail_v3/safety_constrained_policy")
RESULT_DIR = Path("cradle_dialogue/results/policy_high_recall_confirm_guardrail_v3")

OLD_CONFIRM_BASELINE_TEST = {
    "high_risk_unsafe_continuation": 0.2978723404255319,
    "confirm_unsafe_continuation": 0.2978723404255319,
    "over_escalation_burden": 0.11898535564853556,
}

HIGH_RECALL_BASELINE_TEST = {
    "high_risk_unsafe_continuation": 0.2297872340425532,
    "confirm_unsafe_continuation": 0.2297872340425532,
    "over_escalation_burden": 0.12238493723849372,
}

LEVEL_TO_ACTION = {
    0: "continue_conversation",
    1: "offer_supportive_resources",
    2: "defer_to_human_support",
    3: "urgent_escalation",
}


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_split(split: str):
    policy_path = BASE_POLICY_DIR / f"cradle_{split}_safety_constrained_policy_outputs.csv"
    progression_path = PROGRESSION_DIR / f"cradle_{split}_progression_features.csv"
    require_file(policy_path)
    require_file(progression_path)

    policy_df = pd.read_csv(policy_path)
    progression_cols = [
        "dialogue_id",
        "turn_id",
        "user_turn_index_in_dialogue",
        "max_category_prob",
        "top_category",
        "top_category_prob",
        "alert_prob_change_from_prev",
        "confirm_prob_change_from_prev",
    ]
    progression_df = pd.read_csv(progression_path, usecols=lambda col: col in progression_cols)
    merged = policy_df.merge(
        progression_df,
        on=["dialogue_id", "turn_id"],
        how="left",
        suffixes=("", "_from_progression"),
    )
    if "user_turn_index_in_dialogue" not in merged:
        merged["user_turn_index_in_dialogue"] = merged.groupby("dialogue_id").cumcount()
    merged["user_turn_index_in_dialogue"] = merged["user_turn_index_in_dialogue"].fillna(
        merged.groupby("dialogue_id").cumcount()
    )
    merged = add_eval_labels(merged)
    return merged


def add_eval_labels(df: pd.DataFrame):
    out = df.copy()
    category_cols = [f"target_category_{name}" for name in CATEGORY_TARGETS]
    high_risk_category = (
        (out["target_category_SI_active"] == 1)
        | (out["target_category_SH"] == 1)
        | (out["target_category_SHA"] == 1)
    )
    any_category_positive = out[category_cols].sum(axis=1) > 0
    policy_target = np.zeros(len(out), dtype=int)
    level3 = ((out["target_confirm"] == 1) & high_risk_category) | high_risk_category
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
    return out


def level_distributions(levels):
    values = pd.Series(levels).value_counts(normalize=True).sort_index().to_dict()
    return {int(level): float(values.get(level, 0.0)) for level in [0, 1, 2, 3]}


def safe_rate(numerator_mask, denominator_mask):
    denominator = int(denominator_mask.sum())
    if denominator == 0:
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


def classification_metrics(df: pd.DataFrame):
    y_true = df["policy_target"].astype(int)
    y_pred = df["policy_level"].astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
    }


def evaluate_policy(df: pd.DataFrame):
    category_cols = [f"target_category_{name}" for name in CATEGORY_TARGETS]
    high_risk = df["high_risk_target"] == 1
    any_alert = df["target_alert"] == 1
    any_confirm = df["target_confirm"] == 1
    no_risk = (
        (df["target_alert"] == 0)
        & (df["target_confirm"] == 0)
        & (df[category_cols].sum(axis=1) == 0)
    )
    level0 = df["policy_level"] == 0
    level23 = df["policy_level"].isin([2, 3])
    level3 = df["policy_level"] == 3
    any_positive = (
        (df["target_alert"] == 1)
        | (df["target_confirm"] == 1)
        | (df[category_cols].sum(axis=1) > 0)
    )
    metrics = {
        "rows": int(len(df)),
        "policy_level_distribution": level_distributions(df["policy_level"]),
        "unsafe_continuation_rate_high_risk": safe_rate(level0, high_risk),
        "unsafe_continuation_rate_alert": safe_rate(level0, any_alert),
        "unsafe_continuation_rate_confirm": safe_rate(level0, any_confirm),
        "over_escalation_burden": safe_rate(level23, no_risk),
        "urgent_escalation_precision": safe_rate(high_risk, level3),
        "human_defer_precision": safe_rate(any_positive, level23),
        "time_to_escalation": time_to_level(df, lambda g: g["policy_level"] >= 2),
        "time_to_urgent_escalation": time_to_level(df, lambda g: g["policy_level"] == 3),
    }
    metrics.update(classification_metrics(df))
    return metrics


def metric_value(metrics, name):
    if name.startswith("level_"):
        return metrics["policy_level_distribution"][int(name[-1])]
    if name == "high_risk_unsafe_continuation":
        return metrics["unsafe_continuation_rate_high_risk"]
    if name == "alert_unsafe_continuation":
        return metrics["unsafe_continuation_rate_alert"]
    if name == "confirm_unsafe_continuation":
        return metrics["unsafe_continuation_rate_confirm"]
    if name == "over_escalation_burden":
        return metrics["over_escalation_burden"]
    if name == "urgent_escalation_precision":
        return metrics["urgent_escalation_precision"]
    if name == "human_defer_precision":
        return metrics["human_defer_precision"]
    if name == "mean_time_to_escalation":
        return metrics["time_to_escalation"]["mean"]
    if name == "median_time_to_escalation":
        return metrics["time_to_escalation"]["median"]
    if name == "no_escalation_count":
        return metrics["time_to_escalation"]["missing"]
    if name == "mean_time_to_urgent_escalation":
        return metrics["time_to_urgent_escalation"]["mean"]
    if name == "no_urgent_escalation_count":
        return metrics["time_to_urgent_escalation"]["missing"]
    if name in {"accuracy", "macro_f1", "weighted_f1"}:
        return metrics[name]
    raise KeyError(name)


def make_rules():
    return {
        "L1_confirm_ge_0.003": (1, lambda df: df["confirm_prob"] >= 0.003),
        "L1_confirm_ge_0.005": (1, lambda df: df["confirm_prob"] >= 0.005),
        "L1_confirm_ge_0.01": (1, lambda df: df["confirm_prob"] >= 0.01),
        "L1_max_category_ge_0.20": (1, lambda df: df["max_category_prob"] >= 0.20),
        "L1_max_category_ge_0.30": (1, lambda df: df["max_category_prob"] >= 0.30),
        "L1_risk_volatility_ge_0.20": (1, lambda df: df["risk_volatility_score"] >= 0.20),
        "L1_risk_escalation_ge_0.20": (1, lambda df: df["risk_escalation_score"] >= 0.20),
        "L1_alert_ge_0.50": (1, lambda df: df["alert_prob"] >= 0.50),
        "L2_confirm_ge_0.005_and_acc_ge_0.60": (
            2,
            lambda df: (df["confirm_prob"] >= 0.005) & (df["risk_accumulation_score"] >= 0.60),
        ),
        "L2_confirm_ge_0.005_and_max_category_ge_0.20": (
            2,
            lambda df: (df["confirm_prob"] >= 0.005) & (df["max_category_prob"] >= 0.20),
        ),
        "L2_confirm_ge_0.01_and_acc_ge_0.60": (
            2,
            lambda df: (df["confirm_prob"] >= 0.01) & (df["risk_accumulation_score"] >= 0.60),
        ),
        "L2_confirm_ge_0.01_and_max_category_ge_0.20": (
            2,
            lambda df: (df["confirm_prob"] >= 0.01) & (df["max_category_prob"] >= 0.20),
        ),
        "L2_max_category_ge_0.30": (2, lambda df: df["max_category_prob"] >= 0.30),
        "L2_risk_escalation_ge_0.20": (2, lambda df: df["risk_escalation_score"] >= 0.20),
        "L2_risk_volatility_ge_0.20": (2, lambda df: df["risk_volatility_score"] >= 0.20),
        "L2_alert_ge_0.50_and_acc_ge_0.60": (
            2,
            lambda df: (df["alert_prob"] >= 0.50) & (df["risk_accumulation_score"] >= 0.60),
        ),
        "L2_alert_ge_0.50_and_max_category_ge_0.20": (
            2,
            lambda df: (df["alert_prob"] >= 0.50) & (df["max_category_prob"] >= 0.20),
        ),
        "L3_confirm_ge_0.05": (3, lambda df: df["confirm_prob"] >= 0.05),
        "L3_confirm_ge_0.02_and_max_category_ge_0.30": (
            3,
            lambda df: (df["confirm_prob"] >= 0.02) & (df["max_category_prob"] >= 0.30),
        ),
        "L3_confirm_ge_0.01_and_max_category_ge_0.50": (
            3,
            lambda df: (df["confirm_prob"] >= 0.01) & (df["max_category_prob"] >= 0.50),
        ),
        "L3_self_harm_max_category_ge_0.50": (
            3,
            lambda df: (df["max_category_prob"] >= 0.50) & df["top_category"].isin(SELF_HARM_CATEGORIES),
        ),
        "L3_risk_escalation_ge_0.30_and_max_category_ge_0.20": (
            3,
            lambda df: (df["risk_escalation_score"] >= 0.30) & (df["max_category_prob"] >= 0.20),
        ),
    }


def generate_candidate_configs():
    rules = make_rules()
    level1 = [name for name in rules if name.startswith("L1_")]
    level2 = [name for name in rules if name.startswith("L2_")]
    level3 = [name for name in rules if name.startswith("L3_")]

    configs = {tuple()}
    for name in rules:
        configs.add((name,))
    for group in [level1, level2, level3]:
        for pair in combinations(group, 2):
            configs.add(tuple(sorted(pair)))
    for combo in product([None] + level1, [None] + level2, [None] + level3):
        names = tuple(sorted(name for name in combo if name is not None))
        configs.add(names)
    return sorted(configs, key=lambda item: (len(item), item))


def apply_config(df: pd.DataFrame, config):
    rules = make_rules()
    out = df.copy()
    learned_levels = out["learned_policy_level_before_guardrail"].astype(int).to_numpy()
    minimum = np.zeros(len(out), dtype=int)
    reasons = [[] for _ in range(len(out))]

    for rule_name in config:
        min_level, fn = rules[rule_name]
        mask = fn(out).fillna(False).to_numpy()
        update = mask & (minimum < min_level)
        minimum[update] = min_level
        for idx in np.where(mask)[0]:
            reasons[idx].append(rule_name)

    out["v3_guardrail_minimum_level"] = minimum
    out["v3_guardrail_triggered"] = (minimum > learned_levels).astype(int)
    out["v3_guardrail_reason"] = [";".join(reason) for reason in reasons]
    out["policy_level_before_v3"] = out["policy_level"].astype(int)
    out["policy_level"] = np.maximum(learned_levels, minimum).astype(int)
    out["policy_action"] = out["policy_level"].map(LEVEL_TO_ACTION)
    out["guardrail_variant"] = "v3_validation_selected" if config else "high_recall_confirm_baseline_kept"
    out["guardrail_minimum_level"] = out["v3_guardrail_minimum_level"]
    out["guardrail_triggered"] = out["v3_guardrail_triggered"]
    out["guardrail_reason"] = out["v3_guardrail_reason"]
    out["policy_rationale"] = [
        explain_rationale(level, triggered, reason)
        for level, triggered, reason in zip(out["policy_level"], out["v3_guardrail_triggered"], out["v3_guardrail_reason"])
    ]
    return out


def keep_current_policy(df: pd.DataFrame):
    out = df.copy()
    out["policy_level_before_v3"] = out["policy_level"].astype(int)
    out["v3_guardrail_minimum_level"] = 0
    out["v3_guardrail_triggered"] = 0
    out["v3_guardrail_reason"] = ""
    out["guardrail_variant"] = "high_recall_confirm_baseline_kept"
    return out


def explain_rationale(level, triggered, reason):
    if int(triggered) == 1:
        if int(level) == 3:
            return f"Urgent escalation because validation-selected v3 guardrail triggered: {reason}."
        if int(level) == 2:
            return f"Defer to human support because validation-selected v3 guardrail triggered: {reason}."
        return f"Offer supportive resources because validation-selected v3 guardrail triggered: {reason}."
    if int(level) == 3:
        return "Urgent escalation because the learned policy predicted elevated crisis risk."
    if int(level) == 2:
        return "Defer to human support because the learned policy predicted elevated escalation need."
    if int(level) == 1:
        return "Offer supportive resources because the learned policy predicted moderate support need."
    return "Continue conversation because the learned policy and validation-selected v3 guardrails found low immediate risk."


def flatten_metrics(prefix, metrics):
    row = {}
    names = [
        "level_0_percentage",
        "level_1_percentage",
        "level_2_percentage",
        "level_3_percentage",
        "high_risk_unsafe_continuation",
        "alert_unsafe_continuation",
        "confirm_unsafe_continuation",
        "over_escalation_burden",
        "urgent_escalation_precision",
        "human_defer_precision",
        "mean_time_to_escalation",
        "median_time_to_escalation",
        "no_escalation_count",
        "mean_time_to_urgent_escalation",
        "no_urgent_escalation_count",
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ]
    mapping = {
        "level_0_percentage": "level_0",
        "level_1_percentage": "level_1",
        "level_2_percentage": "level_2",
        "level_3_percentage": "level_3",
    }
    for name in names:
        key = mapping.get(name, name)
        row[f"{prefix}_{name}"] = metric_value(metrics, key)
    return row


def build_search_results(validation_df, baseline_validation_metrics):
    rules = make_rules()
    rule_masks = {name: fn(validation_df).fillna(False).to_numpy() for name, (_, fn) in rules.items()}
    rule_levels = {name: level for name, (level, _) in rules.items()}
    learned_levels = validation_df["learned_policy_level_before_guardrail"].astype(int).to_numpy()
    high_risk = (validation_df["high_risk_target"] == 1).to_numpy()
    any_alert = (validation_df["target_alert"] == 1).to_numpy()
    any_confirm = (validation_df["target_confirm"] == 1).to_numpy()
    category_cols = [f"target_category_{name}" for name in CATEGORY_TARGETS]
    no_risk = (
        (validation_df["target_alert"] == 0)
        & (validation_df["target_confirm"] == 0)
        & (validation_df[category_cols].sum(axis=1) == 0)
    ).to_numpy()
    any_positive = (
        (validation_df["target_alert"] == 1)
        | (validation_df["target_confirm"] == 1)
        | (validation_df[category_cols].sum(axis=1) > 0)
    ).to_numpy()
    y_true = validation_df["policy_target"].astype(int).to_numpy()

    def rate(mask, denom):
        count = int(denom.sum())
        if count == 0:
            return None
        return float(mask[denom].mean())

    def fast_metrics(config):
        minimum = np.zeros(len(validation_df), dtype=int)
        for rule_name in config:
            mask = rule_masks[rule_name]
            level = rule_levels[rule_name]
            minimum[mask & (minimum < level)] = level
        levels = np.maximum(learned_levels, minimum).astype(int)
        level0 = levels == 0
        level23 = np.isin(levels, [2, 3])
        level3 = levels == 3
        distribution = level_distributions(levels)
        return {
            "policy_level_distribution": distribution,
            "unsafe_continuation_rate_high_risk": rate(level0, high_risk),
            "unsafe_continuation_rate_alert": rate(level0, any_alert),
            "unsafe_continuation_rate_confirm": rate(level0, any_confirm),
            "over_escalation_burden": rate(level23, no_risk),
            "urgent_escalation_precision": rate(high_risk, level3),
            "human_defer_precision": rate(any_positive, level23),
            "time_to_escalation": {"mean": None, "median": None, "missing": None, "count": None},
            "time_to_urgent_escalation": {"mean": None, "median": None, "missing": None, "count": None},
            "accuracy": float(accuracy_score(y_true, levels)),
            "macro_f1": float(f1_score(y_true, levels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, levels, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, levels, labels=[0, 1, 2, 3]).tolist(),
        }

    rows = []
    for config in generate_candidate_configs():
        metrics = fast_metrics(config)
        row = {
            "config_id": "baseline_learned_only" if not config else "__".join(config),
            "rules": json.dumps(list(config)),
            "num_rules": len(config),
            "passes_over_escalation_constraint": (
                metrics["over_escalation_burden"] is not None and metrics["over_escalation_burden"] <= 0.20
            ),
            "improves_validation_high_risk_unsafe": (
                metrics["unsafe_continuation_rate_high_risk"]
                < baseline_validation_metrics["unsafe_continuation_rate_high_risk"]
            ),
        }
        row.update(flatten_metrics("validation", metrics))
        row["validation_confusion_matrix"] = json.dumps(metrics["confusion_matrix"])
        rows.append(row)
    return pd.DataFrame(rows)


def select_config(search_df):
    eligible = search_df[
        (search_df["passes_over_escalation_constraint"])
        & (search_df["improves_validation_high_risk_unsafe"])
    ].copy()
    if eligible.empty:
        return tuple(), False, "No validation candidate improved high-risk unsafe continuation while keeping over-escalation <= 0.20."
    eligible = eligible.sort_values(
        by=[
            "validation_high_risk_unsafe_continuation",
            "validation_confirm_unsafe_continuation",
            "validation_alert_unsafe_continuation",
            "validation_over_escalation_burden",
            "validation_urgent_escalation_precision",
            "validation_human_defer_precision",
        ],
        ascending=[True, True, True, True, False, False],
    )
    selected = eligible.iloc[0]
    return tuple(json.loads(selected["rules"])), True, "Selected by validation-only rule under over-escalation <= 0.20."


def write_outputs(split_frames):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keep_first = [
        "split",
        "dialogue_id",
        "turn_id",
        "speaker_role",
        "text",
        "policy_level",
        "policy_action",
        "policy_rationale",
        "learned_policy_level_before_guardrail",
        "policy_level_before_v3",
        "guardrail_minimum_level",
        "guardrail_variant",
        "guardrail_triggered",
        "guardrail_reason",
        "v3_guardrail_minimum_level",
        "v3_guardrail_triggered",
        "v3_guardrail_reason",
    ]
    for split, frame in split_frames.items():
        columns = [col for col in keep_first if col in frame.columns] + [
            col for col in frame.columns if col not in keep_first
        ]
        frame[columns].to_csv(OUTPUT_DIR / f"cradle_{split}_safety_constrained_policy_outputs.csv", index=False)


def report_metric_line(label, metrics):
    return (
        f"{label}: high-risk unsafe={metrics['unsafe_continuation_rate_high_risk']}, "
        f"confirm unsafe={metrics['unsafe_continuation_rate_confirm']}, "
        f"alert unsafe={metrics['unsafe_continuation_rate_alert']}, "
        f"over-escalation={metrics['over_escalation_burden']}, "
        f"urgent precision={metrics['urgent_escalation_precision']}, "
        f"human defer precision={metrics['human_defer_precision']}, "
        f"macro F1={metrics['macro_f1']}"
    )


def write_report(selection, baseline_val, selected_val, baseline_test, selected_test, selected_config, selected_flag, reason):
    lines = [
        "CRADLE High-Recall Confirm Guardrail v3 Validation Search",
        "=========================================================",
        "",
        "Selection Rule",
        "--------------",
        "Selection used validation metrics only. Test metrics were computed once after the validation-selected decision.",
        "Hard constraint: validation over-escalation burden <= 0.20.",
        "Primary objective: lowest validation high-risk unsafe continuation among candidates that improve over the high-recall baseline.",
        "",
        "Selected Config",
        "---------------",
        f"v3_selected: {selected_flag}",
        f"selection_reason: {reason}",
        f"selected_rules: {json.dumps(list(selected_config))}",
        "",
        "Validation Metrics",
        "------------------",
        report_metric_line("High-recall Confirm baseline", baseline_val),
        report_metric_line("Guardrail v3 selected/kept", selected_val),
        f"validation_confusion_matrix_baseline: {json.dumps(baseline_val['confusion_matrix'])}",
        f"validation_confusion_matrix_v3: {json.dumps(selected_val['confusion_matrix'])}",
        "",
        "Test Metrics",
        "------------",
        f"Old safety-constrained old Confirm baseline: {json.dumps(OLD_CONFIRM_BASELINE_TEST, sort_keys=True)}",
        f"High-recall Confirm baseline constants: {json.dumps(HIGH_RECALL_BASELINE_TEST, sort_keys=True)}",
        report_metric_line("High-recall Confirm baseline recomputed", baseline_test),
        report_metric_line("Guardrail v3 selected/kept", selected_test),
        f"test_confusion_matrix_baseline: {json.dumps(baseline_test['confusion_matrix'])}",
        f"test_confusion_matrix_v3: {json.dumps(selected_test['confusion_matrix'])}",
        "",
        "Top Validation Candidates Under Constraint",
        "------------------------------------------",
    ]
    top = selection[selection["passes_over_escalation_constraint"]].sort_values(
        by=[
            "validation_high_risk_unsafe_continuation",
            "validation_confirm_unsafe_continuation",
            "validation_alert_unsafe_continuation",
            "validation_over_escalation_burden",
        ],
        ascending=[True, True, True, True],
    ).head(15)
    for _, row in top.iterrows():
        lines.append(
            json.dumps(
                {
                    "config_id": row["config_id"],
                    "rules": json.loads(row["rules"]),
                    "validation_high_risk_unsafe": row["validation_high_risk_unsafe_continuation"],
                    "validation_confirm_unsafe": row["validation_confirm_unsafe_continuation"],
                    "validation_alert_unsafe": row["validation_alert_unsafe_continuation"],
                    "validation_over_escalation": row["validation_over_escalation_burden"],
                    "validation_urgent_precision": row["validation_urgent_escalation_precision"],
                    "validation_human_defer_precision": row["validation_human_defer_precision"],
                },
                sort_keys=True,
            )
        )
    lines.extend(
        [
            "",
            "Final Beginner-Friendly Summary",
            "-------------------------------",
            f"1. Did any validation-selected v3 guardrail improve unsafe continuation? {'Yes' if selected_flag else 'No'}.",
            f"2. Did it keep validation over-escalation <= 0.20? {'Yes' if selected_val['over_escalation_burden'] <= 0.20 else 'No'}.",
            f"3. Selected config: {json.dumps(list(selected_config))}.",
            f"4. Validation unsafe before/after: {baseline_val['unsafe_continuation_rate_high_risk']} -> {selected_val['unsafe_continuation_rate_high_risk']}.",
            f"5. Validation over-escalation before/after: {baseline_val['over_escalation_burden']} -> {selected_val['over_escalation_burden']}.",
            f"6. Test unsafe before/after: {baseline_test['unsafe_continuation_rate_high_risk']} -> {selected_test['unsafe_continuation_rate_high_risk']}.",
            f"7. Test over-escalation before/after: {baseline_test['over_escalation_burden']} -> {selected_test['over_escalation_burden']}.",
            f"8. Is v3 better than the high-recall baseline? {'Yes, by the validation rule.' if selected_flag else 'No, not by the validation rule.'}",
            f"9. Is the improvement large enough to justify rule complexity? {'Maybe; review the selected rules and validation/test delta before freezing it.' if selected_flag else 'No; v3 is not justified.'}",
            f"10. Should v3 become current final policy? {'Only if the small validation-selected improvement is worth the added rule complexity.' if selected_flag else 'No; keep the current high-recall Confirm safety-constrained policy.'}",
            "11. What is the single next implementation step? Freeze this decision in the project notes; do not tune from the test set.",
        ]
    )
    (RESULT_DIR / "guardrail_v3_summary.txt").write_text("\n".join(lines) + "\n")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    splits = {split: load_split(split) for split in ["train", "validation", "test"]}

    baseline_val_metrics = evaluate_policy(splits["validation"])
    baseline_test_metrics = evaluate_policy(splits["test"])
    search_df = build_search_results(splits["validation"], baseline_val_metrics)
    search_df = search_df.sort_values(
        by=[
            "passes_over_escalation_constraint",
            "improves_validation_high_risk_unsafe",
            "validation_high_risk_unsafe_continuation",
            "validation_confirm_unsafe_continuation",
            "validation_alert_unsafe_continuation",
            "validation_over_escalation_burden",
            "validation_urgent_escalation_precision",
            "validation_human_defer_precision",
        ],
        ascending=[False, False, True, True, True, True, False, False],
    )
    search_df.to_csv(RESULT_DIR / "validation_guardrail_search_results.csv", index=False)

    selected_config, selected_flag, reason = select_config(search_df)
    if selected_flag:
        selected_frames = {split: apply_config(frame, selected_config) for split, frame in splits.items()}
    else:
        selected_frames = {split: keep_current_policy(frame) for split, frame in splits.items()}
    selected_val_metrics = evaluate_policy(selected_frames["validation"])
    selected_test_metrics = evaluate_policy(selected_frames["test"])
    write_outputs(selected_frames)

    config_payload = {
        "v3_selected": selected_flag,
        "selection_reason": reason,
        "selected_rules": list(selected_config),
        "selected_using": "validation_only",
        "hard_constraint": {"validation_over_escalation_burden_max": 0.20},
        "baseline_validation_metrics": baseline_val_metrics,
        "selected_validation_metrics": selected_val_metrics,
        "baseline_test_metrics": baseline_test_metrics,
        "selected_test_metrics": selected_test_metrics,
        "test_metrics_used_for_selection": False,
    }
    (RESULT_DIR / "selected_guardrail_config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True))
    write_report(
        search_df,
        baseline_val_metrics,
        selected_val_metrics,
        baseline_test_metrics,
        selected_test_metrics,
        selected_config,
        selected_flag,
        reason,
    )
    print(f"wrote {RESULT_DIR / 'validation_guardrail_search_results.csv'}")
    print(f"wrote {RESULT_DIR / 'selected_guardrail_config.json'}")
    print(f"wrote {RESULT_DIR / 'guardrail_v3_summary.txt'}")
    print(f"wrote policy outputs under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
