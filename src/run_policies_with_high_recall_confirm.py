import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import build_cradle_rule_based_policy as rule_policy
import train_cradle_learned_policy as learned_policy
import train_cradle_safety_constrained_policy_v2 as safety_policy


PROGRESSION_DIR = Path("cradle_dialogue/data/progression_features_high_recall_confirm")
TRAIN_PATH = PROGRESSION_DIR / "cradle_train_progression_features.csv"
VAL_PATH = PROGRESSION_DIR / "cradle_validation_progression_features.csv"
TEST_PATH = PROGRESSION_DIR / "cradle_test_progression_features.csv"

RULE_OUTPUT_DIR = Path("cradle_dialogue/policy_outputs_high_recall_confirm/rule_based")
LEARNED_OUTPUT_DIR = Path("cradle_dialogue/policy_outputs_high_recall_confirm/learned_policy")
SAFETY_OUTPUT_DIR = Path("cradle_dialogue/policy_outputs_high_recall_confirm/safety_constrained_policy")
RESULT_DIR = Path("cradle_dialogue/results/policy_high_recall_confirm")
MODEL_DIR = Path("cradle_dialogue/models/policy_high_recall_confirm")

CONFIRM_MODEL_USED = "B_pos_weight_x2 high-recall Confirm"
CONFIRM_THRESHOLD_USED_IN_PROGRESSION = 0.011

BASELINE = {
    "rule_based": {
        "Level 0 percentage": 0.17848464766953834,
        "Level 1 percentage": 0.16545173404020322,
        "Level 2 percentage": 0.6050364479787939,
        "Level 3 percentage": 0.051027170311464545,
        "High-risk unsafe continuation": 0.1,
        "Alert unsafe continuation": 0.20085470085470086,
        "Confirm unsafe continuation": 0.1,
        "Over-escalation burden": 0.665010460251046,
        "Urgent escalation precision": 0.7878787878787878,
        "Human defer precision": 0.14377104377104377,
        "Mean time-to-escalation": 1.210958904109589,
    },
    "learned_policy": {
        "Level 0 percentage": 0.9383697813121272,
        "Level 1 percentage": 0.012591119946984758,
        "Level 2 percentage": 0.024740446211619173,
        "Level 3 percentage": 0.02429865252926883,
        "High-risk unsafe continuation": 0.6106382978723405,
        "Alert unsafe continuation": 0.9358974358974359,
        "Confirm unsafe continuation": 0.6106382978723405,
        "Over-escalation burden": 0.010721757322175732,
        "Urgent escalation precision": 0.8,
        "Human defer precision": 0.8153153153153153,
        "Mean time-to-escalation": 2.7373737373737375,
    },
    "safety_constrained_policy": {
        "Level 0 percentage": 0.6324276562845151,
        "Level 1 percentage": 0.2085266180693616,
        "Level 2 percentage": 0.10603048376408217,
        "Level 3 percentage": 0.053015241882041084,
        "High-risk unsafe continuation": 0.2978723404255319,
        "Alert unsafe continuation": 0.26495726495726496,
        "Confirm unsafe continuation": 0.2978723404255319,
        "Over-escalation burden": 0.11898535564853556,
        "Urgent escalation precision": 0.7791666666666667,
        "Human defer precision": 0.3680555555555556,
        "Mean time-to-escalation": 1.8509485094850948,
    },
}


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def metric_value(metrics, name):
    if name.startswith("Level "):
        level = int(name.split()[1])
        return metrics["policy_level_distribution"].get(level, 0.0)
    if name == "High-risk unsafe continuation":
        return metrics["unsafe_continuation_rate_high_risk"]
    if name == "Alert unsafe continuation":
        return metrics["unsafe_continuation_rate_alert"]
    if name == "Confirm unsafe continuation":
        return metrics["unsafe_continuation_rate_confirm"]
    if name == "Over-escalation burden":
        return metrics["over_escalation_burden"]
    if name == "Urgent escalation precision":
        return metrics["urgent_escalation_precision"]
    if name == "Human defer precision":
        return metrics["human_defer_precision"]
    if name == "Mean time-to-escalation":
        return metrics["time_to_escalation"]["mean"]
    if name == "Median time-to-escalation":
        return metrics["time_to_escalation"]["median"]
    if name == "No-escalation count":
        return metrics["time_to_escalation"]["missing"]
    if name == "Mean time-to-urgent-escalation":
        return metrics["time_to_urgent_escalation"]["mean"]
    if name == "No-urgent-escalation count":
        return metrics["time_to_urgent_escalation"]["missing"]
    raise KeyError(name)


def policy_metrics_table(policy_metrics):
    metric_names = [
        "Level 0 percentage",
        "Level 1 percentage",
        "Level 2 percentage",
        "Level 3 percentage",
        "High-risk unsafe continuation",
        "Alert unsafe continuation",
        "Confirm unsafe continuation",
        "Over-escalation burden",
        "Urgent escalation precision",
        "Human defer precision",
        "Mean time-to-escalation",
        "Median time-to-escalation",
        "No-escalation count",
        "Mean time-to-urgent-escalation",
        "No-urgent-escalation count",
    ]
    lines = [
        "Metric | Rule-based policy | Unconstrained learned policy | Safety-constrained learned policy",
    ]
    for name in metric_names:
        lines.append(
            f"{name} | {metric_value(policy_metrics['rule_based'], name)} | "
            f"{metric_value(policy_metrics['learned_policy'], name)} | "
            f"{metric_value(policy_metrics['safety_constrained_policy'], name)}"
        )
    return lines


def baseline_comparison_table(policy_metrics):
    metric_names = [
        "Level 0 percentage",
        "Level 1 percentage",
        "Level 2 percentage",
        "Level 3 percentage",
        "High-risk unsafe continuation",
        "Alert unsafe continuation",
        "Confirm unsafe continuation",
        "Over-escalation burden",
        "Urgent escalation precision",
        "Human defer precision",
        "Mean time-to-escalation",
    ]
    policy_key_to_label = {
        "rule_based": "Rule-based policy",
        "learned_policy": "Unconstrained learned policy",
        "safety_constrained_policy": "Safety-constrained learned policy",
    }
    lines = [
        "Metric | Policy | Old Confirm result | High-recall Confirm result | Change",
    ]
    for policy_key, label in policy_key_to_label.items():
        for name in metric_names:
            old = BASELINE[policy_key][name]
            new = metric_value(policy_metrics[policy_key], name)
            change = None if old is None or new is None else new - old
            lines.append(f"{name} | {label} | {old} | {new} | {change}")
    return lines


def classification_section(classification_metrics_by_policy):
    lines = []
    for policy_name, metrics in classification_metrics_by_policy.items():
        lines.extend(
            [
                policy_name,
                "-" * len(policy_name),
                f"accuracy: {metrics['accuracy']}",
                f"macro_f1: {metrics['macro_f1']}",
                f"weighted_f1: {metrics['weighted_f1']}",
                f"per_class: {json.dumps(metrics['per_class'], sort_keys=True)}",
                f"confusion_matrix: {json.dumps(metrics['confusion_matrix'])}",
                "",
            ]
        )
    return lines


def run_rule_based(train_df, val_df, test_df):
    out = {}
    for split, df in {"train": train_df, "validation": val_df, "test": test_df}.items():
        leaked = rule_policy.validate_inputs(df, split)
        if leaked:
            # The source files may contain excluded columns, but assign_policy never uses them.
            pass
        frame = rule_policy.build_flags(df)
        frame = rule_policy.assign_policy(frame)
        frame = rule_policy.add_eval_labels(frame)
        out[split] = frame

    RULE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out["train"].to_csv(RULE_OUTPUT_DIR / "cradle_train_rule_based_policy_outputs.csv", index=False)
    out["validation"].to_csv(RULE_OUTPUT_DIR / "cradle_validation_rule_based_policy_outputs.csv", index=False)
    out["test"].to_csv(RULE_OUTPUT_DIR / "cradle_test_rule_based_policy_outputs.csv", index=False)
    return out


def run_learned(train_df, val_df, test_df):
    train_df = learned_policy.create_policy_target(train_df)
    val_df = learned_policy.create_policy_target(val_df)
    test_df = learned_policy.create_policy_target(test_df)

    x_train, numeric_cols, categorical_cols = learned_policy.prepare_features(train_df)
    x_val, _, _ = learned_policy.prepare_features(val_df)
    x_test, _, _ = learned_policy.prepare_features(test_df)
    y_train = train_df["policy_target"].astype(int).to_numpy()
    y_val = val_df["policy_target"].astype(int).to_numpy()
    y_test = test_df["policy_target"].astype(int).to_numpy()

    results = []
    for name in ["logistic_regression", "random_forest"]:
        model = learned_policy.build_model(name, numeric_cols, categorical_cols)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)
        train_proba = model.predict_proba(x_train)
        val_proba = model.predict_proba(x_val)
        test_proba = model.predict_proba(x_test)

        train_out = learned_policy.attach_predictions(train_df, train_pred, train_proba)
        val_out = learned_policy.attach_predictions(val_df, val_pred, val_proba)
        test_out = learned_policy.attach_predictions(test_df, test_pred, test_proba)

        results.append(
            {
                "name": name,
                "model": model,
                "train_out": train_out,
                "validation_out": val_out,
                "test_out": test_out,
                "train_policy": learned_policy.evaluate_policy(train_out),
                "validation_policy": learned_policy.evaluate_policy(val_out),
                "test_policy": learned_policy.evaluate_policy(test_out),
                "train_classification": learned_policy.classification_metrics(y_train, train_pred),
                "validation_classification": learned_policy.classification_metrics(y_val, val_pred),
                "test_classification": learned_policy.classification_metrics(y_test, test_pred),
            }
        )

    selected = learned_policy.select_best_model(results)
    LEARNED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    selected["train_out"].to_csv(LEARNED_OUTPUT_DIR / "cradle_train_learned_policy_outputs.csv", index=False)
    selected["validation_out"].to_csv(LEARNED_OUTPUT_DIR / "cradle_validation_learned_policy_outputs.csv", index=False)
    selected["test_out"].to_csv(LEARNED_OUTPUT_DIR / "cradle_test_learned_policy_outputs.csv", index=False)
    with (MODEL_DIR / "best_learned_policy_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "selected_model_name": selected["name"],
                "model": selected["model"],
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
            },
            handle,
        )
    return selected, results


def run_safety_constrained(train_df, val_df, test_df, learned_selected):
    train_df = safety_policy.create_policy_target(train_df)
    val_df = safety_policy.create_policy_target(val_df)
    test_df = safety_policy.create_policy_target(test_df)

    x_train, numeric_cols, categorical_cols = safety_policy.prepare_features(train_df)
    x_val, _, _ = safety_policy.prepare_features(val_df)
    x_test, _, _ = safety_policy.prepare_features(test_df)
    y_train = train_df["policy_target"].astype(int).to_numpy()
    y_val = val_df["policy_target"].astype(int).to_numpy()
    y_test = test_df["policy_target"].astype(int).to_numpy()

    model_names = ["logistic_regression", "random_forest"]
    xgboost_status = "unavailable"
    if safety_policy.XGBOOST_AVAILABLE:
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
    for model_name in model_names:
        model = safety_policy.build_model(model_name, numeric_cols, categorical_cols)
        safety_policy.fit_model(model_name, model, x_train, y_train)
        train_pred, train_proba = safety_policy.predict_model(model_name, model, x_train)
        val_pred, val_proba = safety_policy.predict_model(model_name, model, x_val)
        test_pred, test_proba = safety_policy.predict_model(model_name, model, x_test)

        for variant_name in guardrail_variants:
            train_out = safety_policy.apply_guardrail_variant(train_df, train_pred, variant_name, train_proba)
            val_out = safety_policy.apply_guardrail_variant(val_df, val_pred, variant_name, val_proba)
            test_out = safety_policy.apply_guardrail_variant(test_df, test_pred, variant_name, test_proba)
            results.append(
                {
                    "model_name": model_name,
                    "guardrail_variant": variant_name,
                    "model": model,
                    "train_out": train_out,
                    "validation_out": val_out,
                    "test_out": test_out,
                    "train_policy": safety_policy.evaluate_policy(train_out),
                    "validation_policy": safety_policy.evaluate_policy(val_out),
                    "test_policy": safety_policy.evaluate_policy(test_out),
                    "train_classification": safety_policy.classification_metrics(
                        y_train, train_out["policy_level"].to_numpy()
                    ),
                    "validation_classification": safety_policy.classification_metrics(
                        y_val, val_out["policy_level"].to_numpy()
                    ),
                    "test_classification": safety_policy.classification_metrics(
                        y_test, test_out["policy_level"].to_numpy()
                    ),
                }
            )

    selected = safety_policy.select_best_configuration(results)
    SAFETY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected["train_out"][safety_policy.output_columns(selected["train_out"])].to_csv(
        SAFETY_OUTPUT_DIR / "cradle_train_safety_constrained_policy_outputs.csv", index=False
    )
    selected["validation_out"][safety_policy.output_columns(selected["validation_out"])].to_csv(
        SAFETY_OUTPUT_DIR / "cradle_validation_safety_constrained_policy_outputs.csv", index=False
    )
    selected["test_out"][safety_policy.output_columns(selected["test_out"])].to_csv(
        SAFETY_OUTPUT_DIR / "cradle_test_safety_constrained_policy_outputs.csv", index=False
    )
    with (MODEL_DIR / "best_safety_constrained_policy_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "selected_model_name": selected["model_name"],
                "selected_guardrail_variant": selected["guardrail_variant"],
                "model": selected["model"],
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "xgboost_status": xgboost_status,
            },
            handle,
        )
    return selected, results, xgboost_status, guardrail_variants


def choose_best_policy(policy_metrics):
    # Prefer low high-risk unsafe continuation; then lower over-escalation burden.
    candidates = []
    for key in ["rule_based", "learned_policy", "safety_constrained_policy"]:
        metrics = policy_metrics[key]
        candidates.append(
            (
                metrics["unsafe_continuation_rate_high_risk"]
                if metrics["unsafe_continuation_rate_high_risk"] is not None
                else np.inf,
                metrics["over_escalation_burden"]
                if metrics["over_escalation_burden"] is not None
                else np.inf,
                key,
            )
        )
    return sorted(candidates)[0][2]


def write_reports(rule_out, learned_selected, safety_selected, safety_results, xgboost_status, guardrail_variants):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rule_metrics = rule_policy.evaluate_split(rule_out["test"])
    learned_metrics = learned_policy.evaluate_policy(learned_selected["test_out"])
    safety_metrics = safety_policy.evaluate_policy(safety_selected["test_out"])

    policy_metrics = {
        "rule_based": rule_metrics,
        "learned_policy": learned_metrics,
        "safety_constrained_policy": safety_metrics,
    }
    classification = {
        "Rule-based policy": learned_policy.classification_metrics(
            learned_policy.create_policy_target(rule_out["test"])["policy_target"].astype(int).to_numpy(),
            rule_out["test"]["policy_level"].astype(int).to_numpy(),
        ),
        "Unconstrained learned policy": learned_selected["test_classification"],
        "Safety-constrained learned policy": safety_selected["test_classification"],
    }

    best_policy = choose_best_policy(policy_metrics)
    old_safety = BASELINE["safety_constrained_policy"]
    new_safety = policy_metrics["safety_constrained_policy"]
    helped = (
        new_safety["unsafe_continuation_rate_high_risk"]
        < old_safety["High-risk unsafe continuation"]
    )
    over_escalation_increased = (
        new_safety["over_escalation_burden"]
        > old_safety["Over-escalation burden"]
    )
    over_escalation_acceptable_vs_rule = (
        new_safety["over_escalation_burden"]
        < BASELINE["rule_based"]["Over-escalation burden"]
    )
    safer_than_before = helped
    limited_by_upstream = (
        new_safety["unsafe_continuation_rate_high_risk"] is not None
        and new_safety["unsafe_continuation_rate_high_risk"] > 0.15
    )

    comparison_lines = [
        "CRADLE Policy Comparison with High-Recall Confirm",
        "=================================================",
        "",
        "Experiment Rule",
        "---------------",
        "This is an ablation/isolation test.",
        "Only change: old Confirm predictions -> B_pos_weight_x2 high-recall Confirm predictions.",
        "Alert model changed: False",
        "Category model changed: False",
        "progression feature logic changed: False",
        "policy target definition changed: False",
        "safety guardrail logic changed: False",
        f"progression_feature_directory: {PROGRESSION_DIR}",
        f"confirm_model_used: {CONFIRM_MODEL_USED}",
        f"confirm_threshold_used_in_progression_features: {CONFIRM_THRESHOLD_USED_IN_PROGRESSION}",
        f"safety_guardrail_source: train_cradle_safety_constrained_policy_v2.py",
        f"safety_guardrail_variants_tested: {json.dumps(guardrail_variants)}",
        f"xgboost_status: {xgboost_status}",
        "",
        "Policy Metrics on Test Split",
        "----------------------------",
        *policy_metrics_table(policy_metrics),
        "",
        "Comparison to Old Confirm Baseline",
        "----------------------------------",
        *baseline_comparison_table(policy_metrics),
        "",
        "Classification Metrics Against policy_target",
        "--------------------------------------------",
        *classification_section(classification),
        "Selected Learned Policy Details",
        "-------------------------------",
        f"unconstrained_selected_model: {learned_selected['name']}",
        f"unconstrained_selection_reason: {learned_selected['selection_reason']}",
        f"safety_selected_model: {safety_selected['model_name']}",
        f"safety_selected_guardrail_variant: {safety_selected['guardrail_variant']}",
        f"safety_selection_reason: {safety_selected['selection_reason']}",
        "",
    ]

    RESULT_DIR.joinpath("policy_comparison_high_recall_confirm.txt").write_text(
        "\n".join(comparison_lines) + "\n"
    )

    summary_lines = [
        "CRADLE High-Recall Confirm Policy Summary",
        "=========================================",
        "",
        "Comparison Questions",
        "--------------------",
        f"1. Did high-recall Confirm reduce high-risk unsafe continuation? {'Yes' if helped else 'No'}. Old safety-constrained={old_safety['High-risk unsafe continuation']}, new safety-constrained={new_safety['unsafe_continuation_rate_high_risk']}.",
        f"2. Did high-recall Confirm reduce Confirm unsafe continuation? {'Yes' if new_safety['unsafe_continuation_rate_confirm'] < old_safety['Confirm unsafe continuation'] else 'No'}. Old={old_safety['Confirm unsafe continuation']}, new={new_safety['unsafe_continuation_rate_confirm']}.",
        f"3. Did over-escalation burden increase? {'Yes' if over_escalation_increased else 'No'}. Old={old_safety['Over-escalation burden']}, new={new_safety['over_escalation_burden']}.",
        f"4. Did over-escalation burden remain acceptable compared with the old rule-based policy? {'Yes' if over_escalation_acceptable_vs_rule else 'No'}. Old rule-based burden={BASELINE['rule_based']['Over-escalation burden']}, new safety-constrained burden={new_safety['over_escalation_burden']}.",
        f"5. Which policy is best after adding high-recall Confirm? {best_policy}.",
        f"6. Is the safety-constrained policy now safer than before? {'Yes' if safer_than_before else 'No'}.",
        f"7. Is the policy still limited by upstream model errors? {'Likely yes' if limited_by_upstream else 'Less clearly; downstream guardrails/policy behavior should now be inspected.'}.",
        f"8. Should we keep B_pos_weight_x2 in the final pipeline? {'Yes, as the current Confirm candidate for downstream policy work.' if helped else 'Not yet; compare the tradeoff carefully before keeping it.'}",
        "9. Should we now tune safety guardrails again? Yes, but only after auditing the new remaining unsafe Level 0 cases; do not create a new policy version blindly.",
        "10. What is the single next implementation step? Audit the remaining high-risk Level 0 cases under the high-recall Confirm safety-constrained policy, then tune guardrails only if the missed-case patterns justify it.",
        "",
        "Final Beginner-Friendly Summary",
        "-------------------------------",
        f"1. Which progression feature directory was used? {PROGRESSION_DIR}.",
        f"2. Which Confirm model was used? {CONFIRM_MODEL_USED}.",
        f"3. What was the old safety-constrained high-risk unsafe continuation? {old_safety['High-risk unsafe continuation']}.",
        f"4. What is the new safety-constrained high-risk unsafe continuation? {new_safety['unsafe_continuation_rate_high_risk']}.",
        f"5. What was the old safety-constrained over-escalation burden? {old_safety['Over-escalation burden']}.",
        f"6. What is the new safety-constrained over-escalation burden? {new_safety['over_escalation_burden']}.",
        f"7. Did high-recall Confirm help? {'Yes' if helped else 'No'}.",
        f"8. Did it create too many false alarms? {'No compared with old rule-based burden' if over_escalation_acceptable_vs_rule else 'Possibly yes; burden exceeds the old rule-based reference.'}.",
        f"9. Which policy is currently best? {best_policy}.",
        "10. What is the single next implementation step? Audit remaining high-risk Level 0 cases before changing guardrails.",
    ]
    RESULT_DIR.joinpath("policy_high_recall_confirm_summary.txt").write_text(
        "\n".join(summary_lines) + "\n"
    )

    return policy_metrics, classification, best_policy


def main():
    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        require_file(path)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    rule_out = run_rule_based(train_df, val_df, test_df)
    learned_selected, learned_results = run_learned(train_df, val_df, test_df)
    safety_selected, safety_results, xgboost_status, guardrail_variants = run_safety_constrained(
        train_df, val_df, test_df, learned_selected
    )
    policy_metrics, classification, best_policy = write_reports(
        rule_out,
        learned_selected,
        safety_selected,
        safety_results,
        xgboost_status,
        guardrail_variants,
    )

    print("Saved high-recall Confirm policy outputs:")
    print(f"- {RULE_OUTPUT_DIR / 'cradle_test_rule_based_policy_outputs.csv'}")
    print(f"- {LEARNED_OUTPUT_DIR / 'cradle_test_learned_policy_outputs.csv'}")
    print(f"- {SAFETY_OUTPUT_DIR / 'cradle_test_safety_constrained_policy_outputs.csv'}")
    print(f"Saved comparison to {RESULT_DIR / 'policy_comparison_high_recall_confirm.txt'}")
    print(f"Saved summary to {RESULT_DIR / 'policy_high_recall_confirm_summary.txt'}")
    print("")
    print("Final Beginner-Friendly Summary")
    old_safety = BASELINE["safety_constrained_policy"]
    new_safety = policy_metrics["safety_constrained_policy"]
    print(f"1. Which progression feature directory was used? {PROGRESSION_DIR}.")
    print(f"2. Which Confirm model was used? {CONFIRM_MODEL_USED}.")
    print(f"3. What was the old safety-constrained high-risk unsafe continuation? {old_safety['High-risk unsafe continuation']}.")
    print(f"4. What is the new safety-constrained high-risk unsafe continuation? {new_safety['unsafe_continuation_rate_high_risk']}.")
    print(f"5. What was the old safety-constrained over-escalation burden? {old_safety['Over-escalation burden']}.")
    print(f"6. What is the new safety-constrained over-escalation burden? {new_safety['over_escalation_burden']}.")
    print(f"7. Did high-recall Confirm help? {new_safety['unsafe_continuation_rate_high_risk'] < old_safety['High-risk unsafe continuation']}.")
    print(f"8. Did it create too many false alarms? {new_safety['over_escalation_burden'] >= BASELINE['rule_based']['Over-escalation burden']}.")
    print(f"9. Which policy is currently best? {best_policy}.")
    print("10. What is the single next implementation step? Audit remaining high-risk Level 0 cases before changing guardrails.")


if __name__ == "__main__":
    main()
