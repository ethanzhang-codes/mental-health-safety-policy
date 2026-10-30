import json
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ["train", "validation", "test"]
CATEGORY_NAMES = ["CA", "DV", "RA", "SHA", "SH", "SI_active", "SI_passive"]
CATEGORY_PROB_COLS = [f"category_prob_{name}" for name in CATEGORY_NAMES]

FEATURE_DIR = Path("cradle_dialogue/data/progression_features_high_recall_confirm")
POLICY_DIR = Path("cradle_dialogue/policy_outputs_high_recall_confirm/safety_constrained_policy")
OUTPUT_FEATURE_DIR = Path("cradle_dialogue/data/abstract_aligned_features")
RESULT_DIR = Path("cradle_dialogue/results/abstract_alignment")


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def binary_entropy(prob):
    p = np.clip(prob.astype(float), 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def uncertainty_level_from_margin(margin):
    return np.select(
        [margin < 0.10, margin < 0.25],
        ["high", "medium"],
        default="low",
    )


def add_uncertainty_features(df: pd.DataFrame):
    out = df.copy()
    out["alert_entropy"] = binary_entropy(out["alert_prob"])
    out["alert_confidence"] = np.maximum(out["alert_prob"], 1 - out["alert_prob"])
    out["alert_margin"] = (out["alert_prob"] - 0.5).abs()
    out["alert_uncertainty_level"] = uncertainty_level_from_margin(out["alert_margin"])

    out["confirm_entropy"] = binary_entropy(out["confirm_prob"])
    out["confirm_confidence"] = np.maximum(out["confirm_prob"], 1 - out["confirm_prob"])
    out["confirm_margin"] = (out["confirm_prob"] - 0.5).abs()
    out["confirm_uncertainty_level"] = uncertainty_level_from_margin(out["confirm_margin"])

    category_probs = out[CATEGORY_PROB_COLS].astype(float)
    category_entropy = category_probs.apply(binary_entropy)
    category_confidence = np.maximum(category_probs, 1 - category_probs)
    sorted_category_probs = np.sort(category_probs.to_numpy(), axis=1)
    top = sorted_category_probs[:, -1]
    second = sorted_category_probs[:, -2]

    out["category_mean_entropy"] = category_entropy.mean(axis=1)
    out["category_max_entropy"] = category_entropy.max(axis=1)
    out["category_mean_confidence"] = category_confidence.mean(axis=1)
    out["category_min_margin"] = (category_probs - 0.5).abs().min(axis=1)
    out["category_uncertain_count"] = ((category_probs >= 0.30) & (category_probs <= 0.70)).sum(axis=1)
    out["top_category_margin"] = top - second
    out["category_uncertainty_level"] = np.select(
        [
            (out["category_uncertain_count"] >= 2) | (out["top_category_margin"] < 0.10),
            (out["category_uncertain_count"] == 1) | (out["top_category_margin"] < 0.25),
        ],
        ["high", "medium"],
        default="low",
    )

    out = out.sort_values(["dialogue_id", "turn_id"]).copy()
    grouped = out.groupby("dialogue_id", sort=False)
    out["prev_alert_entropy"] = grouped["alert_entropy"].shift(1)
    out["alert_entropy_change_from_prev"] = out["alert_entropy"] - out["prev_alert_entropy"]
    out["alert_entropy_rolling_mean_3"] = grouped["alert_entropy"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    out["alert_entropy_cumulative_max"] = grouped["alert_entropy"].cummax()
    out["prev_confirm_entropy"] = grouped["confirm_entropy"].shift(1)
    out["confirm_entropy_change_from_prev"] = out["confirm_entropy"] - out["prev_confirm_entropy"]
    out["confirm_entropy_rolling_mean_3"] = grouped["confirm_entropy"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    out["confirm_entropy_cumulative_max"] = grouped["confirm_entropy"].cummax()
    out["category_max_entropy_rolling_mean_3"] = grouped["category_max_entropy"].transform(
        lambda s: s.rolling(window=3, min_periods=1).mean()
    )
    out["uncertainty_accumulation_score"] = out[
        ["alert_entropy_cumulative_max", "confirm_entropy_cumulative_max", "category_max_entropy"]
    ].max(axis=1)
    out["uncertainty_rising_flag"] = (
        (out["alert_entropy_change_from_prev"] > 0.10)
        | (out["confirm_entropy_change_from_prev"] > 0.10)
        | (out["category_max_entropy"] > out["category_max_entropy_rolling_mean_3"] + 0.10)
    ).astype(int)
    return out


def source_flags(row, specs):
    flags = [name for name, value in specs if bool(value)]
    return ";".join(flags)


def add_severity_features(df: pd.DataFrame):
    out = df.copy()
    out["severity_score"] = pd.concat(
        [
            out["confirm_prob"],
            out["alert_prob"] * 0.70,
            out["max_category_prob"],
            out["risk_accumulation_score"] * 0.60,
            out["risk_escalation_score"],
        ],
        axis=1,
    ).max(axis=1)

    self_harm_category = (
        (out["category_pred_tuned_SH"] == 1)
        | (out["category_pred_tuned_SHA"] == 1)
        | (out["category_pred_tuned_SI_active"] == 1)
        | (out["category_pred_tuned_SI_passive"] == 1)
    )
    high = (
        (out["confirm_pred_tuned"] == 1)
        | (out["confirm_prob"] >= 0.011)
        | self_harm_category
        | (out["severity_score"] >= 0.75)
    )
    moderate = (
        (out["alert_pred_tuned"] == 1)
        | (out["severity_score"] >= 0.40)
        | (out["risk_accumulation_score"] >= 0.60)
        | (out["risk_volatility_score"] >= 0.20)
    )
    out["severity_level"] = np.select([high, moderate], ["high", "moderate"], default="low")

    out["severity_source_flags"] = [
        source_flags(
            row,
            [
                ("confirm_pred_tuned", row["confirm_pred_tuned"] == 1),
                ("confirm_prob_ge_0.011", row["confirm_prob"] >= 0.011),
                (
                    "self_harm_category",
                    any(
                        row[f"category_pred_tuned_{name}"] == 1
                        for name in ["SH", "SHA", "SI_active", "SI_passive"]
                    ),
                ),
                ("severity_score_ge_0.75", row["severity_score"] >= 0.75),
                ("alert_pred_tuned", row["alert_pred_tuned"] == 1),
                ("severity_score_ge_0.40", row["severity_score"] >= 0.40),
                ("risk_accumulation_ge_0.60", row["risk_accumulation_score"] >= 0.60),
                ("risk_volatility_ge_0.20", row["risk_volatility_score"] >= 0.20),
            ],
        )
        for _, row in out.iterrows()
    ]
    out.loc[out["severity_source_flags"].eq(""), "severity_source_flags"] = "none"
    return out


def add_progression_status(df: pd.DataFrame):
    out = df.copy()
    escalating = (
        (out["risk_escalation_score"] >= 0.20)
        | (out["confirm_rising_flag"] == 1)
        | (out["alert_rising_flag"] == 1)
    )
    accumulated = out["risk_accumulation_score"] >= 0.60
    volatile = out["risk_volatility_score"] >= 0.20
    out["progression_status"] = np.select(
        [escalating, accumulated, volatile],
        ["escalating", "accumulated", "volatile"],
        default="stable_low",
    )
    out["progression_source_flags"] = [
        source_flags(
            row,
            [
                ("risk_escalation_ge_0.20", row["risk_escalation_score"] >= 0.20),
                ("confirm_rising_flag", row["confirm_rising_flag"] == 1),
                ("alert_rising_flag", row["alert_rising_flag"] == 1),
                ("risk_accumulation_ge_0.60", row["risk_accumulation_score"] >= 0.60),
                ("risk_volatility_ge_0.20", row["risk_volatility_score"] >= 0.20),
            ],
        )
        for _, row in out.iterrows()
    ]
    out.loc[out["progression_source_flags"].eq(""), "progression_source_flags"] = "none"
    return out


def add_overall_uncertainty(df: pd.DataFrame):
    out = df.copy()
    levels = out[["alert_uncertainty_level", "confirm_uncertainty_level", "category_uncertainty_level"]]
    out["overall_uncertainty_level"] = np.select(
        [(levels == "high").any(axis=1), (levels == "medium").any(axis=1)],
        ["high", "medium"],
        default="low",
    )
    out["uncertainty_source_flags"] = [
        source_flags(
            row,
            [
                (f"alert_uncertainty_{row['alert_uncertainty_level']}", row["alert_uncertainty_level"] != "low"),
                (
                    f"confirm_uncertainty_{row['confirm_uncertainty_level']}",
                    row["confirm_uncertainty_level"] != "low",
                ),
                (
                    f"category_uncertainty_{row['category_uncertainty_level']}",
                    row["category_uncertainty_level"] != "low",
                ),
                ("uncertainty_rising_flag", row["uncertainty_rising_flag"] == 1),
            ],
        )
        for _, row in out.iterrows()
    ]
    out.loc[out["uncertainty_source_flags"].eq(""), "uncertainty_source_flags"] = "none"
    return out


def merge_policy_outputs(df: pd.DataFrame, split: str):
    policy_path = POLICY_DIR / f"cradle_{split}_safety_constrained_policy_outputs.csv"
    if not policy_path.exists():
        return df, {"policy_file_exists": False, "rows_missing_policy": len(df), "duplicate_policy_keys": 0}
    policy_cols = ["dialogue_id", "turn_id", "policy_level", "policy_action", "policy_rationale"]
    policy = pd.read_csv(policy_path, usecols=policy_cols)
    duplicate_keys = int(policy.duplicated(["dialogue_id", "turn_id"]).sum())
    merged = df.merge(policy, on=["dialogue_id", "turn_id"], how="left")
    missing = int(merged["policy_level"].isna().sum())
    return merged, {
        "policy_file_exists": True,
        "rows_missing_policy": missing,
        "duplicate_policy_keys": duplicate_keys,
    }


def add_rationale(df: pd.DataFrame):
    out = df.copy()
    if "policy_action" not in out.columns:
        out["policy_action"] = "unavailable"
    out["abstract_aligned_rationale"] = [
        (
            "Decision context: "
            f"severity={row.severity_level} due to {row.severity_source_flags}; "
            f"top_category={row.top_category}; "
            f"progression={row.progression_status}; "
            f"uncertainty={row.overall_uncertainty_level}; "
            f"policy_action={row.policy_action}."
        )
        for row in out.itertuples(index=False)
    ]
    return out


def build_split(split: str):
    feature_path = FEATURE_DIR / f"cradle_{split}_progression_features.csv"
    require_file(feature_path)
    df = pd.read_csv(feature_path)
    out = add_uncertainty_features(df)
    out = add_severity_features(out)
    out = add_progression_status(out)
    out = add_overall_uncertainty(out)
    out, merge_info = merge_policy_outputs(out, split)
    out = add_rationale(out)
    output_path = OUTPUT_FEATURE_DIR / f"cradle_{split}_abstract_aligned_features.csv"
    out.to_csv(output_path, index=False)
    merge_info.update({"rows": len(out), "columns": len(out.columns), "output_path": str(output_path)})
    return out, merge_info


def markdown_table(rows, headers):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return lines


def write_schema():
    fields = [
        ("alert_entropy", "Binary entropy of Alert probability."),
        ("alert_confidence", "Maximum of alert_prob and 1 - alert_prob."),
        ("alert_margin", "Distance from the binary decision boundary at 0.5."),
        ("alert_uncertainty_level", "high/medium/low uncertainty derived from alert_margin."),
        ("confirm_entropy", "Binary entropy of Confirm probability."),
        ("confirm_confidence", "Maximum of confirm_prob and 1 - confirm_prob."),
        ("confirm_margin", "Distance from the binary decision boundary at 0.5."),
        ("confirm_uncertainty_level", "high/medium/low uncertainty derived from confirm_margin."),
        ("category_mean_entropy", "Mean binary entropy across category probabilities."),
        ("category_max_entropy", "Maximum category entropy."),
        ("category_mean_confidence", "Mean max(p, 1-p) confidence across categories."),
        ("category_min_margin", "Smallest category distance from 0.5."),
        ("category_uncertain_count", "Number of category probabilities in [0.30, 0.70]."),
        ("top_category_margin", "Highest category probability minus second highest."),
        ("category_uncertainty_level", "high/medium/low category uncertainty."),
        ("uncertainty_accumulation_score", "Max cumulative uncertainty signal across Alert, Confirm, and Category."),
        ("uncertainty_rising_flag", "Whether uncertainty increased sharply relative to recent context."),
        ("severity_score", "Operational severity proxy from Confirm, Alert, Category, accumulation, and escalation scores."),
        ("severity_level", "low/moderate/high operational severity proxy."),
        ("severity_source_flags", "Signals that determined or supported the severity level."),
        ("progression_status", "stable_low, volatile, accumulated, or escalating temporal risk status."),
        ("progression_source_flags", "Signals that determined or supported progression status."),
        ("overall_uncertainty_level", "Highest uncertainty level across Alert, Confirm, and Category."),
        ("uncertainty_source_flags", "Uncertainty dimensions contributing non-low uncertainty or rising uncertainty."),
        ("abstract_aligned_rationale", "Human-readable multidimensional rationale combining severity, category, progression, uncertainty, and policy action."),
    ]
    lines = [
        "# Multi-Dimensional Risk Schema",
        "",
        "These fields augment the frozen high-recall Confirm progression features. They do not replace the trained models or selected policy.",
        "",
    ]
    lines.extend(markdown_table([{"field": f, "description": d} for f, d in fields], ["field", "description"]))
    (RESULT_DIR / "multidimensional_risk_schema.md").write_text("\n".join(lines) + "\n")


def write_claim_mapping():
    rows = [
        {
            "abstract_claim": "single-turn classification limitation",
            "implementation_component": "Context error analysis and temporal Alert comparison",
            "status": "supported with limitation",
            "evidence_or_output_file": "cradle_dialogue/results/context_error_analysis/context_error_summary.md",
            "limitation": "No context-aware Confirm model has been trained yet.",
        },
        {
            "abstract_claim": "temporal risk modeling",
            "implementation_component": "Temporal Alert model and progression features",
            "status": "implemented",
            "evidence_or_output_file": "cradle_dialogue/results/temporal_alert_embeddings_history_window_ablation.txt",
            "limitation": "Temporal Confirm remains future work.",
        },
        {
            "abstract_claim": "uncertainty estimation",
            "implementation_component": "Entropy, confidence, margins, and temporal uncertainty features",
            "status": "implemented as probability-derived estimates",
            "evidence_or_output_file": "cradle_dialogue/results/abstract_alignment/uncertainty_feature_audit.md",
            "limitation": "Uncertainty is post hoc; no Bayesian or ensemble model was trained.",
        },
        {
            "abstract_claim": "multi-dimensional risk assessment: severity",
            "implementation_component": "severity_score and severity_level",
            "status": "implemented as operational proxy",
            "evidence_or_output_file": "cradle_dialogue/results/abstract_alignment/severity_proxy_audit.md",
            "limitation": "CRADLE does not provide ordinal severity labels.",
        },
        {
            "abstract_claim": "multi-dimensional risk assessment: category",
            "implementation_component": "single-turn Category model probabilities and top_category",
            "status": "implemented",
            "evidence_or_output_file": "cradle_dialogue/results/single_turn_category/category_model_summary.txt",
            "limitation": "Low-support categories remain harder.",
        },
        {
            "abstract_claim": "multi-dimensional risk assessment: progression",
            "implementation_component": "risk_accumulation_score, risk_escalation_score, risk_volatility_score, progression_status",
            "status": "implemented",
            "evidence_or_output_file": "cradle_dialogue/data/abstract_aligned_features/",
            "limitation": "Progression is feature-derived rather than a separately validated clinical state.",
        },
        {
            "abstract_claim": "graded escalation",
            "implementation_component": "Policy levels 0-3 and policy actions",
            "status": "implemented",
            "evidence_or_output_file": "cradle_dialogue/results/final_experiment_summary/final_policy_decision.md",
            "limitation": "Policy is an experimental safety layer, not a deployed chatbot.",
        },
        {
            "abstract_claim": "interpretable rationales",
            "implementation_component": "policy_rationale and abstract_aligned_rationale",
            "status": "implemented",
            "evidence_or_output_file": "cradle_dialogue/data/abstract_aligned_features/",
            "limitation": "Rationales are feature/rule summaries, not causal explanations of neural internals.",
        },
        {
            "abstract_claim": "predictive metrics",
            "implementation_component": "AUROC, AUPRC, F1, precision, recall reports",
            "status": "implemented",
            "evidence_or_output_file": "cradle_dialogue/results/final_experiment_summary/final_results_table.csv",
            "limitation": "Metrics are internal to current CRADLE splits.",
        },
        {
            "abstract_claim": "safety-centered metrics",
            "implementation_component": "unsafe continuation, over-escalation, defer precision, urgent precision",
            "status": "implemented",
            "evidence_or_output_file": "cradle_dialogue/results/final_experiment_summary/final_policy_decision.md",
            "limitation": "Requires external validation before deployment claims.",
        },
        {
            "abstract_claim": "selective/responsible autonomy",
            "implementation_component": "Safety-constrained policy that escalates only when risk signals justify it",
            "status": "implemented experimentally",
            "evidence_or_output_file": "cradle_dialogue/results/policy_high_recall_confirm/policy_high_recall_confirm_summary.txt",
            "limitation": "Human factors and real deployment impact were not tested.",
        },
    ]
    pd.DataFrame(rows).to_csv(RESULT_DIR / "final_abstract_claim_mapping.csv", index=False)


def write_uncertainty_audit(frames, merge_infos):
    uncertainty_cols = [
        "alert_entropy",
        "alert_confidence",
        "alert_margin",
        "alert_uncertainty_level",
        "confirm_entropy",
        "confirm_confidence",
        "confirm_margin",
        "confirm_uncertainty_level",
        "category_mean_entropy",
        "category_max_entropy",
        "category_mean_confidence",
        "category_min_margin",
        "category_uncertain_count",
        "top_category_margin",
        "category_uncertainty_level",
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
        "overall_uncertainty_level",
        "uncertainty_source_flags",
    ]
    lines = ["# Uncertainty Feature Audit", "", "## Columns Added", ""]
    lines.extend(f"- `{col}`" for col in uncertainty_cols)
    lines.extend(["", "## Missing Values by Split", ""])
    for split, df in frames.items():
        missing = df[uncertainty_cols].isna().sum()
        missing = missing[missing > 0].to_dict()
        lines.append(f"### {split}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(missing, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    for col in [
        "alert_uncertainty_level",
        "confirm_uncertainty_level",
        "category_uncertainty_level",
        "overall_uncertainty_level",
    ]:
        lines.extend([f"## Distribution of `{col}`", ""])
        for split, df in frames.items():
            lines.append(f"### {split}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(df[col].value_counts(dropna=False).to_dict(), indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")
    lines.extend(["## Merge Issues", ""])
    lines.append("```json")
    lines.append(json.dumps(merge_infos, indent=2, sort_keys=True))
    lines.append("```")
    (RESULT_DIR / "uncertainty_feature_audit.md").write_text("\n".join(lines) + "\n")


def write_severity_audit(frames):
    lines = [
        "# Severity Proxy Audit",
        "",
        "The severity proxy is an operational feature derived from existing model probabilities and temporal risk scores. It is not a gold clinical severity label.",
        "",
    ]
    for split, df in frames.items():
        lines.extend([f"## {split}", "", "### severity_score summary", "", "```json"])
        lines.append(json.dumps(df["severity_score"].describe().to_dict(), indent=2, sort_keys=True))
        lines.extend(["```", "", "### severity_level distribution", "", "```json"])
        lines.append(json.dumps(df["severity_level"].value_counts(dropna=False).to_dict(), indent=2, sort_keys=True))
        lines.extend(["```", "", "### high-risk target distribution by severity_level", ""])
        if "target_confirm" in df.columns:
            high_risk = (
                (df["target_confirm"] == 1)
                | (df["target_category_SI_active"] == 1)
                | (df["target_category_SH"] == 1)
                | (df["target_category_SHA"] == 1)
            ).astype(int)
            table = pd.crosstab(df["severity_level"], high_risk, margins=True).rename(columns={0: "not_high_risk", 1: "high_risk"})
            lines.extend(["```", table.to_string(), "```", ""])
        if "policy_level" in df.columns:
            lines.extend(["### policy_level distribution by severity_level", "", "```"])
            lines.append(pd.crosstab(df["severity_level"], df["policy_level"], margins=True).to_string())
            lines.extend(["```", ""])
    lines.extend(
        [
            "## Limitations",
            "",
            "- CRADLE does not provide ordinal severity labels, so severity is represented as a proxy.",
            "- The proxy uses existing probabilities and temporal risk scores; errors in upstream models can propagate into severity_level.",
            "- Severity should be reported as an operational safety feature, not a clinically validated severity scale.",
        ]
    )
    (RESULT_DIR / "severity_proxy_audit.md").write_text("\n".join(lines) + "\n")


def write_alignment_report(merge_infos):
    lines = [
        "# Abstract Alignment Report",
        "",
        "## 1. Overview",
        "",
        "This augmentation layer aligns the frozen CRADLE implementation with the fixed accepted abstract by adding explicit uncertainty features, an operational severity proxy, a multi-dimensional risk assessment view, and dimension-aware rationales.",
        "",
        "No Alert, Confirm, or Category model was retrained. Existing progression features were not rebuilt from scratch. Frozen policy outputs were not overwritten.",
        "",
        "## 2. Temporal Risk Modeling",
        "",
        "Temporal risk modeling is represented by the temporal Alert model and the progression feature set, including risk accumulation, escalation, volatility, rolling probabilities, cumulative maxima, and rising flags.",
        "",
        "## 3. Uncertainty Estimation",
        "",
        "Uncertainty is implemented through probability-derived entropy, confidence, margins, categorical uncertainty counts, top-category margins, and temporal uncertainty features. This is post hoc uncertainty estimation, not Bayesian or ensemble uncertainty.",
        "",
        "## 4. Multi-Dimensional Risk Assessment",
        "",
        "The abstract-aligned feature files expose severity, category, progression, uncertainty, and policy action together. Severity is an operational proxy because CRADLE does not provide ordinal low/medium/high severity labels.",
        "",
        "## 5. Graded Policy-Based Escalation",
        "",
        "The selected safety-constrained policy preserves graded levels 0-3, mapping to continue conversation, supportive resources, human support deferral, and urgent escalation.",
        "",
        "## 6. Interpretable Rationales",
        "",
        "Each abstract-aligned row includes an `abstract_aligned_rationale` that summarizes severity level, severity drivers, top category, progression status, uncertainty level, and policy action.",
        "",
        "## 7. Safety-Centered Evaluation",
        "",
        "The final policy documentation reports unsafe continuation, over-escalation burden, urgent escalation precision, and human defer precision alongside conventional predictive metrics.",
        "",
        "## 8. Remaining Limitations",
        "",
        "- The system is a safety escalation layer, not a deployed chatbot.",
        "- Uncertainty is estimated from existing probabilities rather than trained as a calibrated uncertainty model.",
        "- Severity is an operational proxy rather than a gold ordinal severity label.",
        "- Remaining unsafe continuations suggest upstream representation and calibration limitations.",
        "",
        "## Output Status",
        "",
        "```json",
        json.dumps(merge_infos, indent=2, sort_keys=True),
        "```",
    ]
    (RESULT_DIR / "abstract_alignment_report.md").write_text("\n".join(lines) + "\n")


def main():
    OUTPUT_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {}
    merge_infos = {}
    for split in SPLITS:
        frame, merge_info = build_split(split)
        frames[split] = frame
        merge_infos[split] = merge_info

    write_schema()
    write_claim_mapping()
    write_uncertainty_audit(frames, merge_infos)
    write_severity_audit(frames)
    write_alignment_report(merge_infos)

    total_missing_policy = sum(info["rows_missing_policy"] for info in merge_infos.values())
    duplicate_policy_keys = sum(info["duplicate_policy_keys"] for info in merge_infos.values())
    missing_values = {
        split: int(frame[["severity_score", "severity_level", "overall_uncertainty_level", "abstract_aligned_rationale"]].isna().sum().sum())
        for split, frame in frames.items()
    }

    print("1. Abstract-aligned feature files created.")
    print("2. Uncertainty features added.")
    print("3. Severity proxy added.")
    print("4. Multi-dimensional risk view created.")
    print("5. Abstract alignment report created.")
    if total_missing_policy == 0 and duplicate_policy_keys == 0 and sum(missing_values.values()) == 0:
        print("6. No missing values in key added fields and no policy merge issues detected.")
    else:
        print(
            "6. Merge/audit issues detected: "
            f"rows_missing_policy={total_missing_policy}, duplicate_policy_keys={duplicate_policy_keys}, "
            f"key_added_field_missing_values={missing_values}."
        )
    print("7. Single next step: cite the abstract-alignment report when drafting the Results and Discussion sections.")


if __name__ == "__main__":
    main()
