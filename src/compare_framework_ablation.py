import argparse
import os

import pandas as pd


REQUIRED_COLUMNS = [
    "conversation_id",
    "turn_id",
    "predicted_label",
    "mean_high_prob",
    "epistemic_uncertainty",
    "aleatoric_uncertainty",
    "confidence",
    "baseline_deviation",
    "trajectory_slope_so_far",
    "rising_trend_flag",
    "change_point_flag",
    "escalation_level",
]


parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", default="outputs/annomi_policy_outputs.csv")
parser.add_argument("--output_txt", default="results/ablation_comparison_annomi.txt")
parser.add_argument("--output_csv", default="results/ablation_comparison_annomi.csv")
args = parser.parse_args()


def validate_input(df):
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required ablation columns: {missing_columns}"
        )


def percent(numerator, denominator):
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def single_turn_risk_policy(df):
    levels = pd.Series(0, index=df.index)
    levels[df["predicted_label"] == 1] = 2
    levels[df["predicted_label"] == 2] = 4
    return levels


def risk_uncertainty_policy(df):
    levels = single_turn_risk_policy(df)
    uncertain = (
        (df["epistemic_uncertainty"] >= df["epistemic_uncertainty"].quantile(0.90))
        | (df["aleatoric_uncertainty"] >= df["aleatoric_uncertainty"].quantile(0.90))
        | (df["confidence"] <= df["confidence"].quantile(0.10))
    )
    medium_or_high = df["predicted_label"].isin([1, 2]) | (df["mean_high_prob"] >= 0.35)
    levels[uncertain & medium_or_high & (levels < 4)] = 3
    return levels


def risk_temporal_policy(df):
    levels = single_turn_risk_policy(df)
    temporal_risk = (
        (df["rising_trend_flag"] == 1)
        | (df["change_point_flag"] == 1)
        | (df["baseline_deviation"] >= 0.20)
        | (df["trajectory_slope_so_far"] >= 0.03)
    )
    medium_or_high = df["predicted_label"].isin([1, 2]) | (df["mean_high_prob"] >= 0.35)
    levels[temporal_risk & medium_or_high & (levels < 4)] = 3
    return levels


def full_framework_policy(df):
    return df["escalation_level"].astype(int)


def compute_metrics(df, system_name, escalation_levels):
    labels = df["predicted_label"].astype(int)
    escalation = escalation_levels.astype(int)

    high_risk_turns = labels == 2
    low_risk_turns = labels == 0
    unsafe_continuations = high_risk_turns & escalation.isin([0, 1])
    over_escalations = low_risk_turns & escalation.isin([3, 4])
    autonomous_turns = ~escalation.isin([3, 4])

    eval_df = df[["conversation_id", "turn_id"]].copy()
    eval_df["escalation_level"] = escalation
    time_to_escalation = (
        eval_df[eval_df["escalation_level"] >= 3]
        .sort_values(["conversation_id", "turn_id"])
        .groupby("conversation_id")["turn_id"]
        .first()
    )

    level_counts = escalation.value_counts().sort_index()

    return {
        "system": system_name,
        "unsafe_continuation_rate": percent(unsafe_continuations.sum(), high_risk_turns.sum()),
        "unsafe_continuation_count": int(unsafe_continuations.sum()),
        "high_risk_turn_count": int(high_risk_turns.sum()),
        "over_escalation_burden": percent(over_escalations.sum(), low_risk_turns.sum()),
        "over_escalation_count": int(over_escalations.sum()),
        "low_risk_turn_count": int(low_risk_turns.sum()),
        "autonomy_coverage_rate": percent(autonomous_turns.sum(), len(df)),
        "autonomous_turn_count": int(autonomous_turns.sum()),
        "total_turns": len(df),
        "conversations_with_escalation_ge_3": len(time_to_escalation),
        "total_conversations": df["conversation_id"].nunique(),
        "average_time_to_escalation": time_to_escalation.mean() if len(time_to_escalation) else -1,
        "median_time_to_escalation": time_to_escalation.median() if len(time_to_escalation) else -1,
        "level_0_count": int(level_counts.get(0, 0)),
        "level_1_count": int(level_counts.get(1, 0)),
        "level_2_count": int(level_counts.get(2, 0)),
        "level_3_count": int(level_counts.get(3, 0)),
        "level_4_count": int(level_counts.get(4, 0)),
        "human_review_cases_level_3": int(level_counts.get(3, 0)),
        "crisis_protocol_cases_level_4": int(level_counts.get(4, 0)),
    }


def build_report(summary_df):
    lines = []
    lines.append("Ablation / Baseline Comparison Report")
    lines.append("=====================================")
    lines.append("")
    lines.append(
        "NOTE: AnnoMI does not provide true clinical risk labels in this pipeline. "
        "These are pseudo-label safety analyses using Module 1 predicted labels, "
        "not validated clinical safety results."
    )
    lines.append("")
    lines.append("Systems compared:")
    lines.append("1. Single-turn risk only: risk severity only, no uncertainty or temporal modeling.")
    lines.append("2. Risk + uncertainty: adds epistemic/aleatoric uncertainty and confidence.")
    lines.append("3. Risk + temporal: adds baseline deviation, slope, rising trend, and change points.")
    lines.append("4. Full framework: uses existing graded policy with risk, uncertainty, and temporal progression.")
    lines.append("")

    for _, row in summary_df.iterrows():
        lines.append(row["system"])
        lines.append("-" * len(row["system"]))
        lines.append(f"Unsafe continuation rate: {row['unsafe_continuation_rate']:.2f}%")
        lines.append(f"Over-escalation burden: {row['over_escalation_burden']:.2f}%")
        lines.append(f"Autonomy / coverage rate: {row['autonomy_coverage_rate']:.2f}%")
        lines.append(
            "Escalation distribution: "
            f"L0={int(row['level_0_count'])}, "
            f"L1={int(row['level_1_count'])}, "
            f"L2={int(row['level_2_count'])}, "
            f"L3={int(row['level_3_count'])}, "
            f"L4={int(row['level_4_count'])}"
        )
        lines.append(f"Human review cases, Level 3: {int(row['human_review_cases_level_3'])}")
        lines.append(f"Crisis protocol cases, Level 4: {int(row['crisis_protocol_cases_level_4'])}")
        lines.append(
            "Time-to-escalation: "
            f"{int(row['conversations_with_escalation_ge_3'])}/"
            f"{int(row['total_conversations'])} conversations, "
            f"mean turn_id={row['average_time_to_escalation']:.2f}, "
            f"median turn_id={row['median_time_to_escalation']:.2f}"
        )
        lines.append("")

    return "\n".join(lines)


def main():
    df = pd.read_csv(args.input_csv)
    validate_input(df)

    systems = {
        "Single-turn risk only baseline": single_turn_risk_policy(df),
        "Risk + uncertainty baseline": risk_uncertainty_policy(df),
        "Risk + temporal baseline": risk_temporal_policy(df),
        "Full framework": full_framework_policy(df),
    }

    summary_rows = [
        compute_metrics(df, system_name, escalation_levels)
        for system_name, escalation_levels in systems.items()
    ]
    summary_df = pd.DataFrame(summary_rows)

    output_csv_dir = os.path.dirname(args.output_csv)
    if output_csv_dir:
        os.makedirs(output_csv_dir, exist_ok=True)
    summary_df.to_csv(args.output_csv, index=False)

    report = build_report(summary_df)
    output_txt_dir = os.path.dirname(args.output_txt)
    if output_txt_dir:
        os.makedirs(output_txt_dir, exist_ok=True)
    with open(args.output_txt, "w") as output_file:
        output_file.write(report)
        output_file.write("\n")

    print(report)
    print(f"Saved text report to: {args.output_txt}")
    print(f"Saved CSV summary to: {args.output_csv}")


if __name__ == "__main__":
    main()
