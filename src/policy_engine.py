import argparse
import os

import pandas as pd


REQUIRED_COLUMNS = [
    "mean_low_prob",
    "mean_medium_prob",
    "mean_high_prob",
    "confidence",
    "epistemic_uncertainty",
    "aleatoric_uncertainty",
    "baseline_deviation",
    "trajectory_slope_so_far",
    "rising_trend_flag",
    "high_uncertainty_flag",
    "change_point_flag",
]

LEVEL_ACTIONS = {
    0: "normal chatbot response",
    1: "supportive response + light safety check",
    2: "offer coping resources / encourage help-seeking",
    3: "pause and require human review",
    4: "crisis protocol / emergency instructions",
}


parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_csv", required=True)
args = parser.parse_args()


def validate_input(df):
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required policy columns: {missing_columns}\n"
            f"Required columns are: {REQUIRED_COLUMNS}"
        )


def active_reasons(row):
    reasons = []

    if row["mean_high_prob"] >= 0.90:
        reasons.append("very high predicted high-risk probability")
    elif row["mean_high_prob"] >= 0.60:
        reasons.append("elevated high-risk probability")

    if row["mean_medium_prob"] >= row["mean_low_prob"] and row["mean_medium_prob"] >= row["mean_high_prob"]:
        reasons.append("medium risk is the dominant prediction")

    if row["high_uncertainty_flag"] == 1:
        reasons.append("high uncertainty")

    if row["rising_trend_flag"] == 1:
        reasons.append("rising risk trend")

    if row["change_point_flag"] == 1:
        reasons.append("detected change point")

    if row["baseline_deviation"] >= 0.20:
        reasons.append("risk is above personal baseline")

    if row["trajectory_slope_so_far"] >= 0.03:
        reasons.append("positive trajectory slope")

    if not reasons:
        reasons.append("low risk, low uncertainty, and stable trend")

    return reasons


def assign_policy(row):
    high_prob = row["mean_high_prob"]
    medium_prob = row["mean_medium_prob"]
    low_prob = row["mean_low_prob"]
    high_uncertainty = row["high_uncertainty_flag"] == 1
    rising_trend = row["rising_trend_flag"] == 1
    change_point = row["change_point_flag"] == 1
    baseline_elevated = row["baseline_deviation"] >= 0.20
    strong_positive_slope = row["trajectory_slope_so_far"] >= 0.03

    medium_or_high_risk = (medium_prob >= low_prob) or (high_prob >= 0.35)
    unstable_temporal_signal = rising_trend or change_point or baseline_elevated or strong_positive_slope

    if high_prob >= 0.90:
        return 4

    if high_prob >= 0.60:
        return 3

    if medium_or_high_risk and (high_uncertainty or unstable_temporal_signal):
        return 3

    if medium_prob >= low_prob and medium_prob >= high_prob:
        return 2

    if high_prob >= 0.25 or medium_prob >= 0.40:
        return 2

    if low_prob >= medium_prob and low_prob >= high_prob and (high_uncertainty or unstable_temporal_signal):
        return 1

    return 0


def make_rationale(level, reasons):
    reason_text = " and ".join(reasons[:3])
    return f"Escalated to Level {level} because {reason_text}."


def main():
    df = pd.read_csv(args.input_csv)
    validate_input(df)

    df["escalation_level"] = df.apply(assign_policy, axis=1)
    df["escalation_action"] = df["escalation_level"].map(LEVEL_ACTIONS)
    df["rationale"] = df.apply(
        lambda row: make_rationale(row["escalation_level"], active_reasons(row)),
        axis=1,
    )

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print("Escalation level summary:")
    counts = df["escalation_level"].value_counts().sort_index()
    for level in range(5):
        print(f"Level {level} - {LEVEL_ACTIONS[level]}: {int(counts.get(level, 0))}")

    print(f"Total rows processed: {len(df)}")
    print(f"Saved policy outputs to: {args.output_csv}")


if __name__ == "__main__":
    main()
