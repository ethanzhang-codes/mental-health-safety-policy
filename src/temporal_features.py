import argparse
import os

import numpy as np
import pandas as pd

try:
    import ruptures as rpt
except ImportError:
    rpt = None


REQUIRED_COLUMNS = [
    "conversation_id",
    "turn_id",
    "speaker_role",
    "text",
    "mean_low_prob",
    "mean_medium_prob",
    "mean_high_prob",
    "confidence",
    "epistemic_uncertainty",
    "aleatoric_uncertainty",
]


parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_csv", required=True)
parser.add_argument("--uncertainty-quantile", type=float, default=0.90)
parser.add_argument("--change-threshold", type=float, default=0.20)
args = parser.parse_args()


def validate_input(df):
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: {missing_columns}\n"
            f"Required columns are: {REQUIRED_COLUMNS}"
        )


def slope_so_far(values):
    slopes = []
    for end_idx in range(len(values)):
        y = values[: end_idx + 1]
        if len(y) < 2:
            slopes.append(0.0)
            continue

        x = np.arange(len(y), dtype=float)
        slope = np.polyfit(x, y, deg=1)[0]
        slopes.append(float(slope))

    return slopes


def detect_change_point_with_pelt(values):
    if rpt is None or len(values) < 6:
        return -1

    signal = np.asarray(values, dtype=float).reshape(-1, 1)
    penalty = max(0.01, np.var(signal) * np.log(len(signal)) * 2)
    change_points = rpt.Pelt(model="rbf", min_size=3).fit(signal).predict(pen=penalty)
    change_points = [point for point in change_points if point < len(values)]

    if not change_points:
        return -1

    return int(change_points[0])


def detect_change_point_fallback(values, threshold):
    if len(values) < 6:
        return -1

    values = np.asarray(values, dtype=float)
    best_idx = -1
    best_shift = 0.0

    for idx in range(3, len(values) - 2):
        before_mean = values[:idx].mean()
        after_mean = values[idx:].mean()
        shift = abs(after_mean - before_mean)
        if shift > best_shift:
            best_shift = shift
            best_idx = idx

    if best_shift >= threshold:
        return int(best_idx)

    return -1


def detect_change_point(values, threshold):
    pelt_change_point = detect_change_point_with_pelt(values)
    if pelt_change_point != -1:
        return pelt_change_point

    return detect_change_point_fallback(values, threshold)


def add_conversation_features(group, epistemic_threshold, aleatoric_threshold):
    group = group.sort_values("turn_id").copy()
    high_probs = group["mean_high_prob"].astype(float)
    high_values = high_probs.to_numpy()

    group["previous_high_prob"] = high_probs.shift(1)
    group["high_prob_change_from_previous_turn"] = high_probs - group["previous_high_prob"]
    group["rolling_mean_high_prob_3"] = high_probs.rolling(window=3, min_periods=1).mean()
    group["trajectory_slope_so_far"] = slope_so_far(high_values)

    baseline_window = min(3, len(group))
    baseline_high_prob = float(high_values[:baseline_window].mean())
    group["baseline_high_prob"] = baseline_high_prob
    group["baseline_deviation"] = high_probs - baseline_high_prob

    recent_slope = group["rolling_mean_high_prob_3"].diff()
    group["rising_trend_flag"] = (
        (recent_slope > 0)
        & (group["baseline_deviation"] > 0)
        & (group["high_prob_change_from_previous_turn"].fillna(0) > 0)
    ).astype(int)

    group["high_uncertainty_flag"] = (
        (group["epistemic_uncertainty"] >= epistemic_threshold)
        | (group["aleatoric_uncertainty"] >= aleatoric_threshold)
    ).astype(int)

    change_point_idx = detect_change_point(
        high_values,
        threshold=args.change_threshold,
    )
    if change_point_idx == -1:
        group["change_point_turn_index"] = -1
        group["change_point_flag"] = 0
    else:
        change_point_turn_id = group.iloc[change_point_idx]["turn_id"]
        group["change_point_turn_index"] = change_point_turn_id
        group["change_point_flag"] = 0
        group.loc[group.index[change_point_idx], "change_point_flag"] = 1

    return group


def main():
    df = pd.read_csv(args.input_csv)
    validate_input(df)

    df = df.sort_values(["conversation_id", "turn_id"]).reset_index(drop=True)

    epistemic_threshold = df["epistemic_uncertainty"].quantile(args.uncertainty_quantile)
    aleatoric_threshold = df["aleatoric_uncertainty"].quantile(args.uncertainty_quantile)

    featured_groups = [
        add_conversation_features(
            group,
            epistemic_threshold,
            aleatoric_threshold,
        )
        for _, group in df.groupby("conversation_id", sort=False)
    ]
    featured_df = pd.concat(featured_groups, ignore_index=True)

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    featured_df.to_csv(args.output_csv, index=False)

    conversation_count = featured_df["conversation_id"].nunique()
    total_turns = len(featured_df)
    avg_turns = total_turns / conversation_count if conversation_count else 0

    conversation_flags = featured_df.groupby("conversation_id").agg(
        has_rising_trend=("rising_trend_flag", "max"),
        has_change_point=("change_point_flag", "max"),
    )

    print(f"Number of conversations processed: {conversation_count}")
    print(f"Total turns processed: {total_turns}")
    print(f"Average turns per conversation: {avg_turns:.2f}")
    print(f"Number of conversations with rising trends: {int(conversation_flags['has_rising_trend'].sum())}")
    print(f"Number of conversations with change points: {int(conversation_flags['has_change_point'].sum())}")
    print(f"Epistemic uncertainty threshold: {epistemic_threshold:.6f}")
    print(f"Aleatoric uncertainty threshold: {aleatoric_threshold:.6f}")
    print(f"Change-point method: {'PELT via ruptures' if rpt is not None else 'fallback strongest-shift detector'}")
    print(f"Saved temporal features to: {args.output_csv}")


if __name__ == "__main__":
    main()
