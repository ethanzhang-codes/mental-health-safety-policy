import argparse
import os

import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_txt", required=True)
args = parser.parse_args()


REQUIRED_POLICY_COLUMNS = [
    "conversation_id",
    "turn_id",
    "predicted_label",
    "escalation_level",
]

LABEL_NAMES = ["Low Risk", "Medium Risk", "High Risk"]


def validate_input(df):
    missing_columns = [
        column for column in REQUIRED_POLICY_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required evaluation columns: {missing_columns}"
        )


def choose_label_column(df):
    if "true_label" in df.columns:
        return "true_label", False
    if "risk_severity" in df.columns:
        return "risk_severity", False
    return "predicted_label", True


def percent(numerator, denominator):
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def predictive_metrics(df, label_column, pseudo_label_eval):
    y_true = df[label_column].astype(int)
    y_pred = df["predicted_label"].astype(int)

    lines = []
    if pseudo_label_eval:
        lines.append(
            "NOTE: No true clinical risk labels were found. "
            "Predictive metrics below compare model predictions against their own pseudo-labels, "
            "so they are sanity checks, not true clinical performance."
        )
    else:
        lines.append("Predictive metrics use available ground-truth labels.")

    lines.append(f"Label column used: {label_column}")
    lines.append(f"Macro-F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    lines.append(
        "Precision for high-risk cases: "
        f"{precision_score(y_true, y_pred, labels=[2], average='macro', zero_division=0):.4f}"
    )
    lines.append(
        "Recall for high-risk cases: "
        f"{recall_score(y_true, y_pred, labels=[2], average='macro', zero_division=0):.4f}"
    )
    lines.append("")
    lines.append("Classification report:")
    lines.append(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    return lines


def safety_metrics(df, label_column, pseudo_label_eval):
    labels = df[label_column].astype(int)
    escalation = df["escalation_level"].astype(int)

    high_risk_turns = labels == 2
    low_risk_turns = labels == 0
    unsafe_continuations = high_risk_turns & escalation.isin([0, 1])
    over_escalations = low_risk_turns & escalation.isin([3, 4])
    autonomous_turns = ~escalation.isin([3, 4])

    level_counts = escalation.value_counts().sort_index()
    time_to_escalation = (
        df[df["escalation_level"] >= 3]
        .sort_values(["conversation_id", "turn_id"])
        .groupby("conversation_id")["turn_id"]
        .first()
    )

    num_conversations = df["conversation_id"].nunique()
    num_escalated_conversations = len(time_to_escalation)
    avg_time_to_escalation = time_to_escalation.mean() if num_escalated_conversations else -1
    median_time_to_escalation = time_to_escalation.median() if num_escalated_conversations else -1

    conversation_high_risk = df.assign(_label=labels).groupby("conversation_id")["_label"].max() == 2
    conversation_unsafe = (
        df.assign(_unsafe=unsafe_continuations)
        .groupby("conversation_id")["_unsafe"]
        .any()
    )
    unsafe_high_risk_conversations = conversation_high_risk & conversation_unsafe

    lines = []
    if pseudo_label_eval:
        lines.append(
            "NOTE: Safety metrics use predicted labels as pseudo-labels because true risk labels are missing."
        )

    lines.append(
        "Unsafe continuation rate, turn-level: "
        f"{percent(unsafe_continuations.sum(), high_risk_turns.sum()):.2f}% "
        f"({int(unsafe_continuations.sum())}/{int(high_risk_turns.sum())} high-risk turns at Level 0 or 1)"
    )
    lines.append(
        "Unsafe continuation rate, conversation-level: "
        f"{percent(unsafe_high_risk_conversations.sum(), conversation_high_risk.sum()):.2f}% "
        f"({int(unsafe_high_risk_conversations.sum())}/{int(conversation_high_risk.sum())} high-risk conversations)"
    )
    lines.append(
        "Over-escalation burden: "
        f"{percent(over_escalations.sum(), low_risk_turns.sum()):.2f}% "
        f"({int(over_escalations.sum())}/{int(low_risk_turns.sum())} low-risk turns at Level 3 or 4)"
    )
    lines.append(
        "Autonomy / coverage rate: "
        f"{percent(autonomous_turns.sum(), len(df)):.2f}% "
        f"({int(autonomous_turns.sum())}/{len(df)} turns handled below Level 3)"
    )
    lines.append("")
    lines.append("Escalation level distribution:")
    for level in range(5):
        count = int(level_counts.get(level, 0))
        lines.append(f"Level {level}: {count} ({percent(count, len(df)):.2f}%)")

    lines.append("")
    lines.append("Time-to-escalation:")
    lines.append(f"Conversations with Level >= 3 escalation: {num_escalated_conversations}/{num_conversations}")
    lines.append(f"Average first escalation turn_id: {avg_time_to_escalation:.2f}")
    lines.append(f"Median first escalation turn_id: {median_time_to_escalation:.2f}")

    return lines


def build_report(df):
    label_column, pseudo_label_eval = choose_label_column(df)

    lines = []
    lines.append("Framework Evaluation Report")
    lines.append("===========================")
    lines.append("")
    lines.append(f"Input CSV: {args.input_csv}")
    lines.append(f"Rows evaluated: {len(df)}")
    lines.append(f"Conversations evaluated: {df['conversation_id'].nunique()}")
    lines.append("")
    lines.append("1. Predictive Metrics")
    lines.append("---------------------")
    lines.extend(predictive_metrics(df, label_column, pseudo_label_eval))
    lines.append("")
    lines.append("2. Safety-Centered Metrics")
    lines.append("--------------------------")
    lines.extend(safety_metrics(df, label_column, pseudo_label_eval))
    lines.append("")
    lines.append("Interpretation Notes")
    lines.append("--------------------")
    lines.append(
        "This evaluation aligns with the fixed abstract by checking predictive performance, "
        "graded escalation behavior, unsafe continuation risk, over-escalation burden, "
        "time-to-escalation, and autonomous coverage."
    )
    if pseudo_label_eval:
        lines.append(
            "Because AnnoMI does not provide true clinical escalation/risk labels here, "
            "these results should not be reported as validated clinical safety performance."
        )

    return "\n".join(lines)


def main():
    df = pd.read_csv(args.input_csv)
    validate_input(df)

    report = build_report(df)
    print(report)

    output_dir = os.path.dirname(args.output_txt)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_txt, "w") as output_file:
        output_file.write(report)
        output_file.write("\n")

    print(f"\nSaved evaluation report to: {args.output_txt}")


if __name__ == "__main__":
    main()
