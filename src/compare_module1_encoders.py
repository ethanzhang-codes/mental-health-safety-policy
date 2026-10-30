import argparse
import csv
import re
from pathlib import Path

from module1_benchmark_config import BENCHMARK_ROOT, RESULTS_ROOT


OUTPUT_CSV = RESULTS_ROOT / "seed1_batch_screening.csv"
OUTPUT_TXT = RESULTS_ROOT / "seed1_batch_screening.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare isolated Module 1 encoder benchmark results."
    )
    parser.add_argument("--results_root", default=str(RESULTS_ROOT))
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def read_value(pattern, text, default=None, cast=str):
    match = re.search(pattern, text)
    if not match:
        return default
    value = match.group(1).strip()
    if value == "not_available":
        return default
    return cast(value)


def parse_result(path):
    text = path.read_text(errors="ignore")
    row = {
        "result_file": str(path.relative_to(BENCHMARK_ROOT)),
        "status": read_value(r"Run status:\s*(.+)", text, "unknown"),
        "model": read_value(r"Model:\s*(.+)", text, ""),
        "encoder_short": read_value(r"Encoder short name:\s*(.+)", text, ""),
        "seed": read_value(r"Seed:\s*(\d+)", text, None, int),
        "split_seed": read_value(r"Split seed:\s*(\d+)", text, None, int),
        "split_checksum": read_value(r"Split checksum:\s*(.+)", text, ""),
        "batch_size": read_value(r"Batch size:\s*(\d+)", text, None, int),
        "precision_dtype": read_value(r"Precision dtype:\s*(.+)", text, ""),
        "best_val_macro_f1": read_value(
            r"Best Val Macro-F1:\s*([0-9.]+)", text, None, float
        ),
        "best_epoch": read_value(r"Best epoch:\s*(\d+)", text, None, int),
        "test_macro_f1": read_value(r"Test Macro-F1:\s*([0-9.]+)", text, None, float),
        "high_risk_precision": read_value(
            r"High-Risk precision:\s*([0-9.]+)", text, None, float
        ),
        "high_risk_recall": read_value(
            r"High-Risk recall:\s*([0-9.]+)", text, None, float
        ),
        "high_risk_f1": read_value(r"High-Risk F1:\s*([0-9.]+)", text, None, float),
        "accuracy": read_value(r"Accuracy:\s*([0-9.]+)", text, None, float),
        "raw_nll": read_value(r"Raw test NLL:\s*([0-9.]+)", text, None, float),
        "raw_ece": read_value(r"Raw ECE:\s*([0-9.]+)", text, None, float),
        "average_confidence": read_value(
            r"Average confidence:\s*([0-9.]+)", text, None, float
        ),
        "calibrated_nll": read_value(
            r"Calibrated NLL:\s*([0-9.]+|not_available)", text, None, float
        ),
        "calibrated_ece": read_value(
            r"Calibrated ECE:\s*([0-9.]+|not_available)", text, None, float
        ),
        "training_duration_seconds": read_value(
            r"Training duration seconds:\s*([0-9.]+)", text, None, float
        ),
    }
    return row


def metric_key(row):
    if row["status"] != "complete":
        return (-1.0, -1.0, -1.0, -1.0)
    return (
        row["best_val_macro_f1"] or -1.0,
        row["high_risk_recall"] or -1.0,
        row["high_risk_f1"] or -1.0,
        -(row["raw_ece"] or 999.0),
    )


def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_table(rows):
    headers = [
        "Rank",
        "Encoder",
        "Batch",
        "Status",
        "Best Val Macro-F1",
        "Best Epoch",
        "Test Macro-F1",
        "High Recall",
        "High Precision",
        "High F1",
        "Accuracy",
        "Raw NLL",
        "Raw ECE",
        "Avg Conf",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in enumerate(rows, start=1):
        rank = str(idx) if row["status"] == "complete" else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    rank,
                    row["encoder_short"],
                    fmt(row["batch_size"]),
                    row["status"],
                    fmt(row["best_val_macro_f1"]),
                    fmt(row["best_epoch"]),
                    fmt(row["test_macro_f1"]),
                    fmt(row["high_risk_recall"]),
                    fmt(row["high_risk_precision"]),
                    fmt(row["high_risk_f1"]),
                    fmt(row["accuracy"]),
                    fmt(row["raw_nll"]),
                    fmt(row["raw_ece"]),
                    fmt(row["average_confidence"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    root = Path(args.results_root)
    result_files = sorted(root.glob(f"*/bs*/seed_{args.seed}.txt"))
    rows = [parse_result(path) for path in result_files]
    rows = sorted(rows, key=metric_key, reverse=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "model",
        "encoder_short",
        "seed",
        "split_seed",
        "split_checksum",
        "batch_size",
        "precision_dtype",
        "status",
        "best_val_macro_f1",
        "best_epoch",
        "test_macro_f1",
        "high_risk_precision",
        "high_risk_recall",
        "high_risk_f1",
        "accuracy",
        "raw_nll",
        "raw_ece",
        "average_confidence",
        "calibrated_nll",
        "calibrated_ece",
        "training_duration_seconds",
        "result_file",
    ]

    with OUTPUT_CSV.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            csv_row = {name: row.get(name) for name in fieldnames}
            csv_row["rank"] = idx if row["status"] == "complete" else ""
            writer.writerow(csv_row)

    lines = [
        "Module 1 Seed-1 Encoder Batch Screening",
        "=======================================",
        "",
        "This is only a single-turn Low/Medium/High risk severity benchmark.",
        "It does not evaluate category classification or temporal progression.",
        "Do not declare a global encoder winner from one seed.",
        "",
        markdown_table(rows),
        "",
        "Safety-oriented interpretation checklist:",
        "1. High-Risk Recall: missed high-risk cases are especially important.",
        "2. High-Risk Precision: false high-risk alarms increase escalation burden.",
        "3. High-Risk F1: balances high-risk precision and recall.",
        "4. Macro-F1: checks balance across Low, Medium, and High classes.",
        "5. Calibration / ECE: lower ECE means confidence is more trustworthy.",
        "6. Stability: final claims require multiple seeds.",
    ]
    OUTPUT_TXT.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved text report: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
