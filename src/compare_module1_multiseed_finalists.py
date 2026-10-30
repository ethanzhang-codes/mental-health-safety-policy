import csv
import math
import re
import statistics
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = PROJECT_DIR / "results" / "module1_multiseed_finalist_comparison.csv"
OUTPUT_TXT = PROJECT_DIR / "results" / "module1_multiseed_finalist_comparison.txt"


FINALISTS = [
    {
        "encoder_short": "deberta_v3_large",
        "model_name": "microsoft/deberta-v3-large",
        "batch_size": 32,
        "precision_dtype": "fp32",
    },
    {
        "encoder_short": "modernbert_large",
        "model_name": "answerdotai/ModernBERT-large",
        "batch_size": 32,
        "precision_dtype": "fp32",
    },
    {
        "encoder_short": "mental_roberta_large",
        "model_name": "AIMH/mental-roberta-large",
        "batch_size": 32,
        "precision_dtype": "fp32",
    },
]

SEEDS = [1, 2, 3, 4, 5]
HIGH_RISK_SUPPORT_DEFAULT = 351


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def read_value(pattern, text, default=None, cast=str):
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return default
    raw = match.group(1).strip()
    if raw.lower() in {"", "nan", "none", "not_available"}:
        return default
    try:
        return cast(raw)
    except ValueError:
        return default


def metric_from_report(label, text):
    match = re.search(rf"{label}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", text)
    if not match:
        return None, None, None, None
    return float(match.group(1)), float(match.group(2)), float(match.group(3)), int(match.group(4))


def finite(value):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def result_path(encoder_short, batch_size, seed):
    return (
        PROJECT_DIR
        / "module1_encoder_benchmark"
        / "results"
        / encoder_short
        / f"bs{batch_size}"
        / f"seed_{seed}.txt"
    )


def parse_result(finalist, seed):
    path = result_path(finalist["encoder_short"], finalist["batch_size"], seed)
    text = read_text(path)
    high_p, high_r, high_f1, high_support = metric_from_report("High Risk", text)
    status = read_value(r"Run status:\s*(.+)", text, "missing" if not path.exists() else "unknown")
    precision_dtype = read_value(
        r"Precision dtype:\s*(.+)",
        text,
        "not_recorded" if seed == 1 and finalist["encoder_short"] == "mental_roberta_large" else finalist["precision_dtype"],
    )
    high_risk_recall = finite(read_value(r"High-Risk recall:\s*([0-9.]+|nan)", text, high_r, float))
    high_risk_support = high_support or HIGH_RISK_SUPPORT_DEFAULT
    high_risk_missed = (
        round(high_risk_support * (1 - high_risk_recall))
        if high_risk_recall is not None
        else None
    )

    row = {
        "row_type": "seed",
        "encoder_short": finalist["encoder_short"],
        "model_name": read_value(r"Model:\s*(.+)", text, finalist["model_name"]),
        "batch_size": read_value(r"Batch size:\s*(\d+)", text, finalist["batch_size"], int),
        "precision_dtype": precision_dtype,
        "seed": read_value(r"Seed:\s*(\d+)", text, seed, int),
        "split_seed": read_value(r"Split seed:\s*(\d+)", text, None, int),
        "split_checksum": read_value(r"Split checksum:\s*(.+)", text, ""),
        "run_status": status,
        "best_val_macro_f1": finite(read_value(r"Best Val Macro-F1:\s*([0-9.]+|nan)", text, None, float)),
        "best_epoch": read_value(r"Best epoch:\s*(\d+)", text, None, int),
        "test_macro_f1": finite(read_value(r"Test Macro-F1:\s*([0-9.]+|nan)", text, None, float)),
        "accuracy": finite(read_value(r"Accuracy:\s*([0-9.]+|nan)", text, None, float)),
        "high_risk_precision": finite(read_value(r"High-Risk precision:\s*([0-9.]+|nan)", text, high_p, float)),
        "high_risk_recall": high_risk_recall,
        "high_risk_f1": finite(read_value(r"High-Risk F1:\s*([0-9.]+|nan)", text, high_f1, float)),
        "high_risk_support": high_risk_support,
        "high_risk_missed_estimate": high_risk_missed,
        "raw_nll": finite(read_value(r"Raw test NLL:\s*([0-9.]+|nan)", text, None, float)),
        "raw_ece": finite(read_value(r"Raw ECE:\s*([0-9.]+|nan)", text, None, float)),
        "calibrated_nll": finite(read_value(r"Calibrated NLL:\s*([0-9.]+|nan)", text, None, float)),
        "calibrated_ece": finite(read_value(r"Calibrated ECE:\s*([0-9.]+|nan)", text, None, float)),
        "temperature": finite(read_value(r"Learned temperature:\s*([0-9.]+|nan)", text, None, float)),
        "training_duration_seconds": finite(read_value(r"Training duration seconds:\s*([0-9.]+|nan)", text, None, float)),
        "result_file": str(path.relative_to(PROJECT_DIR)),
    }
    row["numerical_stability_status"] = "stable"
    if status != "complete":
        row["numerical_stability_status"] = status
    elif re.search(r"\bnan\b|\binf\b|non-finite", text, flags=re.IGNORECASE):
        row["numerical_stability_status"] = "contains_nonfinite_marker"
    return row


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def stdev(values):
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) > 1 else 0.0 if len(values) == 1 else None


def minimum(values):
    values = [v for v in values if v is not None]
    return min(values) if values else None


def all_stable(rows):
    return all(row["run_status"] == "complete" and row["numerical_stability_status"] == "stable" for row in rows)


def summarize(finalist, rows):
    def vals(key):
        return [row[key] for row in rows]

    summary = {
        "row_type": "summary",
        "encoder_short": finalist["encoder_short"],
        "model_name": finalist["model_name"],
        "batch_size": finalist["batch_size"],
        "precision_dtype": finalist["precision_dtype"],
        "seed": "1-5",
        "split_seed": 1,
        "split_checksum": rows[0]["split_checksum"] if rows else "",
        "run_status": "complete" if all(row["run_status"] == "complete" for row in rows) else "incomplete",
        "mean_best_val_macro_f1": mean(vals("best_val_macro_f1")),
        "std_best_val_macro_f1": stdev(vals("best_val_macro_f1")),
        "mean_test_macro_f1": mean(vals("test_macro_f1")),
        "std_test_macro_f1": stdev(vals("test_macro_f1")),
        "mean_high_risk_precision": mean(vals("high_risk_precision")),
        "std_high_risk_precision": stdev(vals("high_risk_precision")),
        "mean_high_risk_recall": mean(vals("high_risk_recall")),
        "std_high_risk_recall": stdev(vals("high_risk_recall")),
        "min_high_risk_recall": minimum(vals("high_risk_recall")),
        "total_high_risk_missed_estimate": sum(
            row["high_risk_missed_estimate"] for row in rows if row["high_risk_missed_estimate"] is not None
        ),
        "mean_high_risk_f1": mean(vals("high_risk_f1")),
        "mean_accuracy": mean(vals("accuracy")),
        "mean_raw_ece": mean(vals("raw_ece")),
        "mean_calibrated_ece": mean(vals("calibrated_ece")),
        "std_calibrated_ece": stdev(vals("calibrated_ece")),
        "mean_temperature": mean(vals("temperature")),
        "numerical_stability_status": "stable_all_seeds" if all_stable(rows) else "review_needed",
        "completed_seed_count": sum(row["run_status"] == "complete" for row in rows),
    }
    return summary


def rank_key(summary):
    # Safety first: consistent high-risk recall and fewer high-risk misses outrank raw Macro-F1.
    stable_score = 1 if summary["numerical_stability_status"] == "stable_all_seeds" else 0
    return (
        summary["min_high_risk_recall"] or -1,
        -(summary["std_high_risk_recall"] if summary["std_high_risk_recall"] is not None else 999),
        -(summary["total_high_risk_missed_estimate"] if summary["total_high_risk_missed_estimate"] is not None else 999999),
        summary["mean_high_risk_precision"] or -1,
        summary["mean_best_val_macro_f1"] or -1,
        summary["mean_test_macro_f1"] or -1,
        -(summary["mean_calibrated_ece"] if summary["mean_calibrated_ece"] is not None else 999),
        stable_score,
    )


def fmt(value):
    if value is None:
        return "not_available"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(summaries, seed_rows):
    fields = [
        "row_type",
        "rank",
        "encoder_short",
        "model_name",
        "batch_size",
        "precision_dtype",
        "seed",
        "split_seed",
        "split_checksum",
        "run_status",
        "best_val_macro_f1",
        "best_epoch",
        "test_macro_f1",
        "accuracy",
        "high_risk_precision",
        "high_risk_recall",
        "high_risk_f1",
        "high_risk_support",
        "high_risk_missed_estimate",
        "raw_nll",
        "raw_ece",
        "calibrated_nll",
        "calibrated_ece",
        "temperature",
        "mean_best_val_macro_f1",
        "std_best_val_macro_f1",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_high_risk_precision",
        "std_high_risk_precision",
        "mean_high_risk_recall",
        "std_high_risk_recall",
        "min_high_risk_recall",
        "total_high_risk_missed_estimate",
        "mean_high_risk_f1",
        "mean_accuracy",
        "mean_raw_ece",
        "mean_calibrated_ece",
        "std_calibrated_ece",
        "mean_temperature",
        "numerical_stability_status",
        "completed_seed_count",
        "training_duration_seconds",
        "result_file",
    ]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for idx, summary in enumerate(summaries, start=1):
            row = {field: summary.get(field) for field in fields}
            row["rank"] = idx
            writer.writerow(row)
        for row in seed_rows:
            csv_row = {field: row.get(field) for field in fields}
            writer.writerow(csv_row)


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def recommendation(summaries):
    best = summaries[0]
    return [
        "Recommendation",
        "--------------",
        (
            f"Recommended Module 1 encoder: {best['model_name']} "
            f"({best['encoder_short']}, batch size {best['batch_size']}, {best['precision_dtype']})."
        ),
        (
            "This recommendation follows the safety ranking first: minimum high-risk recall, "
            "recall consistency, and total estimated high-risk misses are considered before Macro-F1."
        ),
        (
            f"Across seeds, it had min high-risk recall {fmt(best['min_high_risk_recall'])}, "
            f"mean high-risk recall {fmt(best['mean_high_risk_recall'])}, "
            f"and total estimated high-risk misses {fmt(best['total_high_risk_missed_estimate'])}."
        ),
        (
            f"Calibration behavior: mean raw ECE {fmt(best['mean_raw_ece'])}, "
            f"mean calibrated ECE {fmt(best['mean_calibrated_ece'])}, "
            f"mean learned temperature {fmt(best['mean_temperature'])}."
        ),
        (
            f"Numerical stability: {best['numerical_stability_status']}. "
            "Any size/complexity tradeoff should be judged against this safety profile, not only average Macro-F1."
        ),
    ]


def write_text(summaries, seed_rows):
    summary_rows = []
    for idx, row in enumerate(summaries, start=1):
        summary_rows.append(
            [
                str(idx),
                row["encoder_short"],
                fmt(row["mean_high_risk_recall"]),
                fmt(row["std_high_risk_recall"]),
                fmt(row["min_high_risk_recall"]),
                fmt(row["total_high_risk_missed_estimate"]),
                fmt(row["mean_high_risk_precision"]),
                fmt(row["mean_best_val_macro_f1"]),
                fmt(row["mean_test_macro_f1"]),
                fmt(row["mean_calibrated_ece"]),
                row["numerical_stability_status"],
            ]
        )

    seed_table_rows = []
    for row in seed_rows:
        seed_table_rows.append(
            [
                row["encoder_short"],
                str(row["seed"]),
                row["run_status"],
                fmt(row["best_val_macro_f1"]),
                fmt(row["best_epoch"]),
                fmt(row["test_macro_f1"]),
                fmt(row["high_risk_precision"]),
                fmt(row["high_risk_recall"]),
                fmt(row["high_risk_f1"]),
                fmt(row["high_risk_missed_estimate"]),
                fmt(row["raw_ece"]),
                fmt(row["calibrated_ece"]),
                fmt(row["temperature"]),
                row["numerical_stability_status"],
            ]
        )

    lines = [
        "Module 1 Multi-Seed Finalist Comparison",
        "=======================================",
        "",
        "This is a severity-only Module 1 encoder benchmark for the fixed accepted project.",
        "It does not implement the later multi-task, temporal, uncertainty ensemble, or policy modules.",
        "",
        "Safety Ranking Rule",
        "-------------------",
        "Rank first by minimum high-risk recall, then recall consistency, total estimated high-risk misses, high-risk precision, validation Macro-F1, test Macro-F1, calibrated ECE, and numerical stability.",
        "",
        "Finalist Summary",
        "----------------",
        table(
            [
                "Rank",
                "Encoder",
                "Mean High R",
                "Std High R",
                "Min High R",
                "Total Misses",
                "Mean High P",
                "Mean Val F1",
                "Mean Test F1",
                "Mean Cal ECE",
                "Stability",
            ],
            summary_rows,
        ),
        "",
        "Individual Seed Metrics",
        "-----------------------",
        table(
            [
                "Encoder",
                "Seed",
                "Status",
                "Val F1",
                "Best Epoch",
                "Test F1",
                "High P",
                "High R",
                "High F1",
                "Misses",
                "Raw ECE",
                "Cal ECE",
                "Temp",
                "Stability",
            ],
            seed_table_rows,
        ),
        "",
    ]
    lines.extend(recommendation(summaries))
    lines.extend(
        [
            "",
            "Outputs",
            "-------",
            f"CSV: {OUTPUT_CSV.relative_to(PROJECT_DIR)}",
            f"Text report: {OUTPUT_TXT.relative_to(PROJECT_DIR)}",
        ]
    )
    OUTPUT_TXT.write_text("\n".join(lines) + "\n")


def main():
    all_seed_rows = []
    summaries = []
    for finalist in FINALISTS:
        rows = [parse_result(finalist, seed) for seed in SEEDS]
        all_seed_rows.extend(rows)
        summaries.append(summarize(finalist, rows))

    summaries = sorted(summaries, key=rank_key, reverse=True)
    all_seed_rows = sorted(all_seed_rows, key=lambda r: (r["encoder_short"], int(r["seed"])))

    write_csv(summaries, all_seed_rows)
    write_text(summaries, all_seed_rows)

    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved text report: {OUTPUT_TXT}")
    print("\nFinalist ranking:")
    for idx, row in enumerate(summaries, start=1):
        print(
            f"{idx}. {row['encoder_short']} | min high-risk recall={fmt(row['min_high_risk_recall'])}, "
            f"total misses={fmt(row['total_high_risk_missed_estimate'])}, "
            f"mean test Macro-F1={fmt(row['mean_test_macro_f1'])}, "
            f"mean cal ECE={fmt(row['mean_calibrated_ece'])}"
        )


if __name__ == "__main__":
    main()
