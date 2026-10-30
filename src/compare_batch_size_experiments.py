import csv
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

CONFIGS = [
    {
        "encoder": "mental/mental-roberta-base",
        "encoder_short": "roberta_base",
        "batch_size": 32,
        "result_path": PROJECT_DIR / "results" / "seed_1.txt",
    },
    {
        "encoder": "mental/mental-roberta-base",
        "encoder_short": "roberta_base",
        "batch_size": 8,
        "result_path": PROJECT_DIR / "experiment_results" / "base_bs8" / "seed_1.txt",
    },
    {
        "encoder": "mental/mental-roberta-base",
        "encoder_short": "roberta_base",
        "batch_size": 4,
        "result_path": PROJECT_DIR / "experiment_results" / "base_bs4" / "seed_1.txt",
    },
    {
        "encoder": "AIMH/mental-bert-large-cased",
        "encoder_short": "mental_bert_large",
        "batch_size": 32,
        "result_path": PROJECT_DIR / "experiment_results" / "large_bs32" / "seed_1.txt",
    },
    {
        "encoder": "AIMH/mental-bert-large-cased",
        "encoder_short": "mental_bert_large",
        "batch_size": 8,
        "result_path": PROJECT_DIR / "experiment_results" / "large_bs8" / "seed_1.txt",
    },
    {
        "encoder": "AIMH/mental-bert-large-cased",
        "encoder_short": "mental_bert_large",
        "batch_size": 4,
        "result_path": PROJECT_DIR / "experiment_results" / "large_bs4" / "seed_1.txt",
    },
]

OUTPUT_DIR = PROJECT_DIR / "experiment_results"
OUTPUT_TXT = OUTPUT_DIR / "batch_size_comparison.txt"
OUTPUT_CSV = OUTPUT_DIR / "batch_size_comparison.csv"


def parse_float(pattern, text, field_name, path):
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not find {field_name} in {path}")
    return float(match.group(1))


def parse_epoch_metrics(text, path):
    epoch_rows = re.findall(
        r"Epoch\s+(\d+):.*?Val Macro-F1=([0-9.]+)",
        text,
    )
    if not epoch_rows:
        raise ValueError(f"Could not find epoch validation metrics in {path}")

    best_epoch, best_val = max(
        ((int(epoch), float(val_f1)) for epoch, val_f1 in epoch_rows),
        key=lambda row: row[1],
    )
    return best_epoch, best_val


def parse_high_risk_metrics(text, path):
    high_risk_match = re.search(
        r"High Risk\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+",
        text,
    )
    if not high_risk_match:
        raise ValueError(f"Could not find High Risk metrics in {path}")

    return {
        "high_risk_precision": float(high_risk_match.group(1)),
        "high_risk_recall": float(high_risk_match.group(2)),
        "high_risk_f1": float(high_risk_match.group(3)),
    }


def parse_result_file(config):
    path = config["result_path"]
    if not path.exists():
        row = config.copy()
        row.update(
            {
                "status": "missing",
                "best_val_macro_f1": None,
                "test_macro_f1": None,
                "high_risk_precision": None,
                "high_risk_recall": None,
                "high_risk_f1": None,
                "accuracy": None,
                "best_val_epoch": None,
                "notes": f"Missing result file: {path.relative_to(PROJECT_DIR)}",
            }
        )
        return row

    text = path.read_text()
    best_epoch, best_val_from_epochs = parse_epoch_metrics(text, path)
    best_val_reported = parse_float(r"Best Val Macro-F1:\s+([0-9.]+)", text, "Best Val Macro-F1", path)
    test_macro_f1 = parse_float(r"Final Test Macro-F1:\s+([0-9.]+)", text, "Final Test Macro-F1", path)
    accuracy = parse_float(r"accuracy\s+([0-9.]+)\s+\d+", text, "accuracy", path)
    high_risk = parse_high_risk_metrics(text, path)

    row = config.copy()
    row.update(
        {
            "status": "complete",
            "best_val_macro_f1": best_val_reported,
            "test_macro_f1": test_macro_f1,
            "accuracy": accuracy,
            "best_val_epoch": best_epoch,
            "notes": "",
        }
    )
    row.update(high_risk)

    if abs(best_val_reported - best_val_from_epochs) > 0.0001:
        row["notes"] = (
            f"Reported best val {best_val_reported:.4f} differs from parsed epoch max "
            f"{best_val_from_epochs:.4f}."
        )

    return row


def pick_best(rows, encoder):
    completed = [row for row in rows if row["encoder"] == encoder and row["status"] == "complete"]
    if not completed:
        return None

    return max(
        completed,
        key=lambda row: (
            row["best_val_macro_f1"],
            row["test_macro_f1"],
            row["high_risk_recall"],
            row["high_risk_f1"],
        ),
    )


def format_metric(value):
    if value is None:
        return "MISSING"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def make_markdown_table(rows):
    headers = [
        "Rank",
        "Encoder",
        "Batch",
        "Best Val Macro-F1",
        "Best Epoch",
        "Test Macro-F1",
        "High Precision",
        "High Recall",
        "High F1",
        "Accuracy",
        "Status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for rank, row in enumerate(rows, start=1):
        rank_text = str(rank) if row["status"] == "complete" else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    rank_text,
                    row["encoder_short"],
                    str(row["batch_size"]),
                    format_metric(row["best_val_macro_f1"]),
                    format_metric(row["best_val_epoch"]),
                    format_metric(row["test_macro_f1"]),
                    format_metric(row["high_risk_precision"]),
                    format_metric(row["high_risk_recall"]),
                    format_metric(row["high_risk_f1"]),
                    format_metric(row["accuracy"]),
                    row["status"],
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [parse_result_file(config) for config in CONFIGS]
    completed_rows = [row for row in rows if row["status"] == "complete"]
    missing_rows = [row for row in rows if row["status"] != "complete"]

    ranked_rows = sorted(
        completed_rows,
        key=lambda row: (
            row["best_val_macro_f1"],
            row["test_macro_f1"],
            row["high_risk_recall"],
            row["high_risk_f1"],
        ),
        reverse=True,
    ) + missing_rows

    best_base = pick_best(rows, "mental/mental-roberta-base")
    best_large = pick_best(rows, "AIMH/mental-bert-large-cased")

    lines = [
        "Controlled Batch-Size Experiment Comparison",
        "===========================================",
        "",
        "All configurations are seed 1. This report compares the result files currently present on disk.",
        "Do not treat this as a global encoder conclusion because it uses only one seed.",
        "",
        "Ranked comparison table:",
        make_markdown_table(ranked_rows),
        "",
        "Best batch size by encoder:",
    ]

    if best_base:
        lines.append(
            "1. mental/mental-roberta-base: "
            f"batch size {best_base['batch_size']} appears best "
            f"(best validation Macro-F1={best_base['best_val_macro_f1']:.4f}, "
            f"test Macro-F1={best_base['test_macro_f1']:.4f})."
        )
    else:
        lines.append("1. mental/mental-roberta-base: no completed runs found.")

    if best_large:
        lines.append(
            "2. AIMH/mental-bert-large-cased: "
            f"batch size {best_large['batch_size']} appears best among completed runs "
            f"(best validation Macro-F1={best_large['best_val_macro_f1']:.4f}, "
            f"test Macro-F1={best_large['test_macro_f1']:.4f})."
        )
    else:
        lines.append("2. AIMH/mental-bert-large-cased: no completed runs found.")

    if missing_rows:
        lines.extend(
            [
                "",
                "Missing or incomplete configurations:",
            ]
        )
        for row in missing_rows:
            lines.append(
                f"- {row['encoder']} batch {row['batch_size']}: {row['notes']}"
            )

    lines.extend(
        [
            "",
            "Interpretation note:",
            "The ranking prioritizes best validation Macro-F1, then test Macro-F1, high-risk recall, and high-risk F1 as tie-breakers.",
        ]
    )

    report = "\n".join(lines)
    OUTPUT_TXT.write_text(report + "\n")

    csv_fields = [
        "rank",
        "encoder",
        "encoder_short",
        "batch_size",
        "best_val_macro_f1",
        "test_macro_f1",
        "high_risk_precision",
        "high_risk_recall",
        "high_risk_f1",
        "accuracy",
        "best_val_epoch",
        "status",
        "result_path",
        "notes",
    ]

    with OUTPUT_CSV.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()
        for rank, row in enumerate(ranked_rows, start=1):
            csv_row = {field: row.get(field) for field in csv_fields}
            csv_row["rank"] = rank if row["status"] == "complete" else ""
            csv_row["result_path"] = str(row["result_path"].relative_to(PROJECT_DIR))
            writer.writerow(csv_row)

    print(report)
    print(f"\nSaved text report to: {OUTPUT_TXT}")
    print(f"Saved CSV table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
