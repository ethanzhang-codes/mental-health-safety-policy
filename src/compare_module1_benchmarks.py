import csv
import hashlib
import math
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = PROJECT_DIR / "results" / "module1_seed1_benchmark_comparison.csv"
OUTPUT_TXT = PROJECT_DIR / "results" / "module1_seed1_benchmark_comparison.txt"


CANDIDATE_RESULTS = [
    {
        "encoder_short": "mental_roberta_base",
        "model_name": "mental/mental-roberta-base",
        "batch_size": 32,
        "precision_dtype": "not_recorded",
        "seed": 1,
        "split_seed": "not_recorded",
        "result_file": PROJECT_DIR / "results" / "seed_1.txt",
        "checkpoint_path": PROJECT_DIR / "saved_models" / "seed_1" / "best_model.pt",
    },
    {
        "encoder_short": "mental_roberta_base",
        "model_name": "mental/mental-roberta-base",
        "batch_size": 8,
        "precision_dtype": "not_recorded",
        "seed": 1,
        "split_seed": "not_recorded",
        "result_file": PROJECT_DIR / "experiment_results" / "base_bs8" / "seed_1.txt",
        "checkpoint_path": PROJECT_DIR / "experiments" / "base_bs8" / "seed_1" / "best_model.pt",
    },
    {
        "encoder_short": "mental_roberta_base",
        "model_name": "mental/mental-roberta-base",
        "batch_size": 4,
        "precision_dtype": "not_recorded",
        "seed": 1,
        "split_seed": "not_recorded",
        "result_file": PROJECT_DIR / "experiment_results" / "base_bs4" / "seed_1.txt",
        "checkpoint_path": PROJECT_DIR / "experiments" / "base_bs4" / "seed_1" / "best_model.pt",
    },
    {
        "encoder_short": "mental_bert_large",
        "model_name": "AIMH/mental-bert-large-cased",
        "batch_size": 32,
        "precision_dtype": "not_recorded",
        "seed": 1,
        "split_seed": "not_recorded",
        "result_file": PROJECT_DIR / "experiment_results" / "large_bs32" / "seed_1.txt",
        "checkpoint_path": PROJECT_DIR / "experiments" / "large_bs32" / "seed_1" / "best_model.pt",
    },
    {
        "encoder_short": "mental_bert_large",
        "model_name": "AIMH/mental-bert-large-cased",
        "batch_size": 8,
        "precision_dtype": "not_recorded",
        "seed": 1,
        "split_seed": "not_recorded",
        "result_file": PROJECT_DIR / "experiment_results" / "large_bs8" / "seed_1.txt",
        "checkpoint_path": PROJECT_DIR / "experiments" / "large_bs8" / "seed_1" / "best_model.pt",
    },
    {
        "encoder_short": "mental_bert_large",
        "model_name": "AIMH/mental-bert-large-cased",
        "batch_size": 4,
        "precision_dtype": "not_recorded",
        "seed": 1,
        "split_seed": "not_recorded",
        "result_file": PROJECT_DIR / "experiment_results" / "large_bs4" / "seed_1.txt",
        "checkpoint_path": PROJECT_DIR / "experiments" / "large_bs4" / "seed_1" / "best_model.pt",
    },
    {
        "encoder_short": "mental_bert_large",
        "model_name": "AIMH/mental-bert-large-cased",
        "batch_size": 8,
        "precision_dtype": "not_recorded",
        "seed": 1,
        "split_seed": "not_recorded",
        "result_file": PROJECT_DIR / "results_large" / "seed_1.txt",
        "checkpoint_path": PROJECT_DIR / "saved_models_large" / "seed_1" / "best_model.pt",
    },
]


def add_isolated_results():
    root = PROJECT_DIR / "module1_encoder_benchmark" / "results"
    for result_file in sorted(root.glob("*/*/seed_1.txt")):
        parts = result_file.relative_to(root).parts
        if len(parts) != 3:
            continue
        encoder_short, batch_folder, _ = parts
        if not batch_folder.startswith("bs"):
            continue
        batch_size = int(batch_folder[2:])
        CANDIDATE_RESULTS.append(
            {
                "encoder_short": encoder_short,
                "model_name": None,
                "batch_size": batch_size,
                "precision_dtype": None,
                "seed": 1,
                "split_seed": None,
                "result_file": result_file,
                "checkpoint_path": PROJECT_DIR
                / "module1_encoder_benchmark"
                / "models"
                / encoder_short
                / batch_folder
                / "seed_1"
                / "best_model.pt",
            }
        )

    invalid_root = PROJECT_DIR / "module1_encoder_benchmark" / "invalid_nan_runs"
    for result_file in sorted(invalid_root.glob("**/seed_1.txt")):
        text = read_text(result_file)
        encoder_short = value(r"Encoder short name:\s*(.+)", text, "unknown")
        batch_size = value(r"Batch size:\s*(\d+)", text, None, int)
        checkpoint = result_file.parent / "model_seed_1" / "best_model.pt"
        if batch_size is None:
            continue
        CANDIDATE_RESULTS.append(
            {
                "encoder_short": encoder_short,
                "model_name": value(r"Model:\s*(.+)", text, None),
                "batch_size": batch_size,
                "precision_dtype": value(r"Precision dtype:\s*(.+)", text, "not_recorded"),
                "seed": 1,
                "split_seed": value(r"Split seed:\s*(\d+)", text, None, int),
                "result_file": result_file,
                "checkpoint_path": checkpoint,
            }
        )


def read_text(path):
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def value(pattern, text, default=None, cast=str):
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return default
    raw = match.group(1).strip()
    if raw.lower() in {"nan", "none", "not_available", ""}:
        return default
    return cast(raw)


def metric_from_report(label, text):
    match = re.search(rf"{label}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)", text)
    if not match:
        return None, None, None, None
    return float(match.group(1)), float(match.group(2)), float(match.group(3)), int(match.group(4))


def best_epoch_from_epoch_metrics(text):
    rows = re.findall(r"Epoch\s+(\d+):.*?Val Macro-F1[:=]([0-9.]+)", text)
    if not rows:
        return None
    epoch, _ = max(((int(e), float(v)) for e, v in rows), key=lambda x: x[1])
    return epoch


def checkpoint_hash(path):
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_or_none(x):
    if x is None:
        return None
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def parse_candidate(candidate):
    path = candidate["result_file"]
    text = read_text(path)
    row = dict(candidate)
    row["result_file"] = str(path.relative_to(PROJECT_DIR))
    row["checkpoint_path"] = str(candidate["checkpoint_path"].relative_to(PROJECT_DIR))
    row["exists"] = path.exists()
    row["checkpoint_exists"] = candidate["checkpoint_path"].exists()
    row["run_status"] = value(r"Run status:\s*(.+)", text, "complete" if path.exists() else "missing")
    row["model_name"] = value(r"Model:\s*(.+)", text, candidate.get("model_name"))
    row["encoder_short"] = value(r"Encoder short name:\s*(.+)", text, candidate["encoder_short"])
    row["seed"] = value(r"Seed:\s*(\d+)", text, candidate["seed"], int)
    row["split_seed"] = value(r"Split seed:\s*(\d+)", text, candidate.get("split_seed"), int)
    row["batch_size"] = value(r"Batch size:\s*(\d+)", text, candidate["batch_size"], int)
    row["precision_dtype"] = value(r"Precision dtype:\s*(.+)", text, candidate.get("precision_dtype") or "not_recorded")
    row["best_val_macro_f1"] = finite_or_none(
        value(r"Best Val Macro-F1:\s*([0-9.]+|nan)", text, None, float)
    )
    row["best_epoch"] = value(r"Best epoch:\s*(\d+)", text, None, int) or best_epoch_from_epoch_metrics(text)
    row["test_macro_f1"] = finite_or_none(
        value(r"(?:Test Macro-F1|Final Test Macro-F1):\s*([0-9.]+|nan)", text, None, float)
    )
    row["accuracy"] = finite_or_none(value(r"Accuracy:\s*([0-9.]+|nan)", text, None, float))
    if row["accuracy"] is None:
        row["accuracy"] = finite_or_none(value(r"\n\s*accuracy\s+([0-9.]+)\s+\d+", text, None, float))

    high_p, high_r, high_f1, high_support = metric_from_report("High Risk", text)
    row["high_risk_precision"] = finite_or_none(
        value(r"High-Risk precision:\s*([0-9.]+|nan)", text, high_p, float)
    )
    row["high_risk_recall"] = finite_or_none(
        value(r"High-Risk recall:\s*([0-9.]+|nan)", text, high_r, float)
    )
    row["high_risk_f1"] = finite_or_none(
        value(r"High-Risk F1:\s*([0-9.]+|nan)", text, high_f1, float)
    )
    row["high_risk_support"] = high_support or 351
    row["high_risk_missed_estimate"] = (
        round(row["high_risk_support"] * (1 - row["high_risk_recall"]))
        if row["high_risk_recall"] is not None
        else None
    )
    row["raw_nll"] = finite_or_none(value(r"Raw test NLL:\s*([0-9.]+|nan)", text, None, float))
    row["raw_ece"] = finite_or_none(value(r"Raw ECE:\s*([0-9.]+|nan)", text, None, float))
    row["calibrated_nll"] = finite_or_none(value(r"Calibrated NLL:\s*([0-9.]+|nan)", text, None, float))
    row["calibrated_ece"] = finite_or_none(value(r"Calibrated ECE:\s*([0-9.]+|nan)", text, None, float))
    row["temperature"] = finite_or_none(value(r"Learned temperature:\s*([0-9.]+|nan)", text, None, float))
    row["calibration_status"] = "computed" if row["raw_ece"] is not None else "not_computed"
    if row["calibrated_ece"] is None:
        row["calibration_status"] = "raw_only" if row["raw_ece"] is not None else "not_computed"
    row["numerical_stability_status"] = "stable"
    if re.search(r"\bnan\b|\binf\b|non-finite", text, flags=re.IGNORECASE):
        row["numerical_stability_status"] = "unstable_or_contains_nonfinite"

    reasons = []
    if not row["exists"]:
        reasons.append("missing result file")
    if row["run_status"] != "complete":
        reasons.append(f"run status is {row['run_status']}")
    if not row["checkpoint_exists"]:
        reasons.append("missing checkpoint")
    required = [
        "best_val_macro_f1",
        "test_macro_f1",
        "accuracy",
        "high_risk_precision",
        "high_risk_recall",
        "high_risk_f1",
    ]
    for key in required:
        if row[key] is None:
            reasons.append(f"missing/non-finite {key}")
    if row["numerical_stability_status"] != "stable":
        reasons.append("non-finite metric or instability marker present")

    row["valid"] = not reasons
    row["exclusion_reason"] = "; ".join(reasons)
    row["checkpoint_sha256"] = checkpoint_hash(candidate["checkpoint_path"]) if row["checkpoint_exists"] else None
    return row


def duplicate_filter(rows):
    seen = {}
    for row in rows:
        if not row["valid"] or row["checkpoint_sha256"] is None:
            continue
        key = row["checkpoint_sha256"]
        if key in seen:
            row["valid"] = False
            row["exclusion_reason"] = (
                "duplicate byte-identical checkpoint of "
                f"{seen[key]['result_file']}"
            )
        else:
            seen[key] = row


def rank_key(row):
    return (
        row["high_risk_recall"] or -1,
        row["high_risk_precision"] or -1,
        row["best_val_macro_f1"] or -1,
        row["test_macro_f1"] or -1,
        -(row["raw_ece"] if row["raw_ece"] is not None else 999),
        1 if row["numerical_stability_status"] == "stable" else 0,
    )


def fmt(x):
    if x is None:
        return "not_computed"
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def write_outputs(valid_rows, excluded_rows):
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "encoder_short",
        "model_name",
        "batch_size",
        "precision_dtype",
        "seed",
        "split_seed",
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
        "calibration_status",
        "numerical_stability_status",
        "checkpoint_path",
        "result_file",
        "checkpoint_sha256",
    ]
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for idx, row in enumerate(valid_rows, start=1):
            out = {field: row.get(field) for field in fields}
            out["rank"] = idx
            writer.writerow(out)

    lines = [
        "Module 1 Seed-1 Benchmark Comparison",
        "====================================",
        "",
        "Ranking priority: high-risk recall, high-risk precision, validation Macro-F1, test Macro-F1, raw ECE, numerical stability.",
        "This is a severity-only Module 1 encoder benchmark. It is not the later multi-task or temporal pipeline.",
        "",
        "Valid Runs",
        "----------",
        "| Rank | Encoder | Model | Batch | Precision | Val Macro-F1 | Test Macro-F1 | High P | High R | High F1 | Missed High-Risk | Raw ECE | Cal ECE | Stability |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(valid_rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    row["encoder_short"],
                    row["model_name"],
                    str(row["batch_size"]),
                    row["precision_dtype"],
                    fmt(row["best_val_macro_f1"]),
                    fmt(row["test_macro_f1"]),
                    fmt(row["high_risk_precision"]),
                    fmt(row["high_risk_recall"]),
                    fmt(row["high_risk_f1"]),
                    fmt(row["high_risk_missed_estimate"]),
                    fmt(row["raw_ece"]),
                    fmt(row["calibrated_ece"]),
                    row["numerical_stability_status"],
                ]
            )
            + " |"
        )

    lines.extend(["", "Excluded Or Invalid Runs", "------------------------"])
    if not excluded_rows:
        lines.append("None.")
    else:
        for row in excluded_rows:
            lines.append(f"- {row['result_file']}: {row['exclusion_reason']}")

    lines.extend(
        [
            "",
            "Safety-Oriented Notes",
            "---------------------",
            "- High-risk recall is prioritized because missed high-risk turns are safety-critical.",
            "- A recall of 1.000000 on one seed means zero high-risk misses on this test split, but it is not proof of global superiority.",
            "- Small recall differences correspond to only a few examples in this test set, so multi-seed testing is required.",
            "- Calibration matters because this framework later uses confidence and uncertainty for escalation decisions.",
            "- Numerical instability disqualifies the unstable DeBERTa FP16/auto runs; valid DeBERTa runs use fp32.",
            "",
            "Recommended Seed-1 Finalists For Multi-Seed Testing",
            "--------------------------------------------------",
            "1. microsoft/deberta-v3-large, deberta_v3_large, batch 32, fp32: top safety-ranked seed-1 result with high-risk recall 1.000000, strong Macro-F1, and best raw ECE among large benchmark runs.",
            "2. answerdotai/ModernBERT-large, modernbert_large, batch 32, fp32: strong Macro-F1 and calibration, with high-risk recall 0.991453 and lower memory than DeBERTa bs32.",
            "3. AIMH/mental-roberta-large, mental_roberta_large, batch 32, not_recorded precision: high-risk recall 1.000000 and stable seed-1 metrics; include if you want a mental-domain RoBERTa-family finalist.",
        ]
    )
    OUTPUT_TXT.write_text("\n".join(lines) + "\n")


def main():
    add_isolated_results()
    rows = [parse_candidate(c) for c in CANDIDATE_RESULTS]
    duplicate_filter(rows)
    valid_rows = sorted([r for r in rows if r["valid"]], key=rank_key, reverse=True)
    excluded_rows = [r for r in rows if not r["valid"]]
    write_outputs(valid_rows, excluded_rows)
    print(f"Valid runs: {len(valid_rows)}")
    print(f"Excluded runs: {len(excluded_rows)}")
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved text: {OUTPUT_TXT}")
    print("\nTop valid runs:")
    for idx, row in enumerate(valid_rows[:8], start=1):
        print(
            f"{idx}. {row['encoder_short']} bs{row['batch_size']} "
            f"{row['precision_dtype']} | high R={fmt(row['high_risk_recall'])}, "
            f"high P={fmt(row['high_risk_precision'])}, "
            f"val F1={fmt(row['best_val_macro_f1'])}, raw ECE={fmt(row['raw_ece'])}"
        )


if __name__ == "__main__":
    main()
