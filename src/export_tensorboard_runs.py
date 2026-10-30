import csv
import re
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


PROJECT_DIR = Path(__file__).resolve().parent
TB_ROOT = PROJECT_DIR / "tensorboard_logs"


def parse_float(pattern, text):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse_int(pattern, text):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def parse_epoch_metrics(text):
    return [
        {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_macro_f1": float(train_f1),
            "val_loss": float(val_loss),
            "val_macro_f1": float(val_f1),
        }
        for epoch, train_loss, train_f1, val_loss, val_f1 in re.findall(
            r"Epoch\s+(\d+):\s+Train Loss=([0-9.]+),\s+Train Macro-F1=([0-9.]+),\s+Val Loss=([0-9.]+),\s+Val Macro-F1=([0-9.]+)",
            text,
        )
    ]


def parse_class_metrics(text, label):
    match = re.search(
        rf"{re.escape(label)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)",
        text,
    )
    if not match:
        return None
    return {
        "precision": float(match.group(1)),
        "recall": float(match.group(2)),
        "f1": float(match.group(3)),
        "support": int(match.group(4)),
    }


def export_standard_result(path):
    text = path.read_text()
    log_dir = TB_ROOT / "historical" / str(path.relative_to(PROJECT_DIR).with_suffix("")).replace("/", "__")
    writer = SummaryWriter(log_dir=str(log_dir))

    model_name = re.search(r"Model:\s+(.+)", text)
    seed = parse_int(r"Seed:\s+(\d+)", text)
    batch_size = parse_int(r"Batch size:\s+(\d+)", text)
    if model_name:
        writer.add_text("run/model_name", model_name.group(1).strip())
    writer.add_text("run/source_path", str(path.relative_to(PROJECT_DIR)))
    if seed is not None:
        writer.add_scalar("run/seed", seed, 0)
    if batch_size is not None:
        writer.add_scalar("run/batch_size", batch_size, 0)

    epoch_metrics = parse_epoch_metrics(text)
    for row in epoch_metrics:
        step = row["epoch"]
        writer.add_scalar("loss/train", row["train_loss"], step)
        writer.add_scalar("loss/val", row["val_loss"], step)
        writer.add_scalar("macro_f1/train", row["train_macro_f1"], step)
        writer.add_scalar("macro_f1/val", row["val_macro_f1"], step)

    final_step = max([row["epoch"] for row in epoch_metrics] or [0]) + 1
    best_val = parse_float(r"Best Val Macro-F1:\s+([0-9.]+)", text)
    test_f1 = parse_float(r"(?:Final Test Macro-F1|Test Macro-F1):\s+([0-9.]+)", text)
    accuracy = parse_float(r"accuracy\s+([0-9.]+)\s+\d+", text)
    if best_val is not None:
        writer.add_scalar("macro_f1/best_val", best_val, final_step)
    if test_f1 is not None:
        writer.add_scalar("macro_f1/test", test_f1, final_step)
    if accuracy is not None:
        writer.add_scalar("accuracy/test", accuracy, final_step)

    for label, tag in (
        ("Low Risk", "low_risk"),
        ("Medium Risk", "medium_risk"),
        ("High Risk", "high_risk"),
        ("High-Risk", "high_risk"),
    ):
        metrics = parse_class_metrics(text, label)
        if metrics:
            writer.add_scalar(f"class_f1/{tag}", metrics["f1"], final_step)
            writer.add_scalar(f"class_precision/{tag}", metrics["precision"], final_step)
            writer.add_scalar(f"class_recall/{tag}", metrics["recall"], final_step)

    writer.flush()
    writer.close()


def export_ablation_csv(path):
    with path.open(newline="") as handle:
        for step, row in enumerate(csv.DictReader(handle), start=1):
            slug = row["system"].lower().replace(" ", "_").replace("+", "plus").replace("/", "_")
            writer = SummaryWriter(log_dir=str(TB_ROOT / "framework_ablation" / slug))
            writer.add_text("run/system", row["system"])
            writer.add_text("run/source_path", str(path.relative_to(PROJECT_DIR)))
            for key in (
                "unsafe_continuation_rate",
                "over_escalation_burden",
                "autonomy_coverage_rate",
                "average_time_to_escalation",
                "median_time_to_escalation",
            ):
                if row.get(key):
                    writer.add_scalar(key, float(row[key]), step)
            for key in (
                "level_0_count",
                "level_1_count",
                "level_2_count",
                "level_3_count",
                "level_4_count",
            ):
                if row.get(key):
                    writer.add_scalar(f"escalation/{key}", float(row[key]), step)
            writer.flush()
            writer.close()


def main():
    result_paths = []
    result_paths.extend(sorted((PROJECT_DIR / "results").glob("seed_*.txt")))
    result_paths.extend(sorted((PROJECT_DIR / "results_large").glob("seed_*.txt")))
    result_paths.extend(sorted((PROJECT_DIR / "experiment_results").glob("*/seed_*.txt")))
    result_paths.extend(sorted((PROJECT_DIR / "module1_encoder_benchmark" / "results").glob("**/seed_*.txt")))
    single_run = PROJECT_DIR / "results_module1_seed1.txt"
    if single_run.exists():
        result_paths.append(single_run)

    for path in result_paths:
        export_standard_result(path)

    ablation_path = PROJECT_DIR / "results" / "ablation_comparison_annomi.csv"
    if ablation_path.exists():
        export_ablation_csv(ablation_path)

    print(f"Wrote TensorBoard event files under {TB_ROOT}")


if __name__ == "__main__":
    main()
