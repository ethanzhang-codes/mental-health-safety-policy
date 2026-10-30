import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


MODEL_NAME = "microsoft/deberta-v3-large"
TRAIN_CSV = "cradle_dialogue/data/cradle_train_targets.csv"
VAL_CSV = "cradle_dialogue/data/cradle_validation_targets.csv"
TEST_CSV = "cradle_dialogue/data/cradle_test_targets.csv"

TASKS = {
    "alert": {
        "target_col": "target_alert",
        "mask_col": "alert_target_mask",
        "model_dir": "cradle_dialogue/models/single_task_alert/seed_1",
        "result_txt": "cradle_dialogue/results/single_task_alert/seed_1.txt",
        "prediction_csv": "cradle_dialogue/predictions/single_task_alert/seed_1_test_predictions.csv",
        "multitask_val_auroc": 0.582165,
        "multitask_val_auprc": 0.081005,
        "multitask_val_f1": 0.0,
        "multitask_test_auroc": 0.543185,
        "multitask_test_auprc": 0.056562,
        "multitask_test_f1": 0.0,
    },
    "confirm": {
        "target_col": "target_confirm",
        "mask_col": "confirm_target_mask",
        "model_dir": "cradle_dialogue/models/single_task_confirm/seed_1",
        "result_txt": "cradle_dialogue/results/single_task_confirm/seed_1.txt",
        "prediction_csv": "cradle_dialogue/predictions/single_task_confirm/seed_1_test_predictions.csv",
        "multitask_val_auroc": 0.498660,
        "multitask_val_auprc": 0.079472,
        "multitask_val_f1": 0.0,
        "multitask_test_auroc": 0.453007,
        "multitask_test_auprc": 0.090166,
        "multitask_test_f1": 0.0,
    },
}

FORBIDDEN_MODEL_INPUT_COLUMNS = {
    "labels_raw", "label_count", "has_any_crisis_label", "has_alert", "has_confirm",
    "event_types", "category_count", "has_confirmed_category", "has_ongoing_label",
    "has_past_label", "temporal_states", "first_alert_turn_id", "first_confirm_turn_id",
    "first_crisis_labeled_turn_id", "last_crisis_labeled_turn_id",
    "first_confirm_CA_turn_id", "first_confirm_DV_turn_id", "first_confirm_RA_turn_id",
    "first_confirm_SHA_turn_id", "first_confirm_SH_turn_id",
    "first_confirm_SI_active_turn_id", "first_confirm_SI_passive_turn_id",
    "turns_until_first_alert", "turns_until_first_confirm", "turns_since_first_alert",
    "turns_since_first_confirm", "is_pre_alert", "is_alert_turn",
    "is_between_alert_and_confirm", "is_confirm_turn", "is_post_confirm",
    "progression_stage", "target_alert", "target_confirm", "target_category_CA",
    "target_category_DV", "target_category_RA", "target_category_SHA",
    "target_category_SH", "target_category_SI_active", "target_category_SI_passive",
    "target_ongoing", "target_past", "user_turn_index",
    "user_turns_until_first_confirm", "confirm_within_1_user_turn",
    "confirm_within_2_user_turns", "confirm_within_3_user_turns",
    "confirm_within_5_user_turns",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train CRADLE single-task Alert or Confirm baseline.")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_csv", default=TRAIN_CSV)
    parser.add_argument("--val_csv", default=VAL_CSV)
    parser.add_argument("--test_csv", default=TEST_CSV)
    parser.add_argument("--export_all_splits_only", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_targets(path):
    return pd.read_csv(path, keep_default_na=False)


def model_text(row):
    text = str(row["text"])
    role = str(row["speaker_role"]).lower()
    if role == "user":
        return f"User: {text}"
    if role == "listener":
        return f"Listener: {text}"
    return f"Unknown: {text}"


class SingleTaskDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, target_col, mask_col):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_col = target_col
        self.mask_col = mask_col
        self.texts = [model_text(row) for _, row in self.df.iterrows()]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target": torch.tensor([float(row[self.target_col])], dtype=torch.float32),
            "mask": torch.tensor([float(row[self.mask_col])], dtype=torch.float32),
        }


class SingleTaskModel(nn.Module):
    def __init__(self, model_name, dropout=0.3):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self.encoder.float()
        self.hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("Non-finite encoder hidden states detected.")
        logits = self.classifier(hidden[:, 0, :])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits detected.")
        return logits


def masked_loss(logits, targets, masks, criterion):
    unreduced = criterion(logits, targets)
    selected = unreduced[masks > 0]
    if selected.numel() == 0:
        return logits.sum() * 0.0
    loss = selected.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite masked loss detected.")
    return loss


def pos_weight_from_train(df, target_col, mask_col):
    masked = df[df[mask_col].astype(float) == 1]
    pos = int(masked[target_col].astype(float).sum())
    neg = int(len(masked) - pos)
    if pos == 0:
        raise ValueError(f"No positive examples for {target_col}")
    return pos, neg, neg / pos


def metric_or_none(fn, y_true, probs):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(fn(y_true, probs))
    except Exception:
        return None


def distribution(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def format_distribution(name, stats):
    if stats["count"] == 0:
        return f"{name}: count=0"
    return (
        f"{name}: count={stats['count']}, min={stats['min']:.6f}, p05={stats['p05']:.6f}, "
        f"p25={stats['p25']:.6f}, median={stats['median']:.6f}, mean={stats['mean']:.6f}, "
        f"p75={stats['p75']:.6f}, p95={stats['p95']:.6f}, max={stats['max']:.6f}"
    )


def compute_metrics(targets, probs, masks, threshold):
    masks = masks.astype(bool)
    y_true = targets[masks].astype(int)
    y_prob = probs[masks].astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "positive_support": int(support[0]),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "auroc": metric_or_none(roc_auc_score, y_true, y_prob),
        "auprc": metric_or_none(average_precision_score, y_true, y_prob),
        "frac_ge_0_05": float(np.mean(y_prob >= 0.05)),
        "frac_ge_0_1": float(np.mean(y_prob >= 0.1)),
        "frac_ge_0_3": float(np.mean(y_prob >= 0.3)),
        "frac_ge_0_5": float(np.mean(y_prob >= 0.5)),
        "positive_distribution": distribution(y_prob[y_true == 1]),
        "negative_distribution": distribution(y_prob[y_true == 0]),
        "positive_prob_mean": float(np.mean(y_prob[y_true == 1])) if np.any(y_true == 1) else None,
        "negative_prob_mean": float(np.mean(y_prob[y_true == 0])) if np.any(y_true == 0) else None,
        "positive_prob_median": float(np.median(y_prob[y_true == 1])) if np.any(y_true == 1) else None,
        "negative_prob_median": float(np.median(y_prob[y_true == 0])) if np.any(y_true == 0) else None,
    }


def tune_threshold(targets, probs, masks):
    masks = masks.astype(bool)
    y_true = targets[masks].astype(int)
    y_prob = probs[masks].astype(float)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


@torch.no_grad()
def collect_predictions(model, loader, device, criterion=None):
    model.eval()
    probs, targets, masks = [], [], []
    total_loss = 0.0
    steps = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        if criterion is not None:
            loss = masked_loss(logits, batch["target"], batch["mask"], criterion)
            total_loss += float(loss.detach().cpu())
            steps += 1
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(batch["target"].cpu().numpy())
        masks.append(batch["mask"].cpu().numpy())
    return {
        "probs": np.concatenate(probs, axis=0).reshape(-1),
        "targets": np.concatenate(targets, axis=0).reshape(-1),
        "masks": np.concatenate(masks, axis=0).reshape(-1),
        "loss": total_loss / max(steps, 1) if criterion is not None else None,
    }


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = masked_loss(logits, batch["target"], batch["mask"], criterion)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(loader), 1), float(scheduler.get_last_lr()[0])


def summarize_metrics(label, metrics):
    return (
        f"{label}: precision={metrics['precision']:.6f}, recall={metrics['recall']:.6f}, "
        f"f1={metrics['f1']:.6f}, auroc={metrics['auroc']}, auprc={metrics['auprc']}, "
        f"positive_support={metrics['positive_support']}, fp={metrics['false_positives']}, "
        f"fn={metrics['false_negatives']}, threshold={metrics['threshold']:.6f}, "
        f"frac>=0.05={metrics['frac_ge_0_05']:.6f}, frac>=0.1={metrics['frac_ge_0_1']:.6f}, "
        f"frac>=0.3={metrics['frac_ge_0_3']:.6f}, frac>=0.5={metrics['frac_ge_0_5']:.6f}"
    )


def write_predictions(df, preds, threshold, output_path, task_name):
    out = df[["dataset_split", "dialogue_id", "turn_id", "speaker_role", "text"]].copy()
    out[f"target_{task_name}"] = preds["targets"].astype(int)
    out[f"{task_name}_mask"] = preds["masks"].astype(int)
    out[f"{task_name}_prob"] = preds["probs"]
    out[f"{task_name}_pred_fixed_0_5"] = (preds["probs"] >= 0.5).astype(int)
    out[f"{task_name}_pred_tuned"] = (preds["probs"] >= threshold).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def prediction_path_for_split(base_path, split_name):
    name = base_path.name
    if "_test_predictions" in name:
        return base_path.with_name(name.replace("_test_predictions", f"_{split_name}_predictions"))
    return base_path.with_name(f"{base_path.stem}_{split_name}{base_path.suffix}")


def export_all_splits(args, cfg):
    model_dir = Path(cfg["model_dir"])
    result_txt = Path(cfg["result_txt"])
    prediction_csv = Path(cfg["prediction_csv"])
    if not (model_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_dir / 'best_model.pt'}")

    set_seed(args.seed)
    train_df = read_targets(args.train_csv)
    val_df = read_targets(args.val_csv)
    test_df = read_targets(args.test_csv)
    target_col = cfg["target_col"]
    mask_col = cfg["mask_col"]

    tokenizer_source = model_dir if (model_dir / "tokenizer.json").exists() else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleTaskModel(args.model_name, dropout=args.dropout).to(device)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    tuned_threshold = checkpoint.get("selected_validation_threshold")
    if tuned_threshold is None and result_txt.exists():
        for line in result_txt.read_text().splitlines():
            if line.startswith("selected_validation_threshold:"):
                tuned_threshold = float(line.split(":", 1)[1].strip())
                break
    if tuned_threshold is None:
        tuned_threshold = 0.5

    for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        ds = SingleTaskDataset(df, tokenizer, args.max_length, target_col, mask_col)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        preds = collect_predictions(model, loader, device, criterion=None)
        out_path = prediction_path_for_split(prediction_csv, split_name)
        write_predictions(df, preds, tuned_threshold, out_path, args.task)
        print(f"Saved {args.task} {split_name} predictions to {out_path}", flush=True)


def parse_single_task_result(path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        if line.startswith("best_epoch:"):
            values["best_epoch"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("best_validation_auprc:"):
            values["best_validation_auprc"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("selected_validation_threshold:"):
            values["threshold"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Validation fixed 0.5:"):
            values["val_fixed_line"] = line
        elif line.startswith("Clinician test fixed 0.5:"):
            values["test_fixed_line"] = line
        elif line.startswith("Validation tuned:"):
            values["val_tuned_line"] = line
        elif line.startswith("Clinician test tuned:"):
            values["test_tuned_line"] = line
        elif line.startswith("training_duration_seconds:"):
            values["duration"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("numerical_stability:"):
            values["stability"] = line.split(":", 1)[1].strip()
    return values


def metric_from_line(line, key):
    if not line:
        return None
    import re
    match = re.search(rf"{key}=([^,]+)", line)
    if not match:
        return None
    value = match.group(1)
    return None if value == "None" else float(value)


def maybe_write_comparison():
    alert_path = Path(TASKS["alert"]["result_txt"])
    confirm_path = Path(TASKS["confirm"]["result_txt"])
    if not alert_path.exists() or not confirm_path.exists():
        return
    out_path = Path("cradle_dialogue/results/single_task_vs_multitask_comparison.txt")
    rows = []
    summary_answers = []
    for task in ["alert", "confirm"]:
        cfg = TASKS[task]
        values = parse_single_task_result(Path(cfg["result_txt"]))
        val_line = values.get("val_fixed_line")
        test_line = values.get("test_fixed_line")
        val_auroc = metric_from_line(val_line, "auroc")
        val_auprc = metric_from_line(val_line, "auprc")
        val_f1 = metric_from_line(val_line, "f1")
        test_auroc = metric_from_line(test_line, "auroc")
        test_auprc = metric_from_line(test_line, "auprc")
        test_f1 = metric_from_line(test_line, "f1")
        rows.extend([
            task.upper(),
            f"Multi-task validation AUROC: {cfg['multitask_val_auroc']:.6f}",
            f"Single-task validation AUROC: {val_auroc:.6f}",
            f"Validation AUROC change: {val_auroc - cfg['multitask_val_auroc']:.6f}",
            f"Multi-task validation AUPRC: {cfg['multitask_val_auprc']:.6f}",
            f"Single-task validation AUPRC: {val_auprc:.6f}",
            f"Validation AUPRC change: {val_auprc - cfg['multitask_val_auprc']:.6f}",
            f"Multi-task clinician test AUROC: {cfg['multitask_test_auroc']:.6f}",
            f"Single-task clinician test AUROC: {test_auroc:.6f}",
            f"Clinician test AUROC change: {test_auroc - cfg['multitask_test_auroc']:.6f}",
            f"Multi-task clinician test AUPRC: {cfg['multitask_test_auprc']:.6f}",
            f"Single-task clinician test AUPRC: {test_auprc:.6f}",
            f"Clinician test AUPRC change: {test_auprc - cfg['multitask_test_auprc']:.6f}",
            f"Multi-task fixed F1: {cfg['multitask_test_f1']:.6f}",
            f"Single-task fixed F1: {test_f1:.6f}",
            f"Best epoch: {values.get('best_epoch')}",
            f"Selected validation threshold: {values.get('threshold')}",
            f"Numerical stability: {values.get('stability')}",
            f"Training duration seconds: {values.get('duration')}",
            "",
        ])
        summary_answers.append((task, val_auprc, test_auprc, test_auroc, test_f1))
    alert = summary_answers[0]
    confirm = summary_answers[1]
    alert_change = alert[2] - TASKS["alert"]["multitask_test_auprc"]
    confirm_change = confirm[2] - TASKS["confirm"]["multitask_test_auprc"]
    if alert_change > 0.03 and confirm_change > 0.03:
        interpretation = "CASE A: both single-task models substantially improved AUPRC, suggesting multi-task interference/optimization was a major problem."
        next_step = "Investigate multi-task optimization before claiming single-turn limitation."
    elif alert_change <= 0.03 and confirm_change <= 0.03:
        interpretation = "CASE B: both single-task models remained near multi-task performance, suggesting weak performance is not mainly task competition."
        next_step = "Proceed to temporal-context modeling as the next implementation step."
    else:
        interpretation = "CASE C: one task improved more than the other, so Alert and Confirm should be interpreted separately."
        next_step = "Use the task-specific result to decide whether temporal context is needed first for Alert or Confirm."
    rows.extend([
        "Interpretation",
        interpretation,
        "",
        "Final Beginner-Friendly Summary",
        f"1. Did Alert improve when trained alone? Alert clinician-test AUPRC change = {alert_change:.6f}.",
        f"2. Did Confirm improve when trained alone? Confirm clinician-test AUPRC change = {confirm_change:.6f}.",
        f"3. How much did Alert clinician-test AUPRC change? {alert_change:.6f}.",
        f"4. How much did Confirm clinician-test AUPRC change? {confirm_change:.6f}.",
        "5. Are positive and negative probabilities now separated? See each task report's probability distribution section.",
        "6. Is the 0.5 threshold problem still present? See fixed-threshold F1 and fraction>=0.5 in each report.",
        f"7. Was multi-task interference a major problem? {interpretation}",
        "8. Is Alert harder than Confirm from current-turn text? Compare the two single-task clinician-test AUPRC/AUROC values above.",
        f"9. Do these results now support the need for temporal context? {interpretation}",
        f"10. What is the single next implementation step? {next_step}",
    ])
    out_path.write_text("\n".join(rows) + "\n")


def main():
    args = parse_args()
    cfg = TASKS[args.task]
    if args.export_all_splits_only:
        export_all_splits(args, cfg)
        return
    model_dir = Path(cfg["model_dir"])
    result_txt = Path(cfg["result_txt"])
    prediction_csv = Path(cfg["prediction_csv"])
    if model_dir.exists() or result_txt.exists() or prediction_csv.exists():
        raise FileExistsError(f"Refusing to overwrite existing {args.task} outputs.")

    set_seed(args.seed)
    start = time.time()
    train_df = read_targets(args.train_csv)
    val_df = read_targets(args.val_csv)
    test_df = read_targets(args.test_csv)
    target_col = cfg["target_col"]
    mask_col = cfg["mask_col"]
    pos_count, neg_count, pos_weight_value = pos_weight_from_train(train_df, target_col, mask_col)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleTaskModel(args.model_name, dropout=args.dropout).to(device)
    if args.gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    train_ds = SingleTaskDataset(train_df, tokenizer, args.max_length, target_col, mask_col)
    val_ds = SingleTaskDataset(val_df, tokenizer, args.max_length, target_col, mask_col)
    test_ds = SingleTaskDataset(test_df, tokenizer, args.max_length, target_col, mask_col)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    model_feature_keys = {"input_ids", "attention_mask"}
    forbidden_in_inputs = sorted(FORBIDDEN_MODEL_INPUT_COLUMNS.intersection(model_feature_keys))

    report = [
        f"CRADLE Single-Task {args.task.title()} Baseline",
        "=" * (31 + len(args.task)),
        "",
        "No multi-task checkpoint was used. No severity checkpoint was used.",
        f"model_name: {args.model_name}",
        f"seed: {args.seed}",
        f"batch_size: {args.batch_size}",
        "precision_dtype: fp32",
        f"epochs: {args.epochs}",
        f"max_length: {args.max_length}",
        f"dropout: {args.dropout}",
        f"learning_rate: {args.learning_rate}",
        f"weight_decay: {args.weight_decay}",
        f"gradient_checkpointing: {args.gradient_checkpointing}",
        f"target_col: {target_col}",
        f"mask_col: {mask_col}",
        f"train_rows: {len(train_df)}",
        f"validation_rows: {len(val_df)}",
        f"test_rows: {len(test_df)}",
        f"masked_train_rows: {int(train_df[mask_col].astype(float).sum())}",
        f"masked_validation_rows: {int(val_df[mask_col].astype(float).sum())}",
        f"masked_test_rows: {int(test_df[mask_col].astype(float).sum())}",
        f"train_positive_count: {pos_count}",
        f"train_negative_count: {neg_count}",
        f"computed_pos_weight: {pos_weight_value}",
        f"encoder_hidden_size: {model.hidden_size}",
        f"total_parameters: {total_params}",
        f"trainable_parameters: {trainable_params}",
        f"model_input_keys: {sorted(model_feature_keys)}",
        f"forbidden_future_columns_passed_to_model: {forbidden_in_inputs if forbidden_in_inputs else 'none'}",
        "",
    ]

    smoke = next(iter(train_loader))
    smoke = {k: v.to(device) for k, v in smoke.items()}
    smoke_logits = model(smoke["input_ids"], smoke["attention_mask"])
    smoke_loss = masked_loss(smoke_logits, smoke["target"], smoke["mask"], criterion)
    report.extend([
        f"smoke_logit_shape: {tuple(smoke_logits.shape)}",
        f"smoke_loss_finite: {bool(torch.isfinite(smoke_loss).item())}",
        f"microsoft_deberta_v3_large_initialization_verified: {args.model_name == MODEL_NAME}",
        "",
    ])
    print("\n".join(report[-8:]), flush=True)
    if forbidden_in_inputs or not torch.isfinite(smoke_loss):
        raise RuntimeError("Smoke test failed.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_auprc = -math.inf
    best_epoch = -1
    epoch_rows = []
    model_dir.mkdir(parents=True, exist_ok=False)

    for epoch in range(1, args.epochs + 1):
        train_loss, lr = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_preds = collect_predictions(model, val_loader, device, criterion)
        val_metrics = compute_metrics(val_preds["targets"], val_preds["probs"], val_preds["masks"], 0.5)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_preds["loss"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_auroc": val_metrics["auroc"],
            "validation_auprc": val_metrics["auprc"],
            "validation_fraction_ge_0_5": val_metrics["frac_ge_0_5"],
            "validation_positive_prob_mean": val_metrics["positive_prob_mean"],
            "validation_negative_prob_mean": val_metrics["negative_prob_mean"],
            "validation_positive_prob_median": val_metrics["positive_prob_median"],
            "validation_negative_prob_median": val_metrics["negative_prob_median"],
            "learning_rate": lr,
        }
        epoch_rows.append(row)
        print(
            f"{args.task} epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} | "
            f"val_auprc={val_metrics['auprc']} | val_auroc={val_metrics['auroc']} | "
            f"val_f1={val_metrics['f1']:.6f}",
            flush=True,
        )
        score = -math.inf if val_metrics["auprc"] is None else val_metrics["auprc"]
        if score > best_auprc:
            best_auprc = score
            best_epoch = epoch
            torch.save(
                {
                    "model_name": args.model_name,
                    "task": args.task,
                    "target_col": target_col,
                    "mask_col": mask_col,
                    "hidden_size": model.hidden_size,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                    "precision_dtype": "fp32",
                    "epoch": epoch,
                    "best_validation_auprc": best_auprc,
                    "state_dict": model.state_dict(),
                    "optimizer_config": {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay},
                },
                model_dir / "best_model.pt",
            )
            tokenizer.save_pretrained(model_dir)

    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    val_preds = collect_predictions(model, val_loader, device, criterion)
    tuned_threshold, tuned_val_f1 = tune_threshold(val_preds["targets"], val_preds["probs"], val_preds["masks"])
    checkpoint["selected_validation_threshold"] = tuned_threshold
    checkpoint["selected_validation_f1"] = tuned_val_f1
    torch.save(checkpoint, model_dir / "best_model.pt")

    val_fixed = compute_metrics(val_preds["targets"], val_preds["probs"], val_preds["masks"], 0.5)
    val_tuned = compute_metrics(val_preds["targets"], val_preds["probs"], val_preds["masks"], tuned_threshold)
    test_preds = collect_predictions(model, test_loader, device, criterion)
    test_fixed = compute_metrics(test_preds["targets"], test_preds["probs"], test_preds["masks"], 0.5)
    test_tuned = compute_metrics(test_preds["targets"], test_preds["probs"], test_preds["masks"], tuned_threshold)
    write_predictions(test_df, test_preds, tuned_threshold, prediction_csv, args.task)

    duration = time.time() - start
    report.extend([
        "Epoch Metrics",
        "-------------",
    ])
    for row in epoch_rows:
        report.append(json.dumps(row, sort_keys=True))
    report.extend([
        "",
        f"best_epoch: {best_epoch}",
        f"best_validation_auprc: {best_auprc}",
        f"selected_validation_threshold: {tuned_threshold}",
        f"selected_validation_f1: {tuned_val_f1}",
        "",
        summarize_metrics("Validation fixed 0.5", val_fixed),
        summarize_metrics("Validation tuned", val_tuned),
        summarize_metrics("Clinician test fixed 0.5", test_fixed),
        summarize_metrics("Clinician test tuned", test_tuned),
        "",
        "Validation fixed probability distributions:",
        format_distribution("target=1", val_fixed["positive_distribution"]),
        format_distribution("target=0", val_fixed["negative_distribution"]),
        "",
        "Clinician test fixed probability distributions:",
        format_distribution("target=1", test_fixed["positive_distribution"]),
        format_distribution("target=0", test_fixed["negative_distribution"]),
        "",
        "Direct comparison to multi-task baseline:",
        f"multitask_validation_auroc: {cfg['multitask_val_auroc']}",
        f"single_task_validation_auroc: {val_fixed['auroc']}",
        f"validation_auroc_change: {None if val_fixed['auroc'] is None else val_fixed['auroc'] - cfg['multitask_val_auroc']}",
        f"multitask_validation_auprc: {cfg['multitask_val_auprc']}",
        f"single_task_validation_auprc: {val_fixed['auprc']}",
        f"validation_auprc_change: {None if val_fixed['auprc'] is None else val_fixed['auprc'] - cfg['multitask_val_auprc']}",
        f"multitask_clinician_test_auroc: {cfg['multitask_test_auroc']}",
        f"single_task_clinician_test_auroc: {test_fixed['auroc']}",
        f"clinician_test_auroc_change: {None if test_fixed['auroc'] is None else test_fixed['auroc'] - cfg['multitask_test_auroc']}",
        f"multitask_clinician_test_auprc: {cfg['multitask_test_auprc']}",
        f"single_task_clinician_test_auprc: {test_fixed['auprc']}",
        f"clinician_test_auprc_change: {None if test_fixed['auprc'] is None else test_fixed['auprc'] - cfg['multitask_test_auprc']}",
        "",
        "probability_separation_behavior: compare target=1 and target=0 distributions above.",
        "numerical_stability: stable; no non-finite hidden states, logits, loss, or gradient norm detected.",
        f"training_duration_seconds: {duration:.2f}",
        f"checkpoint_path: {model_dir / 'best_model.pt'}",
        f"prediction_file_path: {prediction_csv}",
        "run_status: complete",
    ])

    result_txt.parent.mkdir(parents=True, exist_ok=True)
    prediction_csv.parent.mkdir(parents=True, exist_ok=True)
    result_txt.write_text("\n".join(report) + "\n")
    print(f"Saved {args.task} report to {result_txt}", flush=True)
    print(f"Saved {args.task} predictions to {prediction_csv}", flush=True)
    maybe_write_comparison()


if __name__ == "__main__":
    main()
