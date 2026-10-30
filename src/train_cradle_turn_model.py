import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from cradle_turn_model import CradleTurnMultiTaskModel


CATEGORY_ORDER = ["CA", "DV", "RA", "SHA", "SH", "SI_active", "SI_passive"]
TEMPORAL_ORDER = ["ongoing", "past"]
MODEL_NAME = "microsoft/deberta-v3-large"
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
    parser = argparse.ArgumentParser(description="Train CRADLE current-turn multi-task baseline.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_csv", default="cradle_dialogue/data/cradle_train_targets.csv")
    parser.add_argument("--val_csv", default="cradle_dialogue/data/cradle_validation_targets.csv")
    parser.add_argument("--test_csv", default="cradle_dialogue/data/cradle_test_targets.csv")
    parser.add_argument("--model_dir", default="cradle_dialogue/models/turn_model/seed_1")
    parser.add_argument("--result_txt", default="cradle_dialogue/results/turn_model/seed_1.txt")
    parser.add_argument("--prediction_csv", default="cradle_dialogue/predictions/turn_model/seed_1_test_predictions.csv")
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


class CradleTurnDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
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
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_alert": torch.tensor([float(row["target_alert"])], dtype=torch.float32),
            "alert_mask": torch.tensor([float(row["alert_target_mask"])], dtype=torch.float32),
            "target_confirm": torch.tensor([float(row["target_confirm"])], dtype=torch.float32),
            "confirm_mask": torch.tensor([float(row["confirm_target_mask"])], dtype=torch.float32),
            "target_category": torch.tensor(
                [float(row[f"target_category_{name}"]) for name in CATEGORY_ORDER],
                dtype=torch.float32,
            ),
            "category_mask": torch.tensor([float(row["category_target_mask"])], dtype=torch.float32),
            "target_temporal": torch.tensor(
                [float(row["target_ongoing"]), float(row["target_past"])],
                dtype=torch.float32,
            ),
            "temporal_mask": torch.tensor([float(row["temporal_state_target_mask"])], dtype=torch.float32),
        }
        return item


def compute_pos_weight(df, target_cols, mask_col):
    masked = df[df[mask_col].astype(float) == 1]
    weights = []
    for col in target_cols:
        pos = float(masked[col].astype(float).sum())
        neg = float(len(masked) - pos)
        weights.append(1.0 if pos <= 0 else neg / pos)
    return torch.tensor(weights, dtype=torch.float32)


def masked_bce_loss(logits, targets, mask, criterion):
    per_item = criterion(logits, targets)
    if logits.shape[1] > 1:
        mask = mask.expand_as(per_item)
    selected = per_item[mask > 0]
    if selected.numel() == 0:
        return logits.sum() * 0.0
    return selected.mean()


def compute_loss(logits, batch, criteria):
    alert_loss = masked_bce_loss(logits["alert"], batch["target_alert"], batch["alert_mask"], criteria["alert"])
    confirm_loss = masked_bce_loss(logits["confirm"], batch["target_confirm"], batch["confirm_mask"], criteria["confirm"])
    category_loss = masked_bce_loss(logits["category"], batch["target_category"], batch["category_mask"], criteria["category"])
    temporal_loss = masked_bce_loss(logits["temporal"], batch["target_temporal"], batch["temporal_mask"], criteria["temporal"])
    total = alert_loss + confirm_loss + category_loss + temporal_loss
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite total loss detected.")
    return total, {
        "alert_loss": float(alert_loss.detach().cpu()),
        "confirm_loss": float(confirm_loss.detach().cpu()),
        "category_loss": float(category_loss.detach().cpu()),
        "temporal_loss": float(temporal_loss.detach().cpu()),
    }


def maybe_auc(y_true, scores, kind):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, scores) if kind == "roc" else average_precision_score(y_true, scores))
    except Exception:
        return None


def binary_metrics(y_true, probs, threshold):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    pred = (probs >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=[1], average=None, zero_division=0
    )
    macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "macro_f1": float(macro_f1),
        "support": int(support[0]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "auroc": maybe_auc(y_true, probs, "roc"),
        "auprc": maybe_auc(y_true, probs, "pr"),
    }


def multilabel_metrics(y_true, probs, thresholds, names):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    thresholds = np.asarray(thresholds).reshape(1, -1)
    pred = (probs >= thresholds).astype(int)
    per_label = {}
    for i, name in enumerate(names):
        per_label[name] = binary_metrics(y_true[:, i], probs[:, i], float(thresholds[0, i]))
    return {
        "micro_precision": float(precision_recall_fscore_support(y_true, pred, average="micro", zero_division=0)[0]),
        "micro_recall": float(precision_recall_fscore_support(y_true, pred, average="micro", zero_division=0)[1]),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)[0]),
        "macro_recall": float(precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)[1]),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, pred, average="samples", zero_division=0)),
        "exact_match_accuracy": float((y_true == pred).all(axis=1).mean()),
        "hamming_loss": float(hamming_loss(y_true, pred)),
        "per_label": per_label,
    }


def tune_binary_threshold(y_true, probs):
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (np.asarray(probs) >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def tune_multilabel_thresholds(y_true, probs):
    return [tune_binary_threshold(y_true[:, i], probs[:, i]) for i in range(y_true.shape[1])]


@torch.no_grad()
def collect_predictions(model, loader, device, criteria=None):
    model.eval()
    all_probs = {"alert": [], "confirm": [], "category": [], "temporal": []}
    all_targets = {"alert": [], "confirm": [], "category": [], "temporal": []}
    all_masks = {"alert": [], "confirm": [], "category": [], "temporal": []}
    total_loss = 0.0
    steps = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        if criteria is not None:
            loss, _ = compute_loss(logits, batch, criteria)
            total_loss += float(loss.detach().cpu())
            steps += 1
        all_probs["alert"].append(torch.sigmoid(logits["alert"]).cpu().numpy())
        all_probs["confirm"].append(torch.sigmoid(logits["confirm"]).cpu().numpy())
        all_probs["category"].append(torch.sigmoid(logits["category"]).cpu().numpy())
        all_probs["temporal"].append(torch.sigmoid(logits["temporal"]).cpu().numpy())
        all_targets["alert"].append(batch["target_alert"].cpu().numpy())
        all_targets["confirm"].append(batch["target_confirm"].cpu().numpy())
        all_targets["category"].append(batch["target_category"].cpu().numpy())
        all_targets["temporal"].append(batch["target_temporal"].cpu().numpy())
        all_masks["alert"].append(batch["alert_mask"].cpu().numpy())
        all_masks["confirm"].append(batch["confirm_mask"].cpu().numpy())
        all_masks["category"].append(batch["category_mask"].cpu().numpy())
        all_masks["temporal"].append(batch["temporal_mask"].cpu().numpy())
    out = {}
    for key in all_probs:
        out[f"{key}_probs"] = np.concatenate(all_probs[key], axis=0)
        out[f"{key}_targets"] = np.concatenate(all_targets[key], axis=0)
        out[f"{key}_mask"] = np.concatenate(all_masks[key], axis=0).reshape(-1).astype(bool)
    out["loss"] = total_loss / max(steps, 1) if criteria is not None else None
    return out


def evaluate_predictions(preds, thresholds):
    alert_mask = preds["alert_mask"]
    confirm_mask = preds["confirm_mask"]
    category_mask = preds["category_mask"]
    temporal_mask = preds["temporal_mask"]
    metrics = {
        "alert": binary_metrics(preds["alert_targets"][alert_mask, 0], preds["alert_probs"][alert_mask, 0], thresholds["alert"]),
        "confirm": binary_metrics(preds["confirm_targets"][confirm_mask, 0], preds["confirm_probs"][confirm_mask, 0], thresholds["confirm"]),
        "category": multilabel_metrics(
            preds["category_targets"][category_mask],
            preds["category_probs"][category_mask],
            thresholds["category"],
            CATEGORY_ORDER,
        ),
        "temporal": multilabel_metrics(
            preds["temporal_targets"][temporal_mask],
            preds["temporal_probs"][temporal_mask],
            thresholds["temporal"],
            TEMPORAL_ORDER,
        ),
    }
    metrics["composite"] = float(np.mean([
        metrics["alert"]["macro_f1"],
        metrics["confirm"]["macro_f1"],
        metrics["category"]["macro_f1"],
        metrics["temporal"]["macro_f1"],
    ]))
    return metrics


def train_epoch(model, loader, optimizer, scheduler, device, criteria):
    model.train()
    total = 0.0
    parts_total = {"alert_loss": 0.0, "confirm_loss": 0.0, "category_loss": 0.0, "temporal_loss": 0.0}
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss, parts = compute_loss(logits, batch, criteria)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at batch {step}: {grad_norm}")
        optimizer.step()
        scheduler.step()
        total += float(loss.detach().cpu())
        for key, value in parts.items():
            parts_total[key] += value
    n = max(len(loader), 1)
    return total / n, {k: v / n for k, v in parts_total.items()}


def format_value(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def summarize_binary(name, metrics):
    return (
        f"{name}: precision={format_value(metrics['precision'])}, recall={format_value(metrics['recall'])}, "
        f"f1={format_value(metrics['f1'])}, macro_f1={format_value(metrics['macro_f1'])}, "
        f"support={metrics['support']}, fp={metrics['fp']}, fn={metrics['fn']}, "
        f"auroc={format_value(metrics['auroc'])}, auprc={format_value(metrics['auprc'])}, "
        f"threshold={format_value(metrics['threshold'])}"
    )


def save_predictions(df, preds, thresholds, out_path):
    out = df[["dataset_split", "dialogue_id", "turn_id", "speaker_role", "text"]].copy()
    for task in ["alert", "confirm"]:
        probs = preds[f"{task}_probs"][:, 0]
        out[f"target_{task}"] = preds[f"{task}_targets"][:, 0].astype(int)
        out[f"{task}_prob"] = probs
        out[f"{task}_pred_fixed_0_5"] = (probs >= 0.5).astype(int)
        out[f"{task}_pred_tuned"] = (probs >= thresholds[task]).astype(int)
    for i, name in enumerate(CATEGORY_ORDER):
        probs = preds["category_probs"][:, i]
        out[f"target_category_{name}"] = preds["category_targets"][:, i].astype(int)
        out[f"category_{name}_prob"] = probs
        out[f"category_{name}_pred_fixed_0_5"] = (probs >= 0.5).astype(int)
        out[f"category_{name}_pred_tuned"] = (probs >= thresholds["category"][i]).astype(int)
    for i, name in enumerate(TEMPORAL_ORDER):
        probs = preds["temporal_probs"][:, i]
        out[f"target_{name}"] = preds["temporal_targets"][:, i].astype(int)
        out[f"{name}_prob"] = probs
        out[f"{name}_pred_fixed_0_5"] = (probs >= 0.5).astype(int)
        out[f"{name}_pred_tuned"] = (probs >= thresholds["temporal"][i]).astype(int)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)


def write_report(path, report):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(report) + "\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    start = time.time()
    model_dir = Path(args.model_dir)
    result_txt = Path(args.result_txt)
    prediction_csv = Path(args.prediction_csv)
    if model_dir.exists() or result_txt.exists():
        raise FileExistsError(f"Refusing to overwrite existing CRADLE turn-model output: {model_dir} / {result_txt}")

    train_df = read_targets(args.train_csv)
    val_df = read_targets(args.val_csv)
    test_df = read_targets(args.test_csv)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CradleTurnMultiTaskModel(args.model_name, dropout=args.dropout).to(device)
    if args.gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()

    alert_pos_weight = compute_pos_weight(train_df, ["target_alert"], "alert_target_mask").to(device)
    confirm_pos_weight = compute_pos_weight(train_df, ["target_confirm"], "confirm_target_mask").to(device)
    category_pos_weight = compute_pos_weight(train_df, [f"target_category_{x}" for x in CATEGORY_ORDER], "category_target_mask").to(device)
    temporal_pos_weight = compute_pos_weight(train_df, ["target_ongoing", "target_past"], "temporal_state_target_mask").to(device)
    criteria = {
        "alert": torch.nn.BCEWithLogitsLoss(pos_weight=alert_pos_weight, reduction="none"),
        "confirm": torch.nn.BCEWithLogitsLoss(pos_weight=confirm_pos_weight, reduction="none"),
        "category": torch.nn.BCEWithLogitsLoss(pos_weight=category_pos_weight, reduction="none"),
        "temporal": torch.nn.BCEWithLogitsLoss(pos_weight=temporal_pos_weight, reduction="none"),
    }

    train_ds = CradleTurnDataset(train_df, tokenizer, args.max_length)
    val_ds = CradleTurnDataset(val_df, tokenizer, args.max_length)
    test_ds = CradleTurnDataset(test_df, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    report = [
        "CRADLE-Dialogue Turn-Level Multi-Task Baseline",
        "================================================",
        "",
        "Provenance: train/validation are synthetic/structured conversational data; test is the clinician-annotated benchmark.",
        "No hierarchical transformer was trained. This is a current-turn baseline only.",
        "Encoder initialization: original Hugging Face pretrained encoder, not the severity-trained checkpoint.",
        "",
        f"model_name: {args.model_name}",
        f"seed: {args.seed}",
        f"batch_size: {args.batch_size}",
        "precision_dtype: fp32",
        f"learning_rate: {args.learning_rate}",
        f"weight_decay: {args.weight_decay}",
        f"dropout: {args.dropout}",
        f"epochs: {args.epochs}",
        f"max_length: {args.max_length}",
        f"gradient_checkpointing: {args.gradient_checkpointing}",
        f"device: {device}",
        f"encoder_hidden_size: {model.hidden_size}",
        f"category_order: {CATEGORY_ORDER}",
        "",
        f"train_rows: {len(train_df)}; validation_rows: {len(val_df)}; test_rows: {len(test_df)}",
        f"train_user_mask_count: {int(train_df['alert_target_mask'].astype(float).sum())}",
        f"validation_user_mask_count: {int(val_df['alert_target_mask'].astype(float).sum())}",
        f"test_user_mask_count: {int(test_df['alert_target_mask'].astype(float).sum())}",
        f"train_temporal_state_mask_count: {int(train_df['temporal_state_target_mask'].astype(float).sum())}",
        f"validation_temporal_state_mask_count: {int(val_df['temporal_state_target_mask'].astype(float).sum())}",
        f"test_temporal_state_mask_count: {int(test_df['temporal_state_target_mask'].astype(float).sum())}",
        "",
        f"alert_pos_weight: {alert_pos_weight.detach().cpu().tolist()}",
        f"confirm_pos_weight: {confirm_pos_weight.detach().cpu().tolist()}",
        f"category_pos_weight: {dict(zip(CATEGORY_ORDER, category_pos_weight.detach().cpu().tolist()))}",
        f"temporal_pos_weight: {dict(zip(TEMPORAL_ORDER, temporal_pos_weight.detach().cpu().tolist()))}",
        "",
    ]

    smoke_batch = next(iter(train_loader))
    model_feature_keys = {"input_ids", "attention_mask"}
    forbidden_in_batch = sorted(FORBIDDEN_MODEL_INPUT_COLUMNS.intersection(model_feature_keys))
    report.append(f"model_input_keys: {sorted(model_feature_keys)}")
    report.append(f"forbidden_future_columns_passed_to_model: {forbidden_in_batch if forbidden_in_batch else 'none'}")
    smoke = {k: v.to(device) for k, v in smoke_batch.items()}
    smoke_logits = model(smoke["input_ids"], smoke["attention_mask"])
    smoke_loss, _ = compute_loss(smoke_logits, smoke, criteria)
    report.extend([
        f"smoke_alert_shape: {tuple(smoke_logits['alert'].shape)}",
        f"smoke_confirm_shape: {tuple(smoke_logits['confirm'].shape)}",
        f"smoke_category_shape: {tuple(smoke_logits['category'].shape)}",
        f"smoke_temporal_shape: {tuple(smoke_logits['temporal'].shape)}",
        f"smoke_loss_finite: {bool(torch.isfinite(smoke_loss).item())}",
        "",
    ])
    print("\n".join(report[-8:]), flush=True)
    if forbidden_in_batch or not torch.isfinite(smoke_loss):
        raise RuntimeError("Smoke test failed.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    fixed_thresholds = {"alert": 0.5, "confirm": 0.5, "category": [0.5] * 7, "temporal": [0.5] * 2}
    best_score = -math.inf
    best_epoch = -1
    epoch_rows = []
    model_dir.mkdir(parents=True, exist_ok=False)

    for epoch in range(1, args.epochs + 1):
        train_loss, parts = train_epoch(model, train_loader, optimizer, scheduler, device, criteria)
        val_preds = collect_predictions(model, val_loader, device, criteria)
        val_metrics = evaluate_predictions(val_preds, fixed_thresholds)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_preds["loss"],
            "val_composite": val_metrics["composite"],
            "val_alert_macro_f1": val_metrics["alert"]["macro_f1"],
            "val_confirm_macro_f1": val_metrics["confirm"]["macro_f1"],
            "val_category_macro_f1": val_metrics["category"]["macro_f1"],
            "val_temporal_macro_f1": val_metrics["temporal"]["macro_f1"],
            **parts,
        }
        epoch_rows.append(row)
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} | "
            f"val_composite={val_metrics['composite']:.6f}",
            flush=True,
        )
        if val_metrics["composite"] > best_score:
            best_score = val_metrics["composite"]
            best_epoch = epoch
            torch.save(
                {
                    "model_name": args.model_name,
                    "hidden_size": model.hidden_size,
                    "category_order": CATEGORY_ORDER,
                    "temporal_order": TEMPORAL_ORDER,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                    "precision_dtype": "fp32",
                    "epoch": epoch,
                    "best_composite": best_score,
                    "state_dict": model.state_dict(),
                    "optimizer_config": {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay},
                },
                model_dir / "best_model.pt",
            )
            tokenizer.save_pretrained(model_dir)

    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    val_preds = collect_predictions(model, val_loader, device, criteria)
    tuned_thresholds = {
        "alert": tune_binary_threshold(val_preds["alert_targets"][val_preds["alert_mask"], 0], val_preds["alert_probs"][val_preds["alert_mask"], 0]),
        "confirm": tune_binary_threshold(val_preds["confirm_targets"][val_preds["confirm_mask"], 0], val_preds["confirm_probs"][val_preds["confirm_mask"], 0]),
        "category": tune_multilabel_thresholds(val_preds["category_targets"][val_preds["category_mask"]], val_preds["category_probs"][val_preds["category_mask"]]),
        "temporal": tune_multilabel_thresholds(val_preds["temporal_targets"][val_preds["temporal_mask"]], val_preds["temporal_probs"][val_preds["temporal_mask"]]),
    }
    checkpoint["thresholds"] = tuned_thresholds
    torch.save(checkpoint, model_dir / "best_model.pt")

    val_fixed = evaluate_predictions(val_preds, fixed_thresholds)
    val_tuned = evaluate_predictions(val_preds, tuned_thresholds)
    test_preds = collect_predictions(model, test_loader, device, criteria)
    test_fixed = evaluate_predictions(test_preds, fixed_thresholds)
    test_tuned = evaluate_predictions(test_preds, tuned_thresholds)
    save_predictions(test_df, test_preds, tuned_thresholds, prediction_csv)

    duration = time.time() - start
    report.extend(["Epoch metrics:"])
    for row in epoch_rows:
        report.append(json.dumps(row, sort_keys=True))
    report.extend([
        "",
        f"best_epoch: {best_epoch}",
        f"best_validation_composite: {best_score:.6f}",
        f"tuned_thresholds: {json.dumps(tuned_thresholds, sort_keys=True)}",
        "",
        "Validation metrics, fixed 0.5 thresholds:",
        summarize_binary("alert", val_fixed["alert"]),
        summarize_binary("confirm", val_fixed["confirm"]),
        f"category_macro_f1={val_fixed['category']['macro_f1']:.6f}, category_micro_f1={val_fixed['category']['micro_f1']:.6f}, category_samples_f1={val_fixed['category']['samples_f1']:.6f}, category_exact_match={val_fixed['category']['exact_match_accuracy']:.6f}, category_hamming_loss={val_fixed['category']['hamming_loss']:.6f}",
        f"temporal_macro_f1={val_fixed['temporal']['macro_f1']:.6f}, temporal_micro_f1={val_fixed['temporal']['micro_f1']:.6f}",
        "",
        "Clinician test benchmark metrics, fixed 0.5 thresholds:",
        summarize_binary("alert", test_fixed["alert"]),
        summarize_binary("confirm", test_fixed["confirm"]),
        f"category_macro_f1={test_fixed['category']['macro_f1']:.6f}, category_micro_f1={test_fixed['category']['micro_f1']:.6f}, category_samples_f1={test_fixed['category']['samples_f1']:.6f}, category_exact_match={test_fixed['category']['exact_match_accuracy']:.6f}, category_hamming_loss={test_fixed['category']['hamming_loss']:.6f}",
        f"temporal_macro_f1={test_fixed['temporal']['macro_f1']:.6f}, temporal_micro_f1={test_fixed['temporal']['micro_f1']:.6f}",
        "",
        "Clinician test benchmark metrics, validation-tuned thresholds:",
        summarize_binary("alert", test_tuned["alert"]),
        summarize_binary("confirm", test_tuned["confirm"]),
        f"category_macro_f1={test_tuned['category']['macro_f1']:.6f}, category_micro_f1={test_tuned['category']['micro_f1']:.6f}, category_samples_f1={test_tuned['category']['samples_f1']:.6f}, category_exact_match={test_tuned['category']['exact_match_accuracy']:.6f}, category_hamming_loss={test_tuned['category']['hamming_loss']:.6f}",
        f"temporal_macro_f1={test_tuned['temporal']['macro_f1']:.6f}, temporal_micro_f1={test_tuned['temporal']['micro_f1']:.6f}",
        "",
        "Per-category clinician test metrics, validation-tuned thresholds:",
    ])
    for name, metric in test_tuned["category"]["per_label"].items():
        report.append(summarize_binary(f"category_{name}", metric))
    report.extend(["", "Per-temporal-state clinician test metrics, validation-tuned thresholds:"])
    for name, metric in test_tuned["temporal"]["per_label"].items():
        report.append(summarize_binary(f"temporal_{name}", metric))

    report.extend([
        "",
        "Generalization gaps, fixed 0.5 thresholds (test - validation):",
        f"alert_f1_gap={test_fixed['alert']['f1'] - val_fixed['alert']['f1']:.6f}",
        f"confirm_f1_gap={test_fixed['confirm']['f1'] - val_fixed['confirm']['f1']:.6f}",
        f"category_macro_f1_gap={test_fixed['category']['macro_f1'] - val_fixed['category']['macro_f1']:.6f}",
        f"temporal_macro_f1_gap={test_fixed['temporal']['macro_f1'] - val_fixed['temporal']['macro_f1']:.6f}",
        "",
        "Generalization gaps, validation-tuned thresholds (test - validation):",
        f"alert_f1_gap={test_tuned['alert']['f1'] - val_tuned['alert']['f1']:.6f}",
        f"confirm_f1_gap={test_tuned['confirm']['f1'] - val_tuned['confirm']['f1']:.6f}",
        f"category_macro_f1_gap={test_tuned['category']['macro_f1'] - val_tuned['category']['macro_f1']:.6f}",
        f"temporal_macro_f1_gap={test_tuned['temporal']['macro_f1'] - val_tuned['temporal']['macro_f1']:.6f}",
        "",
        "Listener anomaly predictions on clinician test split:",
    ])
    anomaly = test_df[(test_df["speaker_role"] != "user") & ((test_df["target_alert"].astype(float) == 1) | (test_df["target_confirm"].astype(float) == 1))]
    pred_df = pd.read_csv(prediction_csv)
    for idx in anomaly.index:
        row = pred_df.loc[idx]
        report.append(
            f"dialogue_id={row['dialogue_id']}, turn_id={row['turn_id']}, speaker_role={row['speaker_role']}, "
            f"alert_prob={row['alert_prob']:.6f}, alert_pred_tuned={row['alert_pred_tuned']}, "
            f"confirm_prob={row['confirm_prob']:.6f}, confirm_pred_tuned={row['confirm_pred_tuned']}"
        )
    if anomaly.empty:
        report.append("none")
    report.extend([
        "",
        "Numerical stability: stable; no non-finite hidden states, logits, losses, or gradient norms detected.",
        f"checkpoint_path: {model_dir / 'best_model.pt'}",
        f"prediction_file_path: {prediction_csv}",
        f"training_duration_seconds: {duration:.2f}",
        "run_status: complete",
    ])
    write_report(result_txt, report)
    print(f"Saved CRADLE turn-model report to {result_txt}", flush=True)
    print(f"Saved CRADLE turn-model predictions to {prediction_csv}", flush=True)


if __name__ == "__main__":
    main()
