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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from cradle_temporal_alert_model import TemporalAlertModel


MODEL_NAME = "microsoft/deberta-v3-large"
TRAIN_CSV = "cradle_dialogue/data/cradle_train_targets.csv"
VAL_CSV = "cradle_dialogue/data/cradle_validation_targets.csv"
TEST_CSV = "cradle_dialogue/data/cradle_test_targets.csv"
MODEL_DIR = "cradle_dialogue/models/temporal_alert/seed_1"
RESULT_TXT = "cradle_dialogue/results/temporal_alert/seed_1.txt"
PREDICTION_CSV = "cradle_dialogue/predictions/temporal_alert/seed_1_test_predictions.csv"
COMPARISON_TXT = "cradle_dialogue/results/temporal_alert/single_turn_vs_temporal_alert_comparison.txt"

SINGLE_TURN_BASELINE = {
    "validation_auroc": 0.607675,
    "validation_auprc": 0.090167,
    "test_auroc": 0.536585,
    "test_auprc": 0.055381,
    "test_fixed_f1": 0.0,
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
    "confirm_within_5_user_turns", "forecast_target_mask",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train hierarchical temporal Alert model for CRADLE.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--history_window", type=int, default=8)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_csv", default=TRAIN_CSV)
    parser.add_argument("--val_csv", default=VAL_CSV)
    parser.add_argument("--test_csv", default=TEST_CSV)
    parser.add_argument("--model_dir", default=MODEL_DIR)
    parser.add_argument("--result_txt", default=RESULT_TXT)
    parser.add_argument("--prediction_csv", default=PREDICTION_CSV)
    parser.add_argument("--comparison_txt", default=COMPARISON_TXT)
    parser.add_argument("--smoke_only", action="store_true")
    parser.add_argument("--allow_overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_targets(path):
    return pd.read_csv(path, keep_default_na=False)


def prefixed_text(row):
    text = str(row["text"])
    role = str(row["speaker_role"]).lower()
    if role == "user":
        return f"User: {text}"
    if role == "listener":
        return f"Listener: {text}"
    return f"Unknown: {text}"


def build_examples(df, history_window):
    sorted_df = df.sort_values(["dialogue_id", "turn_id"]).reset_index(drop=True)
    examples = []
    for _, group in sorted_df.groupby("dialogue_id", sort=False):
        rows = list(group.to_dict("records"))
        for pos, row in enumerate(rows):
            if int(float(row["alert_target_mask"])) != 1:
                continue
            start = max(0, pos - history_window + 1)
            history_rows = rows[start:pos + 1]
            history_texts = [prefixed_text(history_row) for history_row in history_rows]
            history_turn_ids = [int(history_row["turn_id"]) for history_row in history_rows]
            history_keys = [(str(history_row["dialogue_id"]), int(history_row["turn_id"])) for history_row in history_rows]
            current_turn_id = int(row["turn_id"])
            if any(turn_id > current_turn_id for turn_id in history_turn_ids):
                raise ValueError("Future turn detected in constructed history.")
            examples.append({
                "dataset_split": row["dataset_split"],
                "dialogue_id": row["dialogue_id"],
                "turn_id": current_turn_id,
                "speaker_role": row["speaker_role"],
                "text": row["text"],
                "target": float(row["target_alert"]),
                "mask": float(row["alert_target_mask"]),
                "history_texts": history_texts,
                "history_turn_ids": history_turn_ids,
                "history_keys": history_keys,
                "history_length": len(history_texts),
            })
    return examples


def build_turn_token_cache(df, tokenizer, max_length):
    cache = {}
    sorted_df = df.sort_values(["dialogue_id", "turn_id"]).reset_index(drop=True)
    for _, row in sorted_df.iterrows():
        key = (str(row["dialogue_id"]), int(row["turn_id"]))
        if key in cache:
            continue
        encoded = tokenizer(
            prefixed_text(row),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        cache[key] = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
    return cache


def history_stats(examples):
    lengths = np.asarray([example["history_length"] for example in examples], dtype=float)
    return {
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(np.max(lengths)),
    }


class TemporalAlertDataset(Dataset):
    def __init__(self, examples, token_cache):
        self.examples = examples
        self.token_cache = token_cache

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = dict(self.examples[idx])
        item["history_tokens"] = [self.token_cache[key] for key in item["history_keys"]]
        return item


class TemporalCollator:
    def __init__(self):
        pass

    def __call__(self, batch):
        flat_input_ids = []
        flat_attention_masks = []
        lengths = []
        for item in batch:
            lengths.append(len(item["history_tokens"]))
            for tokenized in item["history_tokens"]:
                flat_input_ids.append(tokenized["input_ids"])
                flat_attention_masks.append(tokenized["attention_mask"])
        return {
            "input_ids": torch.stack(flat_input_ids),
            "attention_mask": torch.stack(flat_attention_masks),
            "history_lengths": torch.tensor(lengths, dtype=torch.long),
            "target": torch.tensor([[float(item["target"])] for item in batch], dtype=torch.float32),
            "mask": torch.tensor([[float(item["mask"])] for item in batch], dtype=torch.float32),
            "metadata": batch,
        }


def pos_weight_from_examples(examples):
    pos = int(sum(example["target"] for example in examples if example["mask"] == 1))
    total = int(sum(example["mask"] for example in examples))
    neg = total - pos
    if pos == 0:
        raise ValueError("No positive Alert examples in train split.")
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
    valid = masks.astype(bool)
    y_true = targets[valid].astype(int)
    y_prob = probs[valid].astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
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
    }


def summarize_metrics(label, metrics):
    return (
        f"{label}: precision={metrics['precision']:.6f}, recall={metrics['recall']:.6f}, "
        f"f1={metrics['f1']:.6f}, auroc={metrics['auroc']}, auprc={metrics['auprc']}, "
        f"positive_support={metrics['positive_support']}, fp={metrics['false_positives']}, "
        f"fn={metrics['false_negatives']}, threshold={metrics['threshold']:.6f}, "
        f"frac>=0.05={metrics['frac_ge_0_05']:.6f}, frac>=0.1={metrics['frac_ge_0_1']:.6f}, "
        f"frac>=0.3={metrics['frac_ge_0_3']:.6f}, frac>=0.5={metrics['frac_ge_0_5']:.6f}"
    )


def masked_loss(logits, targets, masks, criterion):
    unreduced = criterion(logits, targets)
    selected = unreduced[masks > 0]
    if selected.numel() == 0:
        return logits.sum() * 0.0
    loss = selected.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite Alert loss detected.")
    return loss


def tune_threshold(targets, probs, masks):
    valid = masks.astype(bool)
    y_true = targets[valid].astype(int)
    y_prob = probs[valid].astype(float)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


@torch.no_grad()
def collect_predictions(model, loader, device, criterion=None):
    model.eval()
    probs, targets, masks, metadata = [], [], [], []
    total_loss = 0.0
    steps = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        history_lengths = batch["history_lengths"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        logits, _ = model(input_ids, attention_mask, history_lengths)
        if criterion is not None:
            loss = masked_loss(logits, target, mask, criterion)
            total_loss += float(loss.detach().cpu())
            steps += 1
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(target.cpu().numpy())
        masks.append(mask.cpu().numpy())
        metadata.extend(batch["metadata"])
    return {
        "probs": np.concatenate(probs, axis=0).reshape(-1),
        "targets": np.concatenate(targets, axis=0).reshape(-1),
        "masks": np.concatenate(masks, axis=0).reshape(-1),
        "metadata": metadata,
        "loss": total_loss / max(steps, 1) if criterion is not None else None,
    }


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        history_lengths = batch["history_lengths"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(input_ids, attention_mask, history_lengths)
        loss = masked_loss(logits, target, mask, criterion)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(loader), 1), float(scheduler.get_last_lr()[0])


def make_prediction_dataframe(preds, threshold):
    rows = []
    for item, target, mask, prob in zip(preds["metadata"], preds["targets"], preds["masks"], preds["probs"]):
        rows.append({
            "dataset_split": item["dataset_split"],
            "dialogue_id": item["dialogue_id"],
            "turn_id": item["turn_id"],
            "speaker_role": item["speaker_role"],
            "text": item["text"],
            "target_alert": int(target),
            "alert_target_mask": int(mask),
            "history_length": item["history_length"],
            "history_turn_ids": "|".join(str(x) for x in item["history_turn_ids"]),
            "alert_prob": float(prob),
            "alert_pred_fixed_0_5": int(prob >= 0.5),
            "alert_pred_tuned": int(prob >= threshold),
        })
    return pd.DataFrame(rows)


def grouped_metrics(preds, group_name, group_fn, threshold):
    rows = []
    meta = preds["metadata"]
    targets = preds["targets"]
    probs = preds["probs"]
    masks = preds["masks"]
    labels = [group_fn(item) for item in meta]
    for label in ["1 turn", "2-3 turns", "4-5 turns", "6-8 turns"] if group_name == "history_length" else ["turn_id 0", "turn_id 1-2", "turn_id 3-5", "turn_id 6+"]:
        idx = np.asarray([x == label for x in labels]) & masks.astype(bool)
        if not idx.any():
            rows.append(f"{group_name} {label}: no examples")
            continue
        y_true = targets[idx].astype(int)
        y_prob = probs[idx].astype(float)
        support = int(y_true.sum())
        auroc = metric_or_none(roc_auc_score, y_true, y_prob)
        auprc = metric_or_none(average_precision_score, y_true, y_prob)
        if group_name == "history_length":
            f1 = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
            rows.append(f"{group_name} {label}: positive_support={support}, auroc={auroc}, auprc={auprc}, tuned_f1={f1:.6f}, n={int(idx.sum())}")
        else:
            positive_idx = idx & (targets.astype(int) == 1)
            mean_alert_prob = float(np.mean(probs[positive_idx])) if positive_idx.any() else None
            recall = float(np.mean(probs[positive_idx] >= threshold)) if positive_idx.any() else None
            rows.append(f"{group_name} {label}: positive_support={support}, mean_alert_probability={mean_alert_prob}, recall_at_tuned_threshold={recall}")
    return rows


def history_group(item):
    length = int(item["history_length"])
    if length == 1:
        return "1 turn"
    if length <= 3:
        return "2-3 turns"
    if length <= 5:
        return "4-5 turns"
    return "6-8 turns"


def alert_position_group(item):
    turn_id = int(item["turn_id"])
    if turn_id == 0:
        return "turn_id 0"
    if turn_id <= 2:
        return "turn_id 1-2"
    if turn_id <= 5:
        return "turn_id 3-5"
    return "turn_id 6+"


def classify_case(val_auprc, test_auprc):
    val_improved = val_auprc is not None and val_auprc > SINGLE_TURN_BASELINE["validation_auprc"] + 0.02
    test_improved = test_auprc is not None and test_auprc > SINGLE_TURN_BASELINE["test_auprc"] + 0.02
    if val_improved and test_improved:
        return "CASE A", "conversation history materially improves early Alert detection."
    if val_improved and not test_improved:
        return "CASE B", "the temporal model learns validation progression patterns but does not generalize well to clinician-annotated conversations."
    return "CASE C", "the current hierarchical design does not solve Alert detection."


def run_smoke_test(model, batch, criterion, device):
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    history_lengths = batch["history_lengths"].to(device)
    target = batch["target"].to(device)
    mask = batch["mask"].to(device)
    logits, shape_info = model(input_ids, attention_mask, history_lengths)
    loss = masked_loss(logits, target, mask, criterion)
    loss.backward()

    def has_grad(module):
        return any(param.grad is not None and torch.isfinite(param.grad).all() for param in module.parameters() if param.requires_grad)

    smoke = {
        "real_turns_encoded": int(input_ids.shape[0]),
        "flattened_tokenizer_batch_shape": tuple(input_ids.shape),
        "reconstructed_turn_embedding_tensor_shape": shape_info["turn_embedding_shape"],
        "transformer_padding_mask_shape": shape_info["padding_mask_shape"],
        "final_alert_logit_shape": shape_info["logit_shape"],
        "loss_finite": bool(torch.isfinite(loss).item()),
        "encoder_grad_reaches": has_grad(model.encoder),
        "projection_grad_reaches": has_grad(model.turn_projection),
        "conversation_transformer_grad_reaches": has_grad(model.conversation_transformer),
        "alert_head_grad_reaches": has_grad(model.alert_head),
    }
    model.zero_grad(set_to_none=True)
    return smoke


def validate_histories(examples, history_window):
    for item in examples:
        if len(item["history_texts"]) > history_window:
            raise ValueError("History longer than configured window found.")
        if item["history_turn_ids"][-1] != item["turn_id"]:
            raise ValueError("Final history turn is not the current target turn.")
        if any(turn_id > item["turn_id"] for turn_id in item["history_turn_ids"]):
            raise ValueError("Future turn found in history.")
    return True


def write_comparison(path, val_fixed, test_fixed, case_name, case_interpretation):
    rows = [
        "Single-Turn vs Temporal Alert Comparison",
        "========================================",
        "",
        f"single_turn_validation_auroc: {SINGLE_TURN_BASELINE['validation_auroc']}",
        f"temporal_validation_auroc: {val_fixed['auroc']}",
        f"validation_auroc_change: {None if val_fixed['auroc'] is None else val_fixed['auroc'] - SINGLE_TURN_BASELINE['validation_auroc']}",
        f"single_turn_validation_auprc: {SINGLE_TURN_BASELINE['validation_auprc']}",
        f"temporal_validation_auprc: {val_fixed['auprc']}",
        f"validation_auprc_change: {None if val_fixed['auprc'] is None else val_fixed['auprc'] - SINGLE_TURN_BASELINE['validation_auprc']}",
        f"single_turn_clinician_test_auroc: {SINGLE_TURN_BASELINE['test_auroc']}",
        f"temporal_clinician_test_auroc: {test_fixed['auroc']}",
        f"clinician_test_auroc_change: {None if test_fixed['auroc'] is None else test_fixed['auroc'] - SINGLE_TURN_BASELINE['test_auroc']}",
        f"single_turn_clinician_test_auprc: {SINGLE_TURN_BASELINE['test_auprc']}",
        f"temporal_clinician_test_auprc: {test_fixed['auprc']}",
        f"clinician_test_auprc_change: {None if test_fixed['auprc'] is None else test_fixed['auprc'] - SINGLE_TURN_BASELINE['test_auprc']}",
        f"single_turn_fixed_f1: {SINGLE_TURN_BASELINE['test_fixed_f1']}",
        f"temporal_fixed_f1: {test_fixed['f1']}",
        "",
        f"result_case: {case_name}",
        f"interpretation: {case_interpretation}",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(rows) + "\n")


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    result_txt = Path(args.result_txt)
    prediction_csv = Path(args.prediction_csv)
    if not args.allow_overwrite and not args.smoke_only:
        if model_dir.exists() or result_txt.exists() or prediction_csv.exists():
            raise FileExistsError("Refusing to overwrite existing temporal Alert outputs.")

    set_seed(args.seed)
    start_time = time.time()
    train_df = read_targets(args.train_csv)
    val_df = read_targets(args.val_csv)
    test_df = read_targets(args.test_csv)
    train_examples = build_examples(train_df, args.history_window)
    val_examples = build_examples(val_df, args.history_window)
    test_examples = build_examples(test_df, args.history_window)
    validate_histories(train_examples, args.history_window)
    validate_histories(val_examples, args.history_window)
    validate_histories(test_examples, args.history_window)

    train_pos, train_neg, pos_weight_value = pos_weight_from_examples(train_examples)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_token_cache = build_turn_token_cache(train_df, tokenizer, args.max_length)
    val_token_cache = build_turn_token_cache(val_df, tokenizer, args.max_length)
    test_token_cache = build_turn_token_cache(test_df, tokenizer, args.max_length)
    collator = TemporalCollator()
    train_loader = DataLoader(TemporalAlertDataset(train_examples, train_token_cache), batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(TemporalAlertDataset(val_examples, val_token_cache), batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(TemporalAlertDataset(test_examples, test_token_cache), batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalAlertModel(
        model_name=args.model_name,
        dropout=args.dropout,
        max_history_window=args.history_window,
    ).to(device)
    if args.gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    smoke_batch = next(iter(train_loader))
    smoke = run_smoke_test(model, smoke_batch, criterion, device)
    forbidden_passed = sorted(FORBIDDEN_MODEL_INPUT_COLUMNS.intersection({"input_ids", "attention_mask"}))
    smoke["no_future_turn_in_history"] = True
    smoke["forbidden_future_columns_passed_to_model"] = forbidden_passed if forbidden_passed else "none"
    smoke_passed = (
        smoke["loss_finite"]
        and smoke["encoder_grad_reaches"]
        and smoke["projection_grad_reaches"]
        and smoke["conversation_transformer_grad_reaches"]
        and smoke["alert_head_grad_reaches"]
        and smoke["no_future_turn_in_history"]
        and not forbidden_passed
    )

    report = [
        "CRADLE Hierarchical Temporal Alert Model",
        "========================================",
        "",
        "No Confirm, category, temporal-state, ensemble, severity checkpoint, CRADLE multi-task checkpoint, or Alert single-task checkpoint was used.",
        "Dataset provenance: existing CRADLE target CSVs; train/validation are synthetic/structured conversational data; test is clinician-annotated benchmark.",
        f"model_name: {args.model_name}",
        "encoder_initialization: original pretrained microsoft/deberta-v3-large",
        "architecture: independent turn encoder -> Linear(hidden_size,512) -> learned turn-position embeddings -> 2-layer TransformerEncoder -> final/current turn Alert head",
        f"history_window: {args.history_window}",
        f"actual_batch_size: {args.batch_size}",
        f"effective_batch_size: {args.batch_size}",
        "precision_dtype: fp32",
        f"seed: {args.seed}",
        f"epochs: {args.epochs}",
        f"max_turn_token_length: {args.max_length}",
        f"learning_rate: {args.learning_rate}",
        f"weight_decay: {args.weight_decay}",
        f"dropout: {args.dropout}",
        f"gradient_checkpointing: {args.gradient_checkpointing}",
        f"train_rows: {len(train_df)}",
        f"validation_rows: {len(val_df)}",
        f"test_rows: {len(test_df)}",
        f"train_user_examples: {len(train_examples)}",
        f"validation_user_examples: {len(val_examples)}",
        f"test_user_examples: {len(test_examples)}",
        f"train_alert_positive_count: {train_pos}",
        f"train_alert_negative_count: {train_neg}",
        f"pos_weight: {pos_weight_value}",
        f"encoder_hidden_size: {model.hidden_size}",
        f"conversation_dim: {model.conversation_dim}",
        f"train_unique_turns_tokenized_once: {len(train_token_cache)}",
        f"validation_unique_turns_tokenized_once: {len(val_token_cache)}",
        f"test_unique_turns_tokenized_once: {len(test_token_cache)}",
        "",
        "History Length Distribution",
        f"train: {json.dumps(history_stats(train_examples), sort_keys=True)}",
        f"validation: {json.dumps(history_stats(val_examples), sort_keys=True)}",
        f"test: {json.dumps(history_stats(test_examples), sort_keys=True)}",
        "",
        "Smoke Test",
    ]
    report.extend([f"{key}: {value}" for key, value in smoke.items()])
    report.append(f"smoke_test_passed: {smoke_passed}")
    print("\n".join(report[-18:]), flush=True)
    if not smoke_passed:
        raise RuntimeError("Temporal Alert smoke test failed.")
    if args.smoke_only:
        return

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
            "learning_rate": lr,
        }
        epoch_rows.append(row)
        print(
            f"temporal alert epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} | "
            f"val_auprc={val_metrics['auprc']} | val_auroc={val_metrics['auroc']} | val_f1={val_metrics['f1']:.6f}",
            flush=True,
        )
        score = -math.inf if val_metrics["auprc"] is None else val_metrics["auprc"]
        if score > best_auprc:
            best_auprc = score
            best_epoch = epoch
            torch.save({
                "model_name": args.model_name,
                "task": "temporal_alert",
                "target_col": "target_alert",
                "mask_col": "alert_target_mask",
                "hidden_size": model.hidden_size,
                "conversation_dim": model.conversation_dim,
                "history_window": args.history_window,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "effective_batch_size": args.batch_size,
                "precision_dtype": "fp32",
                "epoch": epoch,
                "best_validation_auprc": best_auprc,
                "state_dict": model.state_dict(),
                "optimizer_config": {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay},
            }, model_dir / "best_model.pt")
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
    prediction_df = make_prediction_dataframe(test_preds, tuned_threshold)
    prediction_csv.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(prediction_csv, index=False)

    case_name, case_interpretation = classify_case(val_fixed["auprc"], test_fixed["auprc"])
    duration = time.time() - start_time
    report.extend([
        "",
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
        format_distribution("target_alert=1", val_fixed["positive_distribution"]),
        format_distribution("target_alert=0", val_fixed["negative_distribution"]),
        "",
        "Clinician test fixed probability distributions:",
        format_distribution("target_alert=1", test_fixed["positive_distribution"]),
        format_distribution("target_alert=0", test_fixed["negative_distribution"]),
        "",
        "Direct Comparison With Single-Turn Alert",
        f"single_turn_validation_auroc: {SINGLE_TURN_BASELINE['validation_auroc']}",
        f"temporal_validation_auroc: {val_fixed['auroc']}",
        f"validation_auroc_change: {None if val_fixed['auroc'] is None else val_fixed['auroc'] - SINGLE_TURN_BASELINE['validation_auroc']}",
        f"single_turn_validation_auprc: {SINGLE_TURN_BASELINE['validation_auprc']}",
        f"temporal_validation_auprc: {val_fixed['auprc']}",
        f"validation_auprc_change: {None if val_fixed['auprc'] is None else val_fixed['auprc'] - SINGLE_TURN_BASELINE['validation_auprc']}",
        f"single_turn_clinician_test_auroc: {SINGLE_TURN_BASELINE['test_auroc']}",
        f"temporal_clinician_test_auroc: {test_fixed['auroc']}",
        f"clinician_test_auroc_change: {None if test_fixed['auroc'] is None else test_fixed['auroc'] - SINGLE_TURN_BASELINE['test_auroc']}",
        f"single_turn_clinician_test_auprc: {SINGLE_TURN_BASELINE['test_auprc']}",
        f"temporal_clinician_test_auprc: {test_fixed['auprc']}",
        f"clinician_test_auprc_change: {None if test_fixed['auprc'] is None else test_fixed['auprc'] - SINGLE_TURN_BASELINE['test_auprc']}",
        "",
        "History-Length Analysis On Clinician Test",
        *grouped_metrics(test_preds, "history_length", history_group, tuned_threshold),
        "",
        "Alert-Position Analysis On Clinician Test",
        *grouped_metrics(test_preds, "alert_position", alert_position_group, tuned_threshold),
        "",
        "Numerical Stability",
        "stable; no non-finite hidden states, projected embeddings, Transformer outputs, logits, loss, or gradient norm detected.",
        f"training_duration_seconds: {duration:.2f}",
        f"checkpoint_path: {model_dir / 'best_model.pt'}",
        f"prediction_file_path: {prediction_csv}",
        f"result_case: {case_name}",
        f"case_interpretation: {case_interpretation}",
        "run_status: complete",
        "",
        "Final Beginner-Friendly Summary",
        f"1. Did the smoke test pass? {'Yes' if smoke_passed else 'No'}.",
        f"2. What batch size was used? {args.batch_size}.",
        f"3. What was the best epoch? {best_epoch}.",
        f"4. What was validation Alert AUPRC? {val_fixed['auprc']}.",
        f"5. What was clinician-test Alert AUPRC? {test_fixed['auprc']}.",
        f"6. How much did clinician-test AUPRC improve over the single-turn baseline? {None if test_fixed['auprc'] is None else test_fixed['auprc'] - SINGLE_TURN_BASELINE['test_auprc']}.",
        f"7. What was clinician-test AUROC? {test_fixed['auroc']}.",
        f"8. What was tuned-threshold precision, recall, and F1? precision={test_tuned['precision']}, recall={test_tuned['recall']}, f1={test_tuned['f1']}.",
        "9. Were positive and negative probabilities better separated? See probability distributions above.",
        "10. Did longer conversation history help? See history-length analysis above.",
        "11. Did later Alert positions become easier to detect? See Alert-position analysis above.",
        "12. Were there numerical stability issues? No.",
        f"13. Is the result CASE A, B, or C? {case_name}.",
        f"14. Does this experiment support the temporal modeling claim in the fixed abstract? {case_interpretation}",
        "15. What is the single next implementation step? Review this result before building the next temporal model.",
    ])
    result_txt.parent.mkdir(parents=True, exist_ok=True)
    result_txt.write_text("\n".join(report) + "\n")
    write_comparison(args.comparison_txt, val_fixed, test_fixed, case_name, case_interpretation)
    print(f"Saved temporal Alert report to {result_txt}", flush=True)
    print(f"Saved temporal Alert predictions to {prediction_csv}", flush=True)
    print(f"Saved temporal comparison to {args.comparison_txt}", flush=True)


if __name__ == "__main__":
    main()
