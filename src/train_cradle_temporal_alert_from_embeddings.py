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


TRAIN_EMBEDDINGS = "cradle_dialogue/embeddings/deberta_v3_large_train_turn_embeddings.pt"
VAL_EMBEDDINGS = "cradle_dialogue/embeddings/deberta_v3_large_validation_turn_embeddings.pt"
TEST_EMBEDDINGS = "cradle_dialogue/embeddings/deberta_v3_large_test_turn_embeddings.pt"
TRAIN_CSV = "cradle_dialogue/data/cradle_train_targets.csv"
VAL_CSV = "cradle_dialogue/data/cradle_validation_targets.csv"
TEST_CSV = "cradle_dialogue/data/cradle_test_targets.csv"
MODEL_DIR = "cradle_dialogue/models/temporal_alert_embeddings/seed_1"
RESULT_TXT = "cradle_dialogue/results/temporal_alert_embeddings/seed_1.txt"
PREDICTION_CSV = "cradle_dialogue/predictions/temporal_alert_embeddings/seed_1_test_predictions.csv"
COMPARISON_TXT = "cradle_dialogue/results/temporal_alert_embeddings/single_turn_vs_temporal_alert_comparison.txt"

TASK_CONFIGS = {
    "alert": {
        "title_name": "Alert",
        "target_key": "target_alert",
        "mask_key": "alert_target_mask",
        "probability_key": "alert_probability",
        "prediction_prefix": "alert",
        "single_turn_baseline": {
            "validation_auroc": 0.607675,
            "validation_auprc": 0.090167,
            "test_auroc": 0.536585,
            "test_auprc": 0.055381,
            "test_fixed_f1": 0.0,
        },
    },
    "confirm": {
        "title_name": "Confirm",
        "target_key": "target_confirm",
        "mask_key": "confirm_target_mask",
        "probability_key": "confirm_probability",
        "prediction_prefix": "confirm",
        "single_turn_baseline": {
            "validation_auroc": 0.9963029569875084,
            "validation_auprc": 0.9618512469857508,
            "test_auroc": 0.8667753134849668,
            "test_auprc": 0.6430111066527171,
            "test_fixed_f1": 0.515021,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train CRADLE temporal model from frozen turn embeddings.")
    parser.add_argument("--task", choices=sorted(TASK_CONFIGS.keys()), default="alert")
    parser.add_argument("--train_embeddings", default=TRAIN_EMBEDDINGS)
    parser.add_argument("--val_embeddings", default=VAL_EMBEDDINGS)
    parser.add_argument("--test_embeddings", default=TEST_EMBEDDINGS)
    parser.add_argument("--train_csv", default=TRAIN_CSV)
    parser.add_argument("--val_csv", default=VAL_CSV)
    parser.add_argument("--test_csv", default=TEST_CSV)
    parser.add_argument("--model_dir", default=MODEL_DIR)
    parser.add_argument("--result_txt", default=RESULT_TXT)
    parser.add_argument("--prediction_csv", default=PREDICTION_CSV)
    parser.add_argument("--comparison_txt", default=COMPARISON_TXT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--history_window", type=int, default=8)
    parser.add_argument("--conversation_dim", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow_overwrite", action="store_true")
    parser.add_argument("--export_all_splits_only", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_embedding_split(path, csv_path):
    payload = torch.load(path, map_location="cpu")
    embeddings = payload["embeddings"].float()
    if not torch.isfinite(embeddings).all():
        raise FloatingPointError(f"Non-finite embeddings detected in {path}")

    csv_frame = None
    if "target_confirm" not in payload or "confirm_target_mask" not in payload:
        csv_frame = pd.read_csv(csv_path, keep_default_na=False)
        if len(csv_frame) != embeddings.shape[0]:
            raise ValueError(f"CSV row count {len(csv_frame)} does not match embedding rows {embeddings.shape[0]} for {path}")

    records = []
    for idx in range(embeddings.shape[0]):
        target_confirm = float(payload["target_confirm"][idx].item()) if "target_confirm" in payload else float(csv_frame.iloc[idx]["target_confirm"])
        confirm_target_mask = (
            float(payload["confirm_target_mask"][idx].item())
            if "confirm_target_mask" in payload
            else float(csv_frame.iloc[idx]["confirm_target_mask"])
        )
        records.append(
            {
                "dataset_split": payload["dataset_split"][idx],
                "dialogue_id": str(payload["dialogue_id"][idx]),
                "turn_id": int(payload["turn_id"][idx].item()),
                "speaker_role": str(payload["speaker_role"][idx]),
                "is_user_turn": int(payload["is_user_turn"][idx].item()),
                "target_alert": float(payload["target_alert"][idx].item()),
                "alert_target_mask": float(payload["alert_target_mask"][idx].item()),
                "target_confirm": target_confirm,
                "confirm_target_mask": confirm_target_mask,
                "row_index": int(payload["row_index"][idx].item()),
            }
        )
    return {
        "records": records,
        "embeddings": embeddings,
        "dimension": int(embeddings.shape[1]),
    }


def build_examples(split_payload, history_window, target_key, mask_key):
    records = sorted(
        split_payload["records"],
        key=lambda row: (row["dialogue_id"], row["turn_id"], row["row_index"]),
    )
    embedding_lookup = {
        (row["dialogue_id"], row["turn_id"], row["row_index"]): split_payload["embeddings"][row["row_index"]]
        for row in split_payload["records"]
    }

    dialogue_groups = {}
    for row in records:
        dialogue_groups.setdefault(row["dialogue_id"], []).append(row)

    examples = []
    for dialogue_id, rows in dialogue_groups.items():
        for pos, row in enumerate(rows):
            if int(row[mask_key]) != 1:
                continue
            start = max(0, pos - history_window + 1)
            history_rows = rows[start : pos + 1]
            current_turn_id = int(row["turn_id"])
            if any(int(history_row["turn_id"]) > current_turn_id for history_row in history_rows):
                raise ValueError("Future turn detected in temporal history.")
            history_indices = [
                (history_row["dialogue_id"], history_row["turn_id"], history_row["row_index"])
                for history_row in history_rows
            ]
            examples.append(
                {
                    "dataset_split": row["dataset_split"],
                    "dialogue_id": dialogue_id,
                    "turn_id": current_turn_id,
                    "speaker_role": row["speaker_role"],
                    "target": float(row[target_key]),
                    "mask": float(row[mask_key]),
                    "history_indices": history_indices,
                    "history_length": len(history_indices),
                    "embedding_lookup": embedding_lookup,
                }
            )
    return examples


def history_stats(examples):
    lengths = np.asarray([example["history_length"] for example in examples], dtype=float)
    return {
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(np.max(lengths)),
    }


class TemporalEmbeddingDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class TemporalEmbeddingCollator:
    def __call__(self, batch):
        history_lengths = []
        flat_embeddings = []
        metadata = []
        for item in batch:
            history_lengths.append(item["history_length"])
            metadata.append(
                {
                    "dataset_split": item["dataset_split"],
                    "dialogue_id": item["dialogue_id"],
                    "turn_id": item["turn_id"],
                    "speaker_role": item["speaker_role"],
                    "history_length": item["history_length"],
                    "target": item["target"],
                    "mask": item["mask"],
                }
            )
            for key in item["history_indices"]:
                flat_embeddings.append(item["embedding_lookup"][key])

        return {
            "flat_embeddings": torch.stack(flat_embeddings).float(),
            "history_lengths": torch.tensor(history_lengths, dtype=torch.long),
            "target": torch.tensor([[float(item["target"])] for item in batch], dtype=torch.float32),
            "mask": torch.tensor([[float(item["mask"])] for item in batch], dtype=torch.float32),
            "metadata": metadata,
        }


class TemporalAlertFromEmbeddingsModel(nn.Module):
    def __init__(
        self,
        input_dim,
        conversation_dim=512,
        nhead=8,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.3,
        max_history_window=8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.conversation_dim = conversation_dim
        self.max_history_window = max_history_window
        self.turn_projection = nn.Linear(input_dim, conversation_dim)
        self.position_embeddings = nn.Embedding(max_history_window, conversation_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=conversation_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.conversation_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.alert_head = nn.Sequential(
            nn.Linear(conversation_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, flat_embeddings, history_lengths):
        if not torch.isfinite(flat_embeddings).all():
            raise FloatingPointError("Non-finite input embeddings detected.")
        projected = self.turn_projection(flat_embeddings)
        if not torch.isfinite(projected).all():
            raise FloatingPointError("Non-finite projected embeddings detected.")

        batch_size = int(history_lengths.shape[0])
        max_turns = int(history_lengths.max().item())
        sequence = projected.new_zeros((batch_size, max_turns, projected.shape[-1]))
        padding_mask = torch.ones((batch_size, max_turns), dtype=torch.bool, device=projected.device)

        cursor = 0
        for row_idx, length_tensor in enumerate(history_lengths):
            length = int(length_tensor.item())
            sequence[row_idx, :length] = projected[cursor : cursor + length]
            padding_mask[row_idx, :length] = False
            cursor += length

        positions = torch.arange(max_turns, device=projected.device).unsqueeze(0).expand(batch_size, max_turns)
        sequence = sequence + self.position_embeddings(positions)
        contextual = self.conversation_transformer(sequence, src_key_padding_mask=padding_mask)
        if not torch.isfinite(contextual).all():
            raise FloatingPointError("Non-finite conversation outputs detected.")

        final_indices = (history_lengths - 1).long()
        current_context = contextual[torch.arange(batch_size, device=projected.device), final_indices]
        logits = self.alert_head(current_context)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite Alert logits detected.")
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


def tune_threshold(targets, probs, masks):
    valid = masks.astype(bool)
    y_true = targets[valid].astype(int)
    y_prob = probs[valid].astype(float)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, predictions, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def run_eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    all_masks = []
    all_metadata = []
    with torch.no_grad():
        for batch in loader:
            flat_embeddings = batch["flat_embeddings"].to(device)
            history_lengths = batch["history_lengths"].to(device)
            targets = batch["target"].to(device)
            masks = batch["mask"].to(device)
            logits = model(flat_embeddings, history_lengths)
            probs = torch.sigmoid(logits)
            if criterion is not None:
                loss = masked_loss(logits, targets, masks, criterion)
                total_loss += float(loss.item())
            all_probs.append(probs.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_masks.append(masks.detach().cpu())
            all_metadata.extend(batch["metadata"])
    return {
        "loss": total_loss / max(len(loader), 1),
        "probs": torch.cat(all_probs).numpy().reshape(-1),
        "targets": torch.cat(all_targets).numpy().reshape(-1),
        "masks": torch.cat(all_masks).numpy().reshape(-1),
        "metadata": all_metadata,
    }


def history_length_analysis(metadata, probs, targets, masks, threshold):
    groups = {
        "1_turn": lambda n: n == 1,
        "2_3_turns": lambda n: 2 <= n <= 3,
        "4_5_turns": lambda n: 4 <= n <= 5,
        "6_8_turns": lambda n: 6 <= n <= 8,
    }
    rows = []
    for label, predicate in groups.items():
        indices = [idx for idx, item in enumerate(metadata) if predicate(int(item["history_length"])) and masks[idx] == 1]
        if not indices:
            rows.append({"group": label, "positive_support": 0, "auroc": None, "auprc": None, "f1_tuned": None})
            continue
        y_true = targets[indices].astype(int)
        y_prob = probs[indices].astype(float)
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "group": label,
                "positive_support": int(y_true.sum()),
                "auroc": metric_or_none(roc_auc_score, y_true, y_prob),
                "auprc": metric_or_none(average_precision_score, y_true, y_prob),
                "f1_tuned": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    return rows


def alert_position_analysis(metadata, probs, targets, masks, threshold):
    groups = {
        "turn_id_0": lambda n: n == 0,
        "turn_id_1_2": lambda n: 1 <= n <= 2,
        "turn_id_3_5": lambda n: 3 <= n <= 5,
        "turn_id_6_plus": lambda n: n >= 6,
    }
    rows = []
    for label, predicate in groups.items():
        indices = [idx for idx, item in enumerate(metadata) if predicate(int(item["turn_id"])) and masks[idx] == 1]
        if not indices:
            rows.append({"group": label, "positive_support": 0, "mean_alert_probability": None, "recall_tuned": None})
            continue
        y_true = targets[indices].astype(int)
        y_prob = probs[indices].astype(float)
        y_pred = (y_prob >= threshold).astype(int)
        positive_support = int(y_true.sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
        rows.append(
            {
                "group": label,
                "positive_support": positive_support,
                "mean_alert_probability": float(np.mean(y_prob)) if len(y_prob) else None,
                "recall_tuned": recall,
            }
        )
    return rows


def write_predictions(path, metadata, probs, targets, masks, fixed_threshold, tuned_threshold, prediction_prefix, target_key, mask_key):
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_split",
        "dialogue_id",
        "turn_id",
        "speaker_role",
        "history_length",
        target_key,
        mask_key,
        f"{prediction_prefix}_probability",
        f"{prediction_prefix}_pred_fixed_0_5",
        f"{prediction_prefix}_pred_tuned",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item, prob, target, mask in zip(metadata, probs, targets, masks):
            writer.writerow(
                {
                    "dataset_split": item["dataset_split"],
                    "dialogue_id": item["dialogue_id"],
                    "turn_id": item["turn_id"],
                    "speaker_role": item["speaker_role"],
                    "history_length": item["history_length"],
                    target_key: int(target),
                    mask_key: int(mask),
                    f"{prediction_prefix}_probability": float(prob),
                    f"{prediction_prefix}_pred_fixed_0_5": int(prob >= fixed_threshold),
                    f"{prediction_prefix}_pred_tuned": int(prob >= tuned_threshold),
                }
            )


def prediction_path_for_split(base_path, split_name):
    name = base_path.name
    if "_test_predictions" in name:
        return base_path.with_name(name.replace("_test_predictions", f"_{split_name}_predictions"))
    return base_path.with_name(f"{base_path.stem}_{split_name}{base_path.suffix}")


def load_selected_threshold(result_path):
    if not result_path.exists():
        return 0.5
    for line in result_path.read_text().splitlines():
        if line.startswith("selected_validation_threshold:"):
            return float(line.split(":", 1)[1].strip())
    return 0.5


def export_all_splits(args, task_config):
    model_dir = Path(args.model_dir)
    result_path = Path(args.result_txt)
    prediction_path = Path(args.prediction_csv)
    if not (model_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_dir / 'best_model.pt'}")

    set_seed(args.seed)
    train_payload = load_embedding_split(args.train_embeddings, args.train_csv)
    val_payload = load_embedding_split(args.val_embeddings, args.val_csv)
    test_payload = load_embedding_split(args.test_embeddings, args.test_csv)
    input_dim = train_payload["dimension"]

    device = torch.device(args.device)
    model = TemporalAlertFromEmbeddingsModel(
        input_dim=input_dim,
        conversation_dim=args.conversation_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_history_window=args.history_window,
    ).to(device)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tuned_threshold = load_selected_threshold(result_path)

    for split_name, payload in [("train", train_payload), ("validation", val_payload), ("test", test_payload)]:
        examples = build_examples(payload, args.history_window, task_config["target_key"], task_config["mask_key"])
        dataset = TemporalEmbeddingDataset(examples)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=TemporalEmbeddingCollator())
        preds = run_eval(model, loader, criterion=None, device=device)
        out_path = prediction_path_for_split(prediction_path, split_name)
        write_predictions(
            out_path,
            preds["metadata"],
            preds["probs"],
            preds["targets"],
            preds["masks"],
            0.5,
            tuned_threshold,
            task_config["prediction_prefix"],
            task_config["target_key"],
            task_config["mask_key"],
        )
        print(f"Saved {args.task} temporal {split_name} predictions to {out_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    task_config = TASK_CONFIGS[args.task]
    baseline = task_config["single_turn_baseline"]
    if args.export_all_splits_only:
        export_all_splits(args, task_config)
        return

    model_dir = Path(args.model_dir)
    result_path = Path(args.result_txt)
    prediction_path = Path(args.prediction_csv)
    comparison_path = Path(args.comparison_txt)

    if not args.allow_overwrite:
        for path in [result_path, prediction_path, comparison_path, model_dir / "best_model.pt"]:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    train_payload = load_embedding_split(args.train_embeddings, args.train_csv)
    val_payload = load_embedding_split(args.val_embeddings, args.val_csv)
    test_payload = load_embedding_split(args.test_embeddings, args.test_csv)

    input_dim = train_payload["dimension"]
    if val_payload["dimension"] != input_dim or test_payload["dimension"] != input_dim:
        raise ValueError("Embedding dimensions do not match across splits.")

    train_examples = build_examples(train_payload, args.history_window, task_config["target_key"], task_config["mask_key"])
    val_examples = build_examples(val_payload, args.history_window, task_config["target_key"], task_config["mask_key"])
    test_examples = build_examples(test_payload, args.history_window, task_config["target_key"], task_config["mask_key"])

    history_distribution = {
        "train": history_stats(train_examples),
        "validation": history_stats(val_examples),
        "test": history_stats(test_examples),
    }

    pos = int(sum(example["target"] for example in train_examples if example["mask"] == 1))
    total = int(sum(example["mask"] for example in train_examples))
    neg = total - pos
    if pos == 0:
        raise ValueError("No positive Alert examples in training set.")
    pos_weight_value = neg / pos

    train_dataset = TemporalEmbeddingDataset(train_examples)
    val_dataset = TemporalEmbeddingDataset(val_examples)
    test_dataset = TemporalEmbeddingDataset(test_examples)
    collator = TemporalEmbeddingCollator()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    device = torch.device(args.device)
    model = TemporalAlertFromEmbeddingsModel(
        input_dim=input_dim,
        conversation_dim=args.conversation_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_history_window=args.history_window,
    ).to(device)

    batch = next(iter(train_loader))
    flat_embeddings = batch["flat_embeddings"].to(device)
    history_lengths = batch["history_lengths"].to(device)
    targets = batch["target"].to(device)
    masks = batch["mask"].to(device)
    logits = model(flat_embeddings, history_lengths)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device),
        reduction="none",
    )
    smoke_loss = masked_loss(logits, targets, masks, criterion)
    smoke_loss.backward()
    smoke_report = {
        "real_turns_encoded": int(flat_embeddings.shape[0]),
        "flattened_embedding_batch_shape": tuple(flat_embeddings.shape),
        "history_lengths_shape": tuple(history_lengths.shape),
        f"final_{task_config['prediction_prefix']}_logit_shape": tuple(logits.shape),
        "loss_finite": bool(torch.isfinite(smoke_loss).item()),
        "projection_grad_reaches": bool(model.turn_projection.weight.grad is not None),
        "conversation_transformer_grad_reaches": bool(
            model.conversation_transformer.layers[0].linear1.weight.grad is not None
        ),
        "alert_head_grad_reaches": bool(model.alert_head[0].weight.grad is not None),
        "smoke_test_passed": True,
    }
    model.zero_grad(set_to_none=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_time = time.time()
    epoch_metrics = []
    best_val_auprc = -1.0
    best_epoch = -1
    best_checkpoint = None
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device),
        reduction="none",
    )

    for epoch in range(args.epochs):
        model.train()
        train_loss_total = 0.0
        for batch in train_loader:
            flat_embeddings = batch["flat_embeddings"].to(device)
            history_lengths = batch["history_lengths"].to(device)
            targets = batch["target"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(flat_embeddings, history_lengths)
            loss = masked_loss(logits, targets, masks, criterion)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(float(grad_norm)):
                raise FloatingPointError("Non-finite gradient norm detected.")
            optimizer.step()
            train_loss_total += float(loss.item())

        val_eval = run_eval(model, val_loader, criterion, device)
        val_fixed = compute_metrics(val_eval["targets"], val_eval["probs"], val_eval["masks"], 0.5)
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss_total / max(len(train_loader), 1),
            "validation_loss": val_eval["loss"],
            "validation_auroc": val_fixed["auroc"],
            "validation_auprc": val_fixed["auprc"],
            "validation_precision": val_fixed["precision"],
            "validation_recall": val_fixed["recall"],
            "validation_f1": val_fixed["f1"],
        }
        epoch_metrics.append(epoch_row)
        print(
            f"temporal_alert_embeddings epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={epoch_row['train_loss']:.6f} | "
            f"val_auprc={epoch_row['validation_auprc']:.6f} | "
            f"val_auroc={epoch_row['validation_auroc']:.6f} | "
            f"val_f1={epoch_row['validation_f1']:.6f}"
        )
        if val_fixed["auprc"] is not None and val_fixed["auprc"] > best_val_auprc:
            best_val_auprc = val_fixed["auprc"]
            best_epoch = epoch + 1
            best_checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": best_epoch,
                "validation_auprc": best_val_auprc,
            }

    if best_checkpoint is None:
        raise RuntimeError("No checkpoint was selected.")

    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_checkpoint, model_dir / "best_model.pt")
    model.load_state_dict(best_checkpoint["state_dict"])

    val_eval = run_eval(model, val_loader, criterion, device)
    test_eval = run_eval(model, test_loader, criterion, device)
    tuned_threshold, tuned_val_f1 = tune_threshold(val_eval["targets"], val_eval["probs"], val_eval["masks"])

    val_fixed = compute_metrics(val_eval["targets"], val_eval["probs"], val_eval["masks"], 0.5)
    val_tuned = compute_metrics(val_eval["targets"], val_eval["probs"], val_eval["masks"], tuned_threshold)
    test_fixed = compute_metrics(test_eval["targets"], test_eval["probs"], test_eval["masks"], 0.5)
    test_tuned = compute_metrics(test_eval["targets"], test_eval["probs"], test_eval["masks"], tuned_threshold)

    history_analysis = history_length_analysis(
        test_eval["metadata"], test_eval["probs"], test_eval["targets"], test_eval["masks"], tuned_threshold
    )
    position_analysis = alert_position_analysis(
        test_eval["metadata"], test_eval["probs"], test_eval["targets"], test_eval["masks"], tuned_threshold
    )

    write_predictions(
        prediction_path,
        test_eval["metadata"],
        test_eval["probs"],
        test_eval["targets"],
        test_eval["masks"],
        0.5,
        tuned_threshold,
        task_config["prediction_prefix"],
        task_config["target_key"],
        task_config["mask_key"],
    )

    test_auprc_change = None if test_tuned["auprc"] is None else test_tuned["auprc"] - baseline["test_auprc"]
    test_auroc_change = None if test_tuned["auroc"] is None else test_tuned["auroc"] - baseline["test_auroc"]
    val_auprc_change = None if val_tuned["auprc"] is None else val_tuned["auprc"] - baseline["validation_auprc"]
    val_auroc_change = None if val_tuned["auroc"] is None else val_tuned["auroc"] - baseline["validation_auroc"]

    case_label = "CASE C"
    if test_auprc_change is not None and val_auprc_change is not None:
        if test_tuned["auprc"] > baseline["test_auprc"] and val_tuned["auprc"] > baseline["validation_auprc"]:
            case_label = "CASE A"
        elif val_tuned["auprc"] > baseline["validation_auprc"] and test_tuned["auprc"] <= baseline["test_auprc"]:
            case_label = "CASE B"

    lines = [
        f"CRADLE Temporal {task_config['title_name']} From Frozen Embeddings",
        "=" * len(f"CRADLE Temporal {task_config['title_name']} From Frozen Embeddings"),
        "",
        "Architecture",
        "------------",
        f"input_embedding_dim: {input_dim}",
        f"conversation_dim: {args.conversation_dim}",
        f"nhead: {args.nhead}",
        f"num_layers: {args.num_layers}",
        f"dim_feedforward: {args.dim_feedforward}",
        f"dropout: {args.dropout}",
        f"history_window: {args.history_window}",
        f"actual_batch_size: {args.batch_size}",
        f"effective_batch_size: {args.batch_size}",
        "encoder_initialization: frozen precomputed DeBERTa turn embeddings",
        f"task: {args.task}",
        f"target_key: {task_config['target_key']}",
        f"mask_key: {task_config['mask_key']}",
        f"seed: {args.seed}",
        f"epochs: {args.epochs}",
        f"learning_rate: {args.learning_rate}",
        f"weight_decay: {args.weight_decay}",
        "",
        "Dataset And Class Balance",
        "-------------------------",
        f"train_user_examples: {len(train_examples)}",
        f"validation_user_examples: {len(val_examples)}",
        f"test_user_examples: {len(test_examples)}",
        f"train_positive_count: {pos}",
        f"train_negative_count: {neg}",
        f"computed_pos_weight: {pos_weight_value}",
        "",
        "History-Length Distribution",
        "---------------------------",
        f"train: {json.dumps(history_distribution['train'], sort_keys=True)}",
        f"validation: {json.dumps(history_distribution['validation'], sort_keys=True)}",
        f"test: {json.dumps(history_distribution['test'], sort_keys=True)}",
        "",
        "Smoke Test",
        "----------",
    ]
    for key, value in smoke_report.items():
        lines.append(f"{key}: {value}")

    lines.extend(["", "Epoch Metrics", "-------------"])
    for row in epoch_metrics:
        lines.append(json.dumps(row, sort_keys=True))

    lines.extend(
        [
            "",
            f"best_epoch: {best_epoch}",
            f"best_validation_auprc: {best_val_auprc}",
            f"selected_validation_threshold: {tuned_threshold}",
            f"selected_validation_f1: {tuned_val_f1}",
            "",
            "Validation fixed 0.5:",
            json.dumps(val_fixed, sort_keys=True),
            "Validation tuned:",
            json.dumps(val_tuned, sort_keys=True),
            "Clinician test fixed 0.5:",
            json.dumps(test_fixed, sort_keys=True),
            "Clinician test tuned:",
            json.dumps(test_tuned, sort_keys=True),
            "",
            "Validation fixed probability distributions:",
            format_distribution("target=1", val_fixed["positive_distribution"]),
            format_distribution("target=0", val_fixed["negative_distribution"]),
            "",
            "Clinician test fixed probability distributions:",
            format_distribution("target=1", test_fixed["positive_distribution"]),
            format_distribution("target=0", test_fixed["negative_distribution"]),
            "",
            f"Direct comparison to single-turn {task_config['title_name']} baseline:",
            f"single_turn_validation_auroc: {baseline['validation_auroc']}",
            f"temporal_validation_auroc: {val_tuned['auroc']}",
            f"validation_auroc_change: {val_auroc_change}",
            f"single_turn_validation_auprc: {baseline['validation_auprc']}",
            f"temporal_validation_auprc: {val_tuned['auprc']}",
            f"validation_auprc_change: {val_auprc_change}",
            f"single_turn_clinician_test_auroc: {baseline['test_auroc']}",
            f"temporal_clinician_test_auroc: {test_tuned['auroc']}",
            f"clinician_test_auroc_change: {test_auroc_change}",
            f"single_turn_clinician_test_auprc: {baseline['test_auprc']}",
            f"temporal_clinician_test_auprc: {test_tuned['auprc']}",
            f"clinician_test_auprc_change: {test_auprc_change}",
            "",
            f"History-length analysis on {task_config['title_name']} clinician test:",
        ]
    )
    for row in history_analysis:
        lines.append(json.dumps(row, sort_keys=True))

    lines.extend(["", f"{task_config['title_name']}-position analysis on clinician test:"])
    for row in position_analysis:
        lines.append(json.dumps(row, sort_keys=True))

    lines.extend(
        [
            "",
            "Scientific interpretation",
            "-------------------------",
            case_label,
            "",
            f"training_duration_seconds: {time.time() - start_time:.2f}",
            "numerical_stability: stable; no non-finite embeddings, logits, loss, or gradient norm detected.",
            f"checkpoint_path: {model_dir / 'best_model.pt'}",
            f"prediction_file_path: {prediction_path}",
            "run_status: complete",
        ]
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("\n".join(lines) + "\n")

    comparison_lines = [
        f"TEMPORAL {task_config['title_name'].upper()} FROM FROZEN EMBEDDINGS",
        f"Single-turn validation AUROC: {baseline['validation_auroc']:.6f}",
        f"Temporal validation AUROC: {val_tuned['auroc']:.6f}",
        f"Validation AUROC change: {val_auroc_change:.6f}",
        f"Single-turn validation AUPRC: {baseline['validation_auprc']:.6f}",
        f"Temporal validation AUPRC: {val_tuned['auprc']:.6f}",
        f"Validation AUPRC change: {val_auprc_change:.6f}",
        f"Single-turn clinician test AUROC: {baseline['test_auroc']:.6f}",
        f"Temporal clinician test AUROC: {test_tuned['auroc']:.6f}",
        f"Clinician test AUROC change: {test_auroc_change:.6f}",
        f"Single-turn clinician test AUPRC: {baseline['test_auprc']:.6f}",
        f"Temporal clinician test AUPRC: {test_tuned['auprc']:.6f}",
        f"Clinician test AUPRC change: {test_auprc_change:.6f}",
        f"Single-turn fixed F1: {baseline['test_fixed_f1']:.6f}",
        f"Temporal fixed F1: {test_fixed['f1']:.6f}",
        f"Temporal tuned F1: {test_tuned['f1']:.6f}",
        f"Best epoch: {best_epoch}",
        f"Selected validation threshold: {tuned_threshold}",
        case_label,
    ]
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text("\n".join(comparison_lines) + "\n")

    print(f"Saved report to {result_path}")
    print(f"Saved predictions to {prediction_path}")
    print(f"Saved comparison to {comparison_path}")


if __name__ == "__main__":
    main()
