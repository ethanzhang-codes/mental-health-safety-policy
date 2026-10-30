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
MODEL_DIR = "cradle_dialogue/models/single_turn_category/seed_1"
RESULT_TXT = "cradle_dialogue/results/single_turn_category/seed_1.txt"
PREDICTION_CSV = "cradle_dialogue/predictions/single_turn_category/seed_1_test_predictions.csv"
THRESHOLD_JSON = "cradle_dialogue/results/single_turn_category/seed_1_thresholds.json"
SUMMARY_TXT = "cradle_dialogue/results/single_turn_category/category_model_summary.txt"

CATEGORY_ORDER = ["CA", "DV", "RA", "SHA", "SH", "SI_active", "SI_passive"]
CATEGORY_COLUMNS = [f"target_category_{name}" for name in CATEGORY_ORDER]
EXPECTED_COUNTS = {
    "train": {"CA": 81, "DV": 318, "RA": 279, "SHA": 221, "SH": 615, "SI_active": 306, "SI_passive": 417},
    "validation": {"CA": 8, "DV": 45, "RA": 41, "SHA": 33, "SH": 79, "SI_active": 44, "SI_passive": 59},
    "test": {"CA": 35, "DV": 56, "RA": 66, "SHA": 63, "SH": 104, "SI_active": 69, "SI_passive": 82},
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
    parser = argparse.ArgumentParser(description="Train CRADLE single-turn multi-label category classifier.")
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
    parser.add_argument("--model_dir", default=MODEL_DIR)
    parser.add_argument("--result_txt", default=RESULT_TXT)
    parser.add_argument("--prediction_csv", default=PREDICTION_CSV)
    parser.add_argument("--threshold_json", default=THRESHOLD_JSON)
    parser.add_argument("--summary_txt", default=SUMMARY_TXT)
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


class CategoryDataset(Dataset):
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
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "targets": torch.tensor([float(row[col]) for col in CATEGORY_COLUMNS], dtype=torch.float32),
            "mask": torch.tensor([float(row["category_target_mask"])], dtype=torch.float32),
        }


class CategoryModel(nn.Module):
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
            nn.Linear(256, len(CATEGORY_ORDER)),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("Non-finite encoder hidden states detected.")
        logits = self.classifier(hidden[:, 0, :])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite category logits detected.")
        return logits


def compute_pos_weight(df):
    masked = df[df["category_target_mask"].astype(float) == 1].copy()
    pos_counts = {}
    neg_counts = {}
    weights = []
    for name, col in zip(CATEGORY_ORDER, CATEGORY_COLUMNS):
        pos = int(masked[col].astype(float).sum())
        neg = int(len(masked) - pos)
        if pos <= 0:
            raise ValueError(f"No positive examples for {col}")
        pos_counts[name] = pos
        neg_counts[name] = neg
        weights.append(neg / pos)
    return pos_counts, neg_counts, torch.tensor(weights, dtype=torch.float32)


def masked_bce_loss(logits, targets, mask, criterion):
    per_item = criterion(logits, targets)
    selected = per_item[mask.expand_as(per_item) > 0]
    if selected.numel() == 0:
        return logits.sum() * 0.0
    loss = selected.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite masked category loss detected.")
    return loss


def maybe_auc(y_true, scores, kind):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        if kind == "roc":
            return float(roc_auc_score(y_true, scores))
        return float(average_precision_score(y_true, scores))
    except Exception:
        return None


def binary_metrics(y_true, probs, threshold):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    pred = (probs >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=[1], average=None, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "positive_support": int(support[0]),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "auroc": maybe_auc(y_true, probs, "roc"),
        "auprc": maybe_auc(y_true, probs, "pr"),
    }


def metric_value(values, reduction):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    if reduction == "macro":
        return float(np.mean(filtered))
    raise ValueError(f"Unknown reduction: {reduction}")


def multilabel_metrics(y_true, probs, thresholds):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs).astype(float)
    thresholds_arr = np.asarray(thresholds).reshape(1, -1)
    pred = (probs >= thresholds_arr).astype(int)

    per_label = {}
    aurocs = []
    auprcs = []
    f1s = []
    for idx, name in enumerate(CATEGORY_ORDER):
        metrics = binary_metrics(y_true[:, idx], probs[:, idx], float(thresholds[idx]))
        per_label[name] = metrics
        aurocs.append(metrics["auroc"])
        auprcs.append(metrics["auprc"])
        f1s.append(metrics["f1"])

    micro_auroc = None
    micro_auprc = None
    try:
        micro_auroc = float(roc_auc_score(y_true.reshape(-1), probs.reshape(-1)))
    except Exception:
        pass
    try:
        micro_auprc = float(average_precision_score(y_true.reshape(-1), probs.reshape(-1)))
    except Exception:
        pass

    return {
        "macro_auroc": metric_value(aurocs, "macro"),
        "macro_auprc": metric_value(auprcs, "macro"),
        "micro_auroc": micro_auroc,
        "micro_auprc": micro_auprc,
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "per_label": per_label,
    }


def tune_thresholds(y_true, probs):
    thresholds = {}
    best_f1s = {}
    for idx, name in enumerate(CATEGORY_ORDER):
        label_true = y_true[:, idx].astype(int)
        label_prob = probs[:, idx].astype(float)
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in np.linspace(0.01, 0.99, 99):
            score = f1_score(label_true, (label_prob >= threshold).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)
        thresholds[name] = best_threshold
        best_f1s[name] = best_f1
    return thresholds, best_f1s


@torch.no_grad()
def collect_predictions(model, loader, device, criterion=None):
    model.eval()
    all_probs = []
    all_targets = []
    all_masks = []
    total_loss = 0.0
    steps = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"])
        if criterion is not None:
            loss = masked_bce_loss(logits, batch["targets"], batch["mask"], criterion)
            total_loss += float(loss.detach().cpu())
            steps += 1
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(batch["targets"].cpu().numpy())
        all_masks.append(batch["mask"].cpu().numpy())
    return {
        "probs": np.concatenate(all_probs, axis=0),
        "targets": np.concatenate(all_targets, axis=0),
        "mask": np.concatenate(all_masks, axis=0).reshape(-1).astype(bool),
        "loss": total_loss / max(steps, 1) if criterion is not None else None,
    }


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = masked_bce_loss(logits, batch["targets"], batch["mask"], criterion)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}: {grad_norm}")
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(loader), 1), float(scheduler.get_last_lr()[0])


def format_value(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def summarize_label_metrics(split_name, label_name, fixed_metrics, tuned_metrics):
    return (
        f"{split_name} {label_name}: "
        f"auroc={format_value(fixed_metrics['auroc'])}, "
        f"auprc={format_value(fixed_metrics['auprc'])}, "
        f"fixed_precision={format_value(fixed_metrics['precision'])}, "
        f"fixed_recall={format_value(fixed_metrics['recall'])}, "
        f"fixed_f1={format_value(fixed_metrics['f1'])}, "
        f"tuned_precision={format_value(tuned_metrics['precision'])}, "
        f"tuned_recall={format_value(tuned_metrics['recall'])}, "
        f"tuned_f1={format_value(tuned_metrics['f1'])}, "
        f"positive_support={fixed_metrics['positive_support']}, "
        f"false_positives={tuned_metrics['false_positives']}, "
        f"false_negatives={tuned_metrics['false_negatives']}, "
        f"threshold={format_value(tuned_metrics['threshold'])}"
    )


def save_predictions(df, preds, tuned_thresholds, output_path):
    out = df[["dataset_split", "dialogue_id", "turn_id", "speaker_role", "text"]].copy()
    for idx, name in enumerate(CATEGORY_ORDER):
        true_col = CATEGORY_COLUMNS[idx]
        prob_col = f"category_{name}_prob"
        fixed_col = f"category_{name}_pred_fixed_0_5"
        tuned_col = f"category_{name}_pred_tuned"
        out[true_col] = preds["targets"][:, idx].astype(int)
        out[prob_col] = preds["probs"][:, idx]
        out[fixed_col] = (preds["probs"][:, idx] >= 0.5).astype(int)
        out[tuned_col] = (preds["probs"][:, idx] >= tuned_thresholds[name]).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def prediction_path_for_split(base_path, split_name):
    name = base_path.name
    if "_test_predictions" in name:
        return base_path.with_name(name.replace("_test_predictions", f"_{split_name}_predictions"))
    return base_path.with_name(f"{base_path.stem}_{split_name}{base_path.suffix}")


def validation_checks(train_df, val_df, test_df):
    checks = []
    for col in CATEGORY_COLUMNS:
        if col not in train_df.columns or col not in val_df.columns or col not in test_df.columns:
            raise ValueError(f"Missing category target column: {col}")
    if "category_target_mask" not in train_df.columns:
        raise ValueError("Missing category_target_mask column.")
    for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if df["text"].isna().any():
            raise ValueError(f"Missing text detected in {split_name}.")
        if df[CATEGORY_COLUMNS + ["category_target_mask"]].isna().any().any():
            raise ValueError(f"NaN labels detected in {split_name}.")
        for col in CATEGORY_COLUMNS + ["category_target_mask"]:
            unique_values = set(pd.Series(df[col]).astype(float).unique().tolist())
            if not unique_values.issubset({0.0, 1.0}):
                raise ValueError(f"Non-binary values found in {split_name}:{col} -> {sorted(unique_values)}")
        masked = df[df["category_target_mask"].astype(float) == 1]
        if not (masked["is_user_turn"].astype(float) == 1).all():
            raise ValueError(f"Non-user masked rows found in {split_name}.")
        counts = {name: int(masked[f"target_category_{name}"].astype(float).sum()) for name in CATEGORY_ORDER}
        checks.append((split_name, len(df), len(masked), counts))
    return checks


def export_all_splits(args):
    model_dir = Path(args.model_dir)
    prediction_csv = Path(args.prediction_csv)
    threshold_json = Path(args.threshold_json)
    if not (model_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_dir / 'best_model.pt'}")
    if not threshold_json.exists():
        raise FileNotFoundError(f"Threshold file not found: {threshold_json}")

    set_seed(args.seed)
    train_df = read_targets(args.train_csv)
    val_df = read_targets(args.val_csv)
    test_df = read_targets(args.test_csv)

    tokenizer_source = model_dir if (model_dir / "tokenizer.json").exists() else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CategoryModel(args.model_name, dropout=args.dropout).to(device)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    threshold_payload = json.loads(threshold_json.read_text())
    tuned_thresholds = threshold_payload["selected_thresholds"]

    for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        ds = CategoryDataset(df, tokenizer, args.max_length)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        preds = collect_predictions(model, loader, device, criterion=None)
        out_path = prediction_path_for_split(prediction_csv, split_name)
        save_predictions(df, preds, tuned_thresholds, out_path)
        print(f"Saved category {split_name} predictions to {out_path}")


def main():
    args = parse_args()
    if args.export_all_splits_only:
        export_all_splits(args)
        return
    model_dir = Path(args.model_dir)
    result_txt = Path(args.result_txt)
    prediction_csv = Path(args.prediction_csv)
    threshold_json = Path(args.threshold_json)
    summary_txt = Path(args.summary_txt)
    if model_dir.exists() or result_txt.exists() or prediction_csv.exists() or threshold_json.exists() or summary_txt.exists():
        raise FileExistsError("Refusing to overwrite existing single-turn category outputs.")

    set_seed(args.seed)
    start_time = time.time()

    train_df = read_targets(args.train_csv)
    val_df = read_targets(args.val_csv)
    test_df = read_targets(args.test_csv)
    checks = validation_checks(train_df, val_df, test_df)
    train_pos_counts, train_neg_counts, pos_weight = compute_pos_weight(train_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CategoryModel(args.model_name, dropout=args.dropout).to(device)
    if args.gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device), reduction="none")

    train_ds = CategoryDataset(train_df, tokenizer, args.max_length)
    val_ds = CategoryDataset(val_df, tokenizer, args.max_length)
    test_ds = CategoryDataset(test_df, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_input_keys = {"input_ids", "attention_mask"}
    forbidden_in_inputs = sorted(FORBIDDEN_MODEL_INPUT_COLUMNS.intersection(model_input_keys))

    report = [
        "CRADLE Single-Turn Category Classifier",
        "======================================",
        "",
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
        f"category_order: {CATEGORY_ORDER}",
        f"encoder_hidden_size: {model.hidden_size}",
        f"total_parameters: {total_params}",
        f"trainable_parameters: {trainable_params}",
        f"model_input_keys: {sorted(model_input_keys)}",
        f"forbidden_future_columns_passed_to_model: {forbidden_in_inputs if forbidden_in_inputs else 'none'}",
        "",
        "Validation Checks",
        "-----------------",
    ]
    for split_name, row_count, masked_count, counts in checks:
        report.append(f"{split_name}_rows: {row_count}")
        report.append(f"{split_name}_masked_user_turns: {masked_count}")
        report.append(f"{split_name}_positive_counts: {json.dumps(counts, sort_keys=True)}")
        report.append(f"{split_name}_expected_positive_counts: {json.dumps(EXPECTED_COUNTS[split_name], sort_keys=True)}")
    report.extend([
        "all_category_target_columns_exist: True",
        "no_missing_text: True",
        "no_nan_labels: True",
        "labels_binary_0_1: True",
        "listener_turns_used_for_category_loss: False",
        "future_metadata_used_as_model_input: False",
        "model_uses_only_text_and_speaker_prefix_as_input: True",
        "",
        "Train Pos Weight",
        "----------------",
    ])
    for name in CATEGORY_ORDER:
        report.append(
            f"{name}: positive_count={train_pos_counts[name]}, negative_count={train_neg_counts[name]}, "
            f"pos_weight={train_neg_counts[name] / train_pos_counts[name]}"
        )
    report.append("")

    smoke = next(iter(train_loader))
    smoke = {k: v.to(device) for k, v in smoke.items()}
    smoke_logits = model(smoke["input_ids"], smoke["attention_mask"])
    smoke_loss = masked_bce_loss(smoke_logits, smoke["targets"], smoke["mask"], criterion)
    report.extend([
        f"smoke_logit_shape: {tuple(smoke_logits.shape)}",
        f"smoke_loss_finite: {bool(torch.isfinite(smoke_loss).item())}",
        f"microsoft_deberta_v3_large_initialization_verified: {args.model_name == MODEL_NAME}",
        "",
    ])
    if forbidden_in_inputs or not torch.isfinite(smoke_loss):
        raise RuntimeError("Smoke test failed.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_macro_auprc = -math.inf
    best_epoch = -1
    epoch_rows = []
    model_dir.mkdir(parents=True, exist_ok=False)

    for epoch in range(1, args.epochs + 1):
        train_loss, lr = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_preds = collect_predictions(model, val_loader, device, criterion)
        val_y_true = val_preds["targets"][val_preds["mask"]]
        val_probs = val_preds["probs"][val_preds["mask"]]
        fixed_thresholds = {name: 0.5 for name in CATEGORY_ORDER}
        val_fixed = multilabel_metrics(val_y_true, val_probs, [0.5] * len(CATEGORY_ORDER))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_preds["loss"],
            "validation_macro_auroc": val_fixed["macro_auroc"],
            "validation_macro_auprc": val_fixed["macro_auprc"],
            "validation_micro_auroc": val_fixed["micro_auroc"],
            "validation_micro_auprc": val_fixed["micro_auprc"],
            "validation_macro_f1_fixed_0_5": val_fixed["macro_f1"],
            "validation_micro_f1_fixed_0_5": val_fixed["micro_f1"],
            "learning_rate": lr,
        }
        epoch_rows.append(row)
        print(
            f"category epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} | "
            f"val_macro_auprc={format_value(val_fixed['macro_auprc'])} | "
            f"val_micro_auprc={format_value(val_fixed['micro_auprc'])} | "
            f"val_macro_f1={format_value(val_fixed['macro_f1'])}",
            flush=True,
        )
        score = -math.inf if val_fixed["macro_auprc"] is None else val_fixed["macro_auprc"]
        if score > best_macro_auprc:
            best_macro_auprc = score
            best_epoch = epoch
            torch.save(
                {
                    "model_name": args.model_name,
                    "seed": args.seed,
                    "batch_size": args.batch_size,
                    "precision_dtype": "fp32",
                    "epoch": epoch,
                    "best_validation_macro_auprc": best_macro_auprc,
                    "category_order": CATEGORY_ORDER,
                    "state_dict": model.state_dict(),
                    "optimizer_config": {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay},
                },
                model_dir / "best_model.pt",
            )
            tokenizer.save_pretrained(model_dir)

    checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    val_preds = collect_predictions(model, val_loader, device, criterion)
    test_preds = collect_predictions(model, test_loader, device, criterion)
    val_y_true = val_preds["targets"][val_preds["mask"]]
    val_probs = val_preds["probs"][val_preds["mask"]]
    test_y_true = test_preds["targets"][test_preds["mask"]]
    test_probs = test_preds["probs"][test_preds["mask"]]

    tuned_thresholds, best_threshold_f1s = tune_thresholds(val_y_true, val_probs)
    tuned_threshold_list = [tuned_thresholds[name] for name in CATEGORY_ORDER]

    val_fixed = multilabel_metrics(val_y_true, val_probs, [0.5] * len(CATEGORY_ORDER))
    val_tuned = multilabel_metrics(val_y_true, val_probs, tuned_threshold_list)
    test_fixed = multilabel_metrics(test_y_true, test_probs, [0.5] * len(CATEGORY_ORDER))
    test_tuned = multilabel_metrics(test_y_true, test_probs, tuned_threshold_list)

    threshold_payload = {
        "category_order": CATEGORY_ORDER,
        "selected_thresholds": tuned_thresholds,
        "validation_best_f1_per_category": best_threshold_f1s,
    }
    threshold_json.parent.mkdir(parents=True, exist_ok=True)
    threshold_json.write_text(json.dumps(threshold_payload, indent=2) + "\n")

    save_predictions(test_df, test_preds, tuned_thresholds, prediction_csv)

    duration = time.time() - start_time

    report.extend(["Epoch Metrics", "-------------"])
    for row in epoch_rows:
        report.append(json.dumps(row, sort_keys=True))

    report.extend(
        [
            "",
            f"best_epoch: {best_epoch}",
            f"best_validation_macro_auprc: {best_macro_auprc}",
            f"validation_macro_auprc_fixed_0_5: {val_fixed['macro_auprc']}",
            f"validation_micro_auprc_fixed_0_5: {val_fixed['micro_auprc']}",
            f"validation_macro_f1_fixed_0_5: {val_fixed['macro_f1']}",
            f"validation_micro_f1_fixed_0_5: {val_fixed['micro_f1']}",
            f"validation_macro_f1_tuned: {val_tuned['macro_f1']}",
            f"validation_micro_f1_tuned: {val_tuned['micro_f1']}",
            "",
            "Selected Validation Thresholds",
            "------------------------------",
            json.dumps(tuned_thresholds, sort_keys=True),
            "",
            "Validation Per-Category Metrics",
            "-------------------------------",
        ]
    )
    for name in CATEGORY_ORDER:
        report.append(
            summarize_label_metrics(
                "validation",
                name,
                val_fixed["per_label"][name],
                val_tuned["per_label"][name],
            )
        )
    report.extend(
        [
            "",
            "Clinician Test Per-Category Metrics",
            "-----------------------------------",
        ]
    )
    for name in CATEGORY_ORDER:
        report.append(
            summarize_label_metrics(
                "clinician_test",
                name,
                test_fixed["per_label"][name],
                test_tuned["per_label"][name],
            )
        )

    report.extend(
        [
            "",
            "Aggregate Validation Metrics",
            "----------------------------",
            f"macro_auroc_fixed_0_5: {val_fixed['macro_auroc']}",
            f"macro_auprc_fixed_0_5: {val_fixed['macro_auprc']}",
            f"micro_auroc_fixed_0_5: {val_fixed['micro_auroc']}",
            f"micro_auprc_fixed_0_5: {val_fixed['micro_auprc']}",
            f"macro_f1_fixed_0_5: {val_fixed['macro_f1']}",
            f"micro_f1_fixed_0_5: {val_fixed['micro_f1']}",
            f"macro_f1_tuned: {val_tuned['macro_f1']}",
            f"micro_f1_tuned: {val_tuned['micro_f1']}",
            "",
            "Aggregate Clinician Test Metrics",
            "---------------------------------",
            f"macro_auroc_fixed_0_5: {test_fixed['macro_auroc']}",
            f"macro_auprc_fixed_0_5: {test_fixed['macro_auprc']}",
            f"micro_auroc_fixed_0_5: {test_fixed['micro_auroc']}",
            f"micro_auprc_fixed_0_5: {test_fixed['micro_auprc']}",
            f"macro_f1_fixed_0_5: {test_fixed['macro_f1']}",
            f"micro_f1_fixed_0_5: {test_fixed['micro_f1']}",
            f"macro_f1_tuned: {test_tuned['macro_f1']}",
            f"micro_f1_tuned: {test_tuned['micro_f1']}",
            "",
        ]
    )

    best_category = max(test_fixed["per_label"].items(), key=lambda item: (-1 if item[1]["auprc"] is None else item[1]["auprc"]))
    worst_category = min(test_fixed["per_label"].items(), key=lambda item: (math.inf if item[1]["auprc"] is None else item[1]["auprc"]))
    interpretation = (
        "Category prediction can be handled as a current-turn multi-label recognition task linked to explicit Confirm evidence."
        if (test_fixed["macro_auprc"] or 0.0) >= 0.30
        else "Category prediction may require either more data, stronger class balancing, or temporal context for some categories, especially low-support categories such as CA."
    )

    report.extend(
        [
            "Scientific Interpretation",
            "-------------------------",
            interpretation,
            "",
            f"training_duration_seconds: {duration:.2f}",
            "numerical_stability: stable; no non-finite hidden states, logits, loss, or gradient norm detected.",
            f"checkpoint_path: {model_dir / 'best_model.pt'}",
            f"prediction_file_path: {prediction_csv}",
            f"threshold_file_path: {threshold_json}",
            "run_status: complete",
        ]
    )
    result_txt.parent.mkdir(parents=True, exist_ok=True)
    result_txt.write_text("\n".join(report) + "\n")

    summary_lines = [
        "CRADLE Single-Turn Category Summary",
        "===================================",
        f"Best epoch: {best_epoch}",
        f"Validation macro AUPRC: {format_value(val_fixed['macro_auprc'])}",
        f"Clinician-test macro AUPRC: {format_value(test_fixed['macro_auprc'])}",
        f"Clinician-test micro AUPRC: {format_value(test_fixed['micro_auprc'])}",
        f"Best category by clinician-test AUPRC: {best_category[0]} ({format_value(best_category[1]['auprc'])})",
        f"Worst category by clinician-test AUPRC: {worst_category[0]} ({format_value(worst_category[1]['auprc'])})",
        interpretation,
        "",
        "Final Beginner-Friendly Summary",
        f"1. Did the category target columns load correctly? Yes.",
        f"2. How many training examples were used? {int(train_df['category_target_mask'].astype(float).sum())} masked user turns.",
        f"3. What were the positive counts for each category? {json.dumps(train_pos_counts, sort_keys=True)}.",
        f"4. What was the best epoch? {best_epoch}.",
        f"5. What was validation macro AUPRC? {format_value(val_fixed['macro_auprc'])}.",
        f"6. What was clinician-test macro AUPRC? {format_value(test_fixed['macro_auprc'])}.",
        f"7. What was clinician-test micro AUPRC? {format_value(test_fixed['micro_auprc'])}.",
        f"8. Which category performed best? {best_category[0]} with clinician-test AUPRC {format_value(best_category[1]['auprc'])}.",
        f"9. Which category performed worst? {worst_category[0]} with clinician-test AUPRC {format_value(worst_category[1]['auprc'])}.",
        f"10. Did current-turn category classification work well enough to keep as the working design? {interpretation}",
        "11. What is the single next implementation step? Inspect per-category failures, especially low-support categories, before deciding whether any specific category needs temporal context.",
    ]
    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text("\n".join(summary_lines) + "\n")

    print(f"Saved report to {result_txt}")
    print(f"Saved predictions to {prediction_csv}")
    print(f"Saved thresholds to {threshold_json}")
    print(f"Saved summary to {summary_txt}")


if __name__ == "__main__":
    main()
