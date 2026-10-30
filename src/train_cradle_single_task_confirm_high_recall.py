import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


MODEL_NAME = "microsoft/deberta-v3-large"
TRAIN_CSV = "cradle_dialogue/data/cradle_train_targets.csv"
VAL_CSV = "cradle_dialogue/data/cradle_validation_targets.csv"
TEST_CSV = "cradle_dialogue/data/cradle_test_targets.csv"

MODEL_DIR = Path("cradle_dialogue/models/single_task_confirm_high_recall/seed_1")
RESULT_TXT = Path("cradle_dialogue/results/single_task_confirm_high_recall/seed_1.txt")
SUMMARY_TXT = Path("cradle_dialogue/results/single_task_confirm_high_recall/high_recall_confirm_summary.txt")
COMPARISON_TXT = Path("cradle_dialogue/results/single_task_confirm_high_recall/confirm_high_recall_comparison.txt")
PRED_DIR = Path("cradle_dialogue/predictions/single_task_confirm_high_recall")

TARGET_COL = "target_confirm"
MASK_COL = "confirm_target_mask"
FIXED_THRESHOLD = 0.5
THRESHOLDS = np.round(np.arange(0.001, 0.5001, 0.001), 3)

PREVIOUS_METRICS = {
    "validation_auroc": 0.996303,
    "validation_auprc": 0.961851,
    "test_auroc": 0.866775,
    "test_auprc": 0.643011,
    "tuned_threshold": 0.24,
    "test_recall": 0.3885350318471338,
    "test_false_negatives": 288,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a high-recall CRADLE single-turn Confirm model.")
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
    parser.add_argument(
        "--only_variant",
        choices=["A_standard_bce", "B_pos_weight_x2", "C_pos_weight_x4", "D_focal_gamma2"],
        default=None,
        help="Train and save only a single loss variant.",
    )
    return parser.parse_args()


def output_paths(only_variant=None):
    if only_variant is None:
        return {
            "model_dir": MODEL_DIR,
            "result_txt": RESULT_TXT,
            "summary_txt": SUMMARY_TXT,
            "comparison_txt": COMPARISON_TXT,
            "pred_dir": PRED_DIR,
            "tb_dir": Path("cradle_dialogue/tensorboard/single_task_confirm_high_recall/all_variants"),
        }
    suffix_dir = Path("cradle_dialogue") / "results" / "single_task_confirm_high_recall" / only_variant
    return {
        "model_dir": Path("cradle_dialogue/models/single_task_confirm_high_recall") / only_variant / "seed_1",
        "result_txt": suffix_dir / "seed_1.txt",
        "summary_txt": suffix_dir / "high_recall_confirm_summary.txt",
        "comparison_txt": suffix_dir / "confirm_high_recall_comparison.txt",
        "pred_dir": Path("cradle_dialogue/predictions/single_task_confirm_high_recall") / only_variant,
        "tb_dir": Path("cradle_dialogue/tensorboard/single_task_confirm_high_recall") / only_variant,
    }


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


class ConfirmDataset(Dataset):
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
            "target": torch.tensor([float(row[TARGET_COL])], dtype=torch.float32),
            "mask": torch.tensor([float(row[MASK_COL])], dtype=torch.float32),
        }


class ConfirmModel(nn.Module):
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


class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha_pos: float, gamma: float = 2.0):
        super().__init__()
        self.alpha_pos = alpha_pos
        self.alpha_neg = 1.0 - alpha_pos
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha = torch.where(targets == 1, torch.full_like(targets, self.alpha_pos), torch.full_like(targets, self.alpha_neg))
        return alpha * ((1 - pt) ** self.gamma) * bce


def masked_loss(logits, targets, masks, criterion):
    unreduced = criterion(logits, targets)
    selected = unreduced[masks > 0]
    if selected.numel() == 0:
        return logits.sum() * 0.0
    loss = selected.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite masked loss detected.")
    return loss


def pos_weight_from_train(df):
    masked = df[df[MASK_COL].astype(float) == 1]
    pos = int(masked[TARGET_COL].astype(float).sum())
    neg = int(len(masked) - pos)
    if pos == 0:
        raise ValueError("No positive training examples.")
    return pos, neg, neg / pos


def metric_or_none(fn, y_true, probs):
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(fn(y_true, probs))
    except Exception:
        return None


def compute_metrics(targets, probs, masks, threshold):
    active = masks.astype(bool)
    y_true = targets[active].astype(int)
    y_prob = probs[active].astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {
        "threshold": float(threshold),
        "precision": float(precision[0]),
        "recall": float(recall[0]),
        "f1": float(f1[0]),
        "positive_support": int(support[0]),
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "true_negatives": tn,
        "auroc": metric_or_none(roc_auc_score, y_true, y_prob),
        "auprc": metric_or_none(average_precision_score, y_true, y_prob),
    }


def scan_thresholds(targets, probs, masks):
    rows = [compute_metrics(targets, probs, masks, float(t)) for t in THRESHOLDS]
    return pd.DataFrame(rows)


def select_threshold_from_scan(scan_df: pd.DataFrame):
    recall95 = scan_df[scan_df["recall"] >= 0.95]
    recall98 = scan_df[scan_df["recall"] >= 0.98]
    if not recall95.empty:
        chosen = recall95.sort_values(
            ["precision", "false_negatives", "f1", "false_positives", "threshold"],
            ascending=[False, True, False, True, True],
        ).iloc[0]
        reason = "validation recall >= 0.95 with highest precision"
    else:
        chosen = scan_df.sort_values(
            ["recall", "precision", "false_negatives", "f1"],
            ascending=[False, False, True, False],
        ).iloc[0]
        reason = "no configuration reached validation recall >= 0.95, so highest recall then highest precision was selected"
    recall98_best = None
    if not recall98.empty:
        recall98_best = recall98.sort_values(
            ["precision", "false_negatives", "f1", "false_positives", "threshold"],
            ascending=[False, True, False, True, True],
        ).iloc[0].to_dict()
    return chosen.to_dict(), recall98_best, reason


def compare_selection(candidate, incumbent):
    if incumbent is None:
        return True
    cand_ok = candidate["metrics"]["recall"] >= 0.95
    inc_ok = incumbent["metrics"]["recall"] >= 0.95
    if cand_ok and not inc_ok:
        return True
    if inc_ok and not cand_ok:
        return False
    if cand_ok and inc_ok:
        cand_key = (
            candidate["metrics"]["precision"],
            -candidate["metrics"]["false_negatives"],
            candidate["metrics"]["f1"],
            -candidate["metrics"]["false_positives"],
        )
        inc_key = (
            incumbent["metrics"]["precision"],
            -incumbent["metrics"]["false_negatives"],
            incumbent["metrics"]["f1"],
            -incumbent["metrics"]["false_positives"],
        )
        return cand_key > inc_key
    cand_key = (
        candidate["metrics"]["recall"],
        candidate["metrics"]["precision"],
        -candidate["metrics"]["false_negatives"],
        candidate["metrics"]["f1"],
        -candidate["metrics"]["false_positives"],
    )
    inc_key = (
        incumbent["metrics"]["recall"],
        incumbent["metrics"]["precision"],
        -incumbent["metrics"]["false_negatives"],
        incumbent["metrics"]["f1"],
        -incumbent["metrics"]["false_positives"],
    )
    return cand_key > inc_key


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


def write_predictions(df, preds, threshold, output_path, loss_variant):
    out = df[["dataset_split", "dialogue_id", "turn_id", "speaker_role", "text"]].copy()
    out["target_confirm"] = preds["targets"].astype(int)
    out["confirm_mask"] = preds["masks"].astype(int)
    out["confirm_prob"] = preds["probs"]
    out["confirm_pred_fixed_0_5"] = (preds["probs"] >= FIXED_THRESHOLD).astype(int)
    out["confirm_pred_high_recall"] = (preds["probs"] >= threshold).astype(int)
    out["selected_threshold"] = float(threshold)
    out["loss_variant"] = loss_variant
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def build_loss(variant_name, pos_weight_value):
    alpha_pos = pos_weight_value / (1.0 + pos_weight_value)
    if variant_name == "A_standard_bce":
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32),
            reduction="none",
        )
        metadata = {"loss_type": "bce", "pos_weight": pos_weight_value}
    elif variant_name == "B_pos_weight_x2":
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value * 2.0], dtype=torch.float32),
            reduction="none",
        )
        metadata = {"loss_type": "bce", "pos_weight": pos_weight_value * 2.0}
    elif variant_name == "C_pos_weight_x4":
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value * 4.0], dtype=torch.float32),
            reduction="none",
        )
        metadata = {"loss_type": "bce", "pos_weight": pos_weight_value * 4.0}
    elif variant_name == "D_focal_gamma2":
        criterion = FocalLossWithLogits(alpha_pos=alpha_pos, gamma=2.0)
        metadata = {"loss_type": "focal", "alpha_pos": alpha_pos, "gamma": 2.0}
    else:
        raise ValueError(variant_name)
    return criterion, metadata


def criterion_to_device(criterion, device):
    if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
        criterion.pos_weight = criterion.pos_weight.to(device)
    return criterion.to(device) if hasattr(criterion, "to") else criterion


def train_variant(variant_name, loss_metadata, args, train_df, val_df, test_df, tokenizer, device, batch_size, tb_dir):
    set_seed(args.seed)
    model = ConfirmModel(args.model_name, dropout=args.dropout).to(device)
    if args.gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    criterion, _ = build_loss(variant_name, loss_metadata["base_pos_weight"])
    criterion = criterion_to_device(criterion, device)

    train_ds = ConfirmDataset(train_df, tokenizer, args.max_length)
    val_ds = ConfirmDataset(val_df, tokenizer, args.max_length)
    test_ds = ConfirmDataset(test_df, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_scalar("run/seed", args.seed, 0)
    writer.add_scalar("run/batch_size", batch_size, 0)
    writer.add_scalar("run/learning_rate", args.learning_rate, 0)
    writer.add_scalar("run/weight_decay", args.weight_decay, 0)
    writer.add_scalar("run/dropout", args.dropout, 0)
    writer.add_scalar("run/base_pos_weight", loss_metadata["base_pos_weight"], 0)

    variant_rows = []
    best_for_variant = None
    best_state = None
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, lr = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            val_preds = collect_predictions(model, val_loader, device, criterion)
            val_fixed = compute_metrics(val_preds["targets"], val_preds["probs"], val_preds["masks"], FIXED_THRESHOLD)
            scan_df = scan_thresholds(val_preds["targets"], val_preds["probs"], val_preds["masks"])
            selected_metrics, recall98_metrics, selection_reason = select_threshold_from_scan(scan_df)
            row = {
                "variant": variant_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_preds["loss"],
                "validation_auroc": val_fixed["auroc"],
                "validation_auprc": val_fixed["auprc"],
                "validation_fixed_precision": val_fixed["precision"],
                "validation_fixed_recall": val_fixed["recall"],
                "validation_fixed_f1": val_fixed["f1"],
                "selected_threshold": selected_metrics["threshold"],
                "selected_validation_precision": selected_metrics["precision"],
                "selected_validation_recall": selected_metrics["recall"],
                "selected_validation_f1": selected_metrics["f1"],
                "selected_validation_false_positives": selected_metrics["false_positives"],
                "selected_validation_false_negatives": selected_metrics["false_negatives"],
                "validation_recall_ge_0_98_threshold": None if recall98_metrics is None else recall98_metrics["threshold"],
                "validation_recall_ge_0_98_precision": None if recall98_metrics is None else recall98_metrics["precision"],
                "learning_rate": lr,
                "selection_reason": selection_reason,
            }
            variant_rows.append(row)
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/validation", val_preds["loss"], epoch)
            if val_fixed["auroc"] is not None:
                writer.add_scalar("metrics/validation_auroc_fixed_0_5", val_fixed["auroc"], epoch)
            if val_fixed["auprc"] is not None:
                writer.add_scalar("metrics/validation_auprc_fixed_0_5", val_fixed["auprc"], epoch)
            writer.add_scalar("metrics/validation_fixed_precision", val_fixed["precision"], epoch)
            writer.add_scalar("metrics/validation_fixed_recall", val_fixed["recall"], epoch)
            writer.add_scalar("metrics/validation_fixed_f1", val_fixed["f1"], epoch)
            writer.add_scalar("metrics/validation_selected_precision", selected_metrics["precision"], epoch)
            writer.add_scalar("metrics/validation_selected_recall", selected_metrics["recall"], epoch)
            writer.add_scalar("metrics/validation_selected_f1", selected_metrics["f1"], epoch)
            writer.add_scalar("metrics/validation_selected_false_positives", selected_metrics["false_positives"], epoch)
            writer.add_scalar("metrics/validation_selected_false_negatives", selected_metrics["false_negatives"], epoch)
            writer.add_scalar("metrics/selected_threshold", selected_metrics["threshold"], epoch)
            writer.add_scalar("run/learning_rate_step", lr, epoch)
            candidate = {
                "variant": variant_name,
                "epoch": epoch,
                "metrics": selected_metrics,
                "val_fixed": val_fixed,
                "val_preds": val_preds,
                "batch_size": batch_size,
                "selection_reason": selection_reason,
            }
            if compare_selection(candidate, best_for_variant):
                best_for_variant = candidate
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"{variant_name} epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} | "
                f"val_selected_recall={selected_metrics['recall']:.6f} | "
                f"val_selected_precision={selected_metrics['precision']:.6f} | "
                f"threshold={selected_metrics['threshold']:.3f}",
                flush=True,
            )
    finally:
        writer.close()

    if best_for_variant is None:
        raise RuntimeError(f"No valid checkpoint found for {variant_name}")

    model.load_state_dict(best_state)
    val_preds = collect_predictions(model, val_loader, device, criterion)
    test_preds = collect_predictions(model, test_loader, device, criterion)
    selected_threshold = float(best_for_variant["metrics"]["threshold"])
    val_selected = compute_metrics(val_preds["targets"], val_preds["probs"], val_preds["masks"], selected_threshold)
    test_selected = compute_metrics(test_preds["targets"], test_preds["probs"], test_preds["masks"], selected_threshold)
    val_fixed = compute_metrics(val_preds["targets"], val_preds["probs"], val_preds["masks"], FIXED_THRESHOLD)
    test_fixed = compute_metrics(test_preds["targets"], test_preds["probs"], test_preds["masks"], FIXED_THRESHOLD)
    train_preds = collect_predictions(model, train_loader, device, criterion=None)
    val_preds_no_loss = collect_predictions(model, val_loader, device, criterion=None)
    test_preds_no_loss = collect_predictions(model, test_loader, device, criterion=None)

    return {
        "variant": variant_name,
        "loss_metadata": loss_metadata,
        "batch_size": batch_size,
        "model": model,
        "state_dict": best_state,
        "epoch_rows": variant_rows,
        "best_epoch": best_for_variant["epoch"],
        "selected_threshold": selected_threshold,
        "selection_reason": best_for_variant["selection_reason"],
        "validation_selected": val_selected,
        "validation_fixed": val_fixed,
        "test_selected": test_selected,
        "test_fixed": test_fixed,
        "train_preds": train_preds,
        "val_preds": val_preds_no_loss,
        "test_preds": test_preds_no_loss,
    }


def metric_line(label, metrics):
    return (
        f"{label}: precision={metrics['precision']:.6f}, recall={metrics['recall']:.6f}, "
        f"f1={metrics['f1']:.6f}, auroc={metrics['auroc']}, auprc={metrics['auprc']}, "
        f"fp={metrics['false_positives']}, fn={metrics['false_negatives']}, threshold={metrics['threshold']:.6f}"
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    paths = output_paths(args.only_variant)

    if (
        paths["model_dir"].exists()
        or paths["result_txt"].exists()
        or paths["summary_txt"].exists()
        or paths["comparison_txt"].exists()
    ):
        raise FileExistsError("Refusing to overwrite existing high-recall Confirm outputs.")

    start = time.time()
    train_df = read_targets(args.train_csv)
    val_df = read_targets(args.val_csv)
    test_df = read_targets(args.test_csv)

    checks = {
        "target_confirm_exists": TARGET_COL in train_df.columns and TARGET_COL in val_df.columns and TARGET_COL in test_df.columns,
        "confirm_mask_exists": MASK_COL in train_df.columns and MASK_COL in val_df.columns and MASK_COL in test_df.columns,
        "splits_separate": set(train_df["dataset_split"]) == {"train"} and set(val_df["dataset_split"]) == {"validation"} and set(test_df["dataset_split"]) == {"test"},
        "no_nan_labels": bool(train_df[TARGET_COL].notna().all() and val_df[TARGET_COL].notna().all() and test_df[TARGET_COL].notna().all()),
        "binary_labels": bool(set(train_df[TARGET_COL].unique()).issubset({0, 1}) and set(val_df[TARGET_COL].unique()).issubset({0, 1}) and set(test_df[TARGET_COL].unique()).issubset({0, 1})),
        "test_not_used_for_selection": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Validation checks failed: {checks}")

    pos_count, neg_count, base_pos_weight = pos_weight_from_train(train_df)
    loss_variants = {
        "A_standard_bce": {"base_pos_weight": base_pos_weight},
        "B_pos_weight_x2": {"base_pos_weight": base_pos_weight},
        "C_pos_weight_x4": {"base_pos_weight": base_pos_weight},
        "D_focal_gamma2": {"base_pos_weight": base_pos_weight},
    }
    variant_names = [args.only_variant] if args.only_variant is not None else list(loss_variants.keys())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attempted_batch_sizes = [args.batch_size]
    for candidate in [16, 8]:
        if candidate < args.batch_size and candidate not in attempted_batch_sizes:
            attempted_batch_sizes.append(candidate)

    all_results = []
    used_batch_size = None
    for batch_size in attempted_batch_sizes:
        try:
            print(f"Trying batch_size={batch_size}", flush=True)
            used_batch_size = batch_size
            for variant_name in variant_names:
                criterion, metadata = build_loss(variant_name, base_pos_weight)
                del criterion
                print(f"Training {variant_name} with settings {metadata}", flush=True)
                result = train_variant(
                    variant_name=variant_name,
                    loss_metadata={"base_pos_weight": base_pos_weight, **metadata},
                    args=args,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    tokenizer=tokenizer,
                    device=device,
                    batch_size=batch_size,
                    tb_dir=paths["tb_dir"],
                )
                all_results.append(result)
            break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and batch_size != attempted_batch_sizes[-1]:
                print(f"OOM at batch_size={batch_size}; retrying smaller batch size.", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                all_results = []
                continue
            raise

    best_result = None
    for result in all_results:
        candidate = {"metrics": result["validation_selected"]}
        if best_result is None or compare_selection(candidate, {"metrics": best_result["validation_selected"]}):
            best_result = result
    if best_result is None:
        raise RuntimeError("No successful high-recall Confirm training run completed.")

    paths["model_dir"].mkdir(parents=True, exist_ok=False)
    paths["result_txt"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_txt"].parent.mkdir(parents=True, exist_ok=True)
    paths["comparison_txt"].parent.mkdir(parents=True, exist_ok=True)
    paths["pred_dir"].mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": args.model_name,
            "task": "confirm_high_recall",
            "target_col": TARGET_COL,
            "mask_col": MASK_COL,
            "seed": args.seed,
            "batch_size": best_result["batch_size"],
            "precision_dtype": "fp32",
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "gradient_checkpointing": args.gradient_checkpointing,
            "selected_variant": best_result["variant"],
            "selected_validation_threshold": best_result["selected_threshold"],
            "validation_selected_metrics": best_result["validation_selected"],
            "test_selected_metrics": best_result["test_selected"],
            "loss_metadata": best_result["loss_metadata"],
            "state_dict": best_result["state_dict"],
        },
        paths["model_dir"] / "best_model.pt",
    )
    tokenizer.save_pretrained(paths["model_dir"])

    write_predictions(train_df, best_result["train_preds"], best_result["selected_threshold"], paths["pred_dir"] / "seed_1_train_predictions.csv", best_result["variant"])
    write_predictions(val_df, best_result["val_preds"], best_result["selected_threshold"], paths["pred_dir"] / "seed_1_validation_predictions.csv", best_result["variant"])
    write_predictions(test_df, best_result["test_preds"], best_result["selected_threshold"], paths["pred_dir"] / "seed_1_test_predictions.csv", best_result["variant"])

    duration = time.time() - start
    report_lines = [
        "CRADLE Single-Turn Confirm High-Recall Training",
        "===============================================",
        "",
        "Validation Checks",
        "-----------------",
    ]
    for key, value in checks.items():
        report_lines.append(f"{key}: {value}")
    report_lines.extend(
        [
            f"train_rows: {len(train_df)}",
            f"validation_rows: {len(val_df)}",
            f"test_rows: {len(test_df)}",
            f"masked_train_rows: {int(train_df[MASK_COL].astype(float).sum())}",
            f"masked_validation_rows: {int(val_df[MASK_COL].astype(float).sum())}",
            f"masked_test_rows: {int(test_df[MASK_COL].astype(float).sum())}",
            f"train_positive_count: {pos_count}",
            f"train_negative_count: {neg_count}",
            f"standard_pos_weight: {base_pos_weight}",
            f"used_batch_size: {used_batch_size}",
            "precision_dtype: fp32",
            "",
            "Loss Variants",
            "-------------",
        ]
    )
    for variant_name in variant_names:
        _, metadata = build_loss(variant_name, base_pos_weight)
        report_lines.append(f"{variant_name}: {json.dumps(metadata, sort_keys=True)}")
    report_lines.extend(["", "Epoch Metrics", "-------------"])
    for result in all_results:
        for row in result["epoch_rows"]:
            report_lines.append(json.dumps(row, sort_keys=True))
    report_lines.extend(
        [
            "",
            f"selected_variant: {best_result['variant']}",
            f"selected_best_epoch: {best_result['best_epoch']}",
            f"selected_validation_threshold: {best_result['selected_threshold']}",
            f"selection_reason: {best_result['selection_reason']}",
            "",
            metric_line("Validation fixed 0.5", best_result["validation_fixed"]),
            metric_line("Validation selected threshold", best_result["validation_selected"]),
            metric_line("Test fixed 0.5", best_result["test_fixed"]),
            metric_line("Test selected threshold", best_result["test_selected"]),
            "",
            f"training_duration_seconds: {duration:.2f}",
            f"checkpoint_path: {paths['model_dir'] / 'best_model.pt'}",
            f"train_prediction_path: {paths['pred_dir'] / 'seed_1_train_predictions.csv'}",
            f"validation_prediction_path: {paths['pred_dir'] / 'seed_1_validation_predictions.csv'}",
            f"test_prediction_path: {paths['pred_dir'] / 'seed_1_test_predictions.csv'}",
            "run_status: complete",
        ]
    )
    paths["result_txt"].write_text("\n".join(report_lines) + "\n")

    improvement_fn = PREVIOUS_METRICS["test_false_negatives"] - best_result["test_selected"]["false_negatives"]
    summary_lines = [
        "CRADLE High-Recall Confirm Summary",
        "==================================",
        f"Selected variant: {best_result['variant']}",
        f"Selected validation threshold: {best_result['selected_threshold']}",
        f"Validation recall: {best_result['validation_selected']['recall']}",
        f"Validation precision: {best_result['validation_selected']['precision']}",
        f"Validation F1: {best_result['validation_selected']['f1']}",
        f"Validation AUROC: {best_result['validation_selected']['auroc']}",
        f"Validation AUPRC: {best_result['validation_selected']['auprc']}",
        f"Test recall: {best_result['test_selected']['recall']}",
        f"Test precision: {best_result['test_selected']['precision']}",
        f"Test F1: {best_result['test_selected']['f1']}",
        f"Test AUROC: {best_result['test_selected']['auroc']}",
        f"Test AUPRC: {best_result['test_selected']['auprc']}",
        f"Test false negatives: {best_result['test_selected']['false_negatives']}",
        f"Test false positives: {best_result['test_selected']['false_positives']}",
        "",
        "Final Beginner-Friendly Summary",
        f"1. Which loss variants were trained? {json.dumps(variant_names)}.",
        f"2. Which variant was selected? {best_result['variant']}.",
        f"3. What validation threshold was selected? {best_result['selected_threshold']}.",
        f"4. What was validation recall? {best_result['validation_selected']['recall']}.",
        f"5. What was validation precision? {best_result['validation_selected']['precision']}.",
        f"6. What was test recall? {best_result['test_selected']['recall']}.",
        f"7. What was test precision? {best_result['test_selected']['precision']}.",
        f"8. How many test Confirm false negatives remain? {best_result['test_selected']['false_negatives']}.",
        f"9. How many test false positives were created? {best_result['test_selected']['false_positives']}.",
        f"10. Did this improve over the old Confirm model? {'Yes' if best_result['test_selected']['false_negatives'] < PREVIOUS_METRICS['test_false_negatives'] else 'No'}; false_negative_change={improvement_fn}.",
        f"11. Should we plug this model into the progression/policy pipeline next? {'Yes, if the recall gain is worth the added false positives.' if best_result['test_selected']['false_negatives'] < PREVIOUS_METRICS['test_false_negatives'] else 'Not yet.'}",
        "12. What is the single next implementation step? Evaluate whether the new false-positive burden is acceptable in the progression and policy pipeline before swapping models.",
    ]
    paths["summary_txt"].write_text("\n".join(summary_lines) + "\n")

    comparison_lines = [
        "Original vs High-Recall Confirm Comparison",
        "==========================================",
        "",
        f"Previous validation AUROC: {PREVIOUS_METRICS['validation_auroc']}",
        f"New validation AUROC: {best_result['validation_selected']['auroc']}",
        f"Validation AUROC change: {None if best_result['validation_selected']['auroc'] is None else best_result['validation_selected']['auroc'] - PREVIOUS_METRICS['validation_auroc']}",
        f"Previous validation AUPRC: {PREVIOUS_METRICS['validation_auprc']}",
        f"New validation AUPRC: {best_result['validation_selected']['auprc']}",
        f"Validation AUPRC change: {None if best_result['validation_selected']['auprc'] is None else best_result['validation_selected']['auprc'] - PREVIOUS_METRICS['validation_auprc']}",
        f"Previous test AUROC: {PREVIOUS_METRICS['test_auroc']}",
        f"New test AUROC: {best_result['test_selected']['auroc']}",
        f"Test AUROC change: {None if best_result['test_selected']['auroc'] is None else best_result['test_selected']['auroc'] - PREVIOUS_METRICS['test_auroc']}",
        f"Previous test AUPRC: {PREVIOUS_METRICS['test_auprc']}",
        f"New test AUPRC: {best_result['test_selected']['auprc']}",
        f"Test AUPRC change: {None if best_result['test_selected']['auprc'] is None else best_result['test_selected']['auprc'] - PREVIOUS_METRICS['test_auprc']}",
        f"Previous tuned threshold: {PREVIOUS_METRICS['tuned_threshold']}",
        f"New selected threshold: {best_result['selected_threshold']}",
        f"Previous test recall: {PREVIOUS_METRICS['test_recall']}",
        f"New test recall: {best_result['test_selected']['recall']}",
        f"Recall change: {best_result['test_selected']['recall'] - PREVIOUS_METRICS['test_recall']}",
        f"Previous test false negatives: {PREVIOUS_METRICS['test_false_negatives']}",
        f"New test false negatives: {best_result['test_selected']['false_negatives']}",
        f"False negative reduction: {PREVIOUS_METRICS['test_false_negatives'] - best_result['test_selected']['false_negatives']}",
        f"New test false positives: {best_result['test_selected']['false_positives']}",
        "",
        "Analysis",
        f"Did high-recall training reduce Confirm false negatives? {'Yes' if improvement_fn > 0 else 'No'}.",
        f"How many false negatives remain? {best_result['test_selected']['false_negatives']}.",
        f"How many false positives were introduced? {best_result['test_selected']['false_positives']}.",
        f"Did recall improve enough to help the escalation policy? {'Likely yes' if best_result['test_selected']['recall'] > PREVIOUS_METRICS['test_recall'] else 'No'}.",
        f"Did AUROC/AUPRC change compared with the original Confirm model? val_auprc_change={None if best_result['validation_selected']['auprc'] is None else best_result['validation_selected']['auprc'] - PREVIOUS_METRICS['validation_auprc']}, test_auprc_change={None if best_result['test_selected']['auprc'] is None else best_result['test_selected']['auprc'] - PREVIOUS_METRICS['test_auprc']}.",
        f"Is the model over-predicting Confirm? {'Possibly' if best_result['test_selected']['false_positives'] > 200 else 'Not obviously'}.",
        f"Should this high-recall Confirm model replace the original Confirm model in the progression/policy pipeline? {'Possibly, pending downstream policy tradeoff checks.' if improvement_fn > 0 else 'No.'}",
    ]
    paths["comparison_txt"].write_text("\n".join(comparison_lines) + "\n")

    print(f"Saved checkpoint to {paths['model_dir'] / 'best_model.pt'}", flush=True)
    print(f"Saved detailed report to {paths['result_txt']}", flush=True)
    print(f"Saved summary to {paths['summary_txt']}", flush=True)
    print(f"Saved comparison to {paths['comparison_txt']}", flush=True)
    print(f"Saved prediction files to {paths['pred_dir']}", flush=True)


if __name__ == "__main__":
    main()
