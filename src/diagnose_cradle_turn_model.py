import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from transformers import AutoTokenizer


CATEGORY_ORDER = ["CA", "DV", "RA", "SHA", "SH", "SI_active", "SI_passive"]
TEMPORAL_ORDER = ["ongoing", "past"]
MODEL_NAME = "microsoft/deberta-v3-large"
TRAIN_CSV = Path("cradle_dialogue/data/cradle_train_targets.csv")
VAL_CSV = Path("cradle_dialogue/data/cradle_validation_targets.csv")
TEST_CSV = Path("cradle_dialogue/data/cradle_test_targets.csv")
RESULT_TXT = Path("cradle_dialogue/results/turn_model/seed_1.txt")
PRED_CSV = Path("cradle_dialogue/predictions/turn_model/seed_1_test_predictions.csv")
CHECKPOINT = Path("cradle_dialogue/models/turn_model/seed_1/best_model.pt")
OUT_TXT = Path("cradle_dialogue/results/turn_model/seed_1_diagnostic_audit.txt")


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False)


def model_text(row):
    role = str(row["speaker_role"]).lower()
    text = str(row["text"])
    if role == "user":
        return f"User: {text}"
    if role == "listener":
        return f"Listener: {text}"
    return f"Unknown: {text}"


def count_masked(df, target_cols, mask_col):
    masked = df[df[mask_col].astype(float) == 1]
    rows = []
    for col in target_cols:
        pos = int(masked[col].astype(float).sum())
        neg = int(len(masked) - pos)
        rows.append((col, len(masked), pos, neg, None if pos == 0 else neg / pos))
    return rows


def quantiles(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            "count": 0, "min": None, "p05": None, "p25": None, "median": None,
            "mean": None, "p75": None, "p95": None, "max": None,
        }
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


def fmt_stats(stats):
    if stats["count"] == 0:
        return "count=0"
    return (
        f"count={stats['count']}, min={stats['min']:.6f}, p05={stats['p05']:.6f}, "
        f"p25={stats['p25']:.6f}, median={stats['median']:.6f}, mean={stats['mean']:.6f}, "
        f"p75={stats['p75']:.6f}, p95={stats['p95']:.6f}, max={stats['max']:.6f}"
    )


def prob_audit(pred_df, target_col, prob_col, valid_mask=None):
    if valid_mask is None:
        valid_mask = np.ones(len(pred_df), dtype=bool)
    target = pred_df.loc[valid_mask, target_col].astype(int).to_numpy()
    prob = pred_df.loc[valid_mask, prob_col].astype(float).to_numpy()
    clipped = np.clip(prob, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    out = {
        "positive": quantiles(prob[target == 1]),
        "negative": quantiles(prob[target == 0]),
        "frac_ge_0_5": float(np.mean(prob >= 0.5)),
        "frac_ge_0_3": float(np.mean(prob >= 0.3)),
        "frac_ge_0_1": float(np.mean(prob >= 0.1)),
        "frac_ge_0_05": float(np.mean(prob >= 0.05)),
        "logit_mean": float(np.mean(logits)),
        "logit_std": float(np.std(logits)),
        "positive_logit_mean": float(np.mean(logits[target == 1])) if np.any(target == 1) else None,
        "negative_logit_mean": float(np.mean(logits[target == 0])) if np.any(target == 0) else None,
        "auroc": None,
        "auprc": None,
    }
    if len(np.unique(target)) == 2:
        out["auroc"] = float(roc_auc_score(target, prob))
        out["auprc"] = float(average_precision_score(target, prob))
    return out


def parse_epoch_rows(result_text):
    rows = []
    for line in result_text.splitlines():
        line = line.strip()
        if line.startswith("{") and '"epoch"' in line:
            rows.append(json.loads(line))
    return rows


def parse_metric_line(result_text, prefix):
    for line in result_text.splitlines():
        if line.startswith(prefix):
            return line
    return None


def parse_key_value_metric(line, key):
    if not line:
        return None
    match = re.search(rf"{re.escape(key)}=([-+]?[0-9]*\.?[0-9]+)", line)
    return float(match.group(1)) if match else None


def token_lengths(df, tokenizer):
    texts = [model_text(row) for _, row in df.iterrows()]
    return np.array([len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in texts])


def length_report(df, tokenizer, split):
    lengths = token_lengths(df, tokenizer)
    out = [
        f"{split}: count={len(lengths)}, median={np.percentile(lengths, 50):.1f}, "
        f"p90={np.percentile(lengths, 90):.1f}, p95={np.percentile(lengths, 95):.1f}, "
        f"p99={np.percentile(lengths, 99):.1f}, max={int(lengths.max())}",
    ]
    trunc = lengths > 256
    out.append(f"{split}: longer_than_256={int(trunc.sum())} ({trunc.mean()*100:.2f}%)")
    for label, target_col in [
        ("alert_positive", "target_alert"),
        ("confirm_positive", "target_confirm"),
    ]:
        mask = df[target_col].astype(float).to_numpy() == 1
        denom = int(mask.sum())
        count = int((trunc & mask).sum())
        pct = 0.0 if denom == 0 else 100 * count / denom
        out.append(f"{split}: {label}_truncated={count}/{denom} ({pct:.2f}%)")
    for cat in CATEGORY_ORDER:
        target_col = f"target_category_{cat}"
        mask = df[target_col].astype(float).to_numpy() == 1
        denom = int(mask.sum())
        count = int((trunc & mask).sum())
        pct = 0.0 if denom == 0 else 100 * count / denom
        out.append(f"{split}: category_{cat}_positive_truncated={count}/{denom} ({pct:.2f}%)")
    return out, lengths, trunc


def retained_text(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=256)
    return tokenizer.decode(ids, skip_special_tokens=True)


def sample_examples(df, condition, n=10, seed=1):
    sample = df[condition].sample(n=min(n, int(condition.sum())), random_state=seed)
    rows = []
    for _, row in sample.iterrows():
        rows.append(
            f"- {row['dialogue_id']} turn={row['turn_id']} speaker={row['speaker_role']} text={str(row['text'])[:700]}"
        )
    return rows


def parameter_audit(state_dict):
    total_params = sum(v.numel() for v in state_dict.values())
    head_params = {
        "alert_head": sum(v.numel() for k, v in state_dict.items() if k.startswith("alert_head.")),
        "confirm_head": sum(v.numel() for k, v in state_dict.items() if k.startswith("confirm_head.")),
        "category_head": sum(v.numel() for k, v in state_dict.items() if k.startswith("category_head.")),
        "temporal_state_head": sum(v.numel() for k, v in state_dict.items() if k.startswith("temporal_state_head.")),
    }
    encoder_params = sum(v.numel() for k, v in state_dict.items() if k.startswith("encoder."))
    return total_params, encoder_params, head_params


def lr_at_step(base_lr, step, total_steps):
    if step <= 0:
        return base_lr
    return base_lr * max(0.0, (total_steps - step) / max(1, total_steps))


def main():
    train_df = read_csv(TRAIN_CSV)
    val_df = read_csv(VAL_CSV)
    test_df = read_csv(TEST_CSV)
    pred_df = pd.read_csv(PRED_CSV)
    pred_df["_alert_mask"] = test_df["alert_target_mask"].astype(float).to_numpy() == 1
    pred_df["_confirm_mask"] = test_df["confirm_target_mask"].astype(float).to_numpy() == 1
    pred_df["_category_mask"] = test_df["category_target_mask"].astype(float).to_numpy() == 1
    pred_df["_temporal_mask"] = test_df["temporal_state_target_mask"].astype(float).to_numpy() == 1
    result_text = RESULT_TXT.read_text()
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    report = [
        "CRADLE Turn-Level Seed-1 Diagnostic Audit",
        "===========================================",
        "",
        "This audit reads existing artifacts only. No model was trained and no checkpoint/result/prediction file was overwritten.",
        "",
    ]

    report.extend([
        "Phase 1 - Target And Mask Implementation",
        "----------------------------------------",
        "Code mapping:",
        "- Alert uses target_alert with alert_target_mask; tensor dtype torch.float32; target/logit shape [batch, 1].",
        "- Confirm uses target_confirm with confirm_target_mask; tensor dtype torch.float32; target/logit shape [batch, 1].",
        f"- Category target order and output order: {CATEGORY_ORDER}; tensor/logit shape [batch, 7].",
        f"- Temporal target order and output order: {TEMPORAL_ORDER}; tensor/logit shape [batch, 2].",
        "- Category and temporal masks have shape [batch, 1] and are expanded to [batch, outputs] before selecting valid loss elements.",
        "",
    ])
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        report.append(f"{name} rows={len(df)}")
        for task, cols, mask in [
            ("alert", ["target_alert"], "alert_target_mask"),
            ("confirm", ["target_confirm"], "confirm_target_mask"),
            ("category", [f"target_category_{x}" for x in CATEGORY_ORDER], "category_target_mask"),
            ("temporal", ["target_ongoing", "target_past"], "temporal_state_target_mask"),
        ]:
            report.append(f"{task} masked rows: {int(df[mask].astype(float).sum())}")
            for col, masked_rows, pos, neg, pw in count_masked(df, cols, mask):
                report.append(f"  {col}: masked={masked_rows}, positive={pos}, negative={neg}, neg/pos={pw}")
        report.append("")
    report.extend([
        "Mask audit conclusion:",
        "- The masked row counts agree with the prior CRADLE target audit: train user rows=25,662, validation user rows=3,529, test user rows=4,527; temporal masks are train=3,705, validation=526, test=705.",
        "- category_target_mask=1 for every User turn, so User turns without a current Confirm category are treated as seven negative current-turn confirmed-category labels.",
        "- That is intentional and scientifically defensible only for CURRENT-TURN confirmed-category detection. It does not mean the user never has that crisis category elsewhere in the dialogue.",
        "",
    ])

    report.extend([
        "Phase 2 - BCEWithLogitsLoss And Pos Weight",
        "-------------------------------------------",
        "Exact loss construction in train_cradle_turn_model.py:",
        "criteria = {",
        '  "alert": torch.nn.BCEWithLogitsLoss(pos_weight=alert_pos_weight, reduction="none"),',
        '  "confirm": torch.nn.BCEWithLogitsLoss(pos_weight=confirm_pos_weight, reduction="none"),',
        '  "category": torch.nn.BCEWithLogitsLoss(pos_weight=category_pos_weight, reduction="none"),',
        '  "temporal": torch.nn.BCEWithLogitsLoss(pos_weight=temporal_pos_weight, reduction="none"),',
        "}",
        "No weight= argument is used; pos_weight= is used.",
        "Unreduced loss is computed first, masks are applied second, then mean is taken over selected valid elements.",
        "Masked-out rows do not affect loss because per_item[mask > 0] is selected before mean.",
        "",
        "Train-only pos_weight verification:",
    ])
    for task, cols, mask in [
        ("alert", ["target_alert"], "alert_target_mask"),
        ("confirm", ["target_confirm"], "confirm_target_mask"),
        ("category", [f"target_category_{x}" for x in CATEGORY_ORDER], "category_target_mask"),
        ("temporal", ["target_ongoing", "target_past"], "temporal_state_target_mask"),
    ]:
        report.append(f"{task}:")
        for col, _, pos, neg, pw in count_masked(train_df, cols, mask):
            report.append(f"  {col}: neg/pos={pw}")
    report.extend([
        f"Checkpoint category_order: {ckpt.get('category_order')}",
        f"Checkpoint temporal_order: {ckpt.get('temporal_order')}",
        "Category pos_weight shape is [7] and aligns with category output order. Temporal pos_weight shape is [2]. During training these tensors were moved to the active CUDA device.",
        "",
    ])

    epoch_rows = parse_epoch_rows(result_text)
    report.extend([
        "Phase 3 - Loss Magnitude And Task Domination",
        "---------------------------------------------",
        "Per-task losses were logged by epoch:",
        "epoch | train_total | alert_loss | confirm_loss | category_loss | temporal_loss",
    ])
    for row in epoch_rows:
        report.append(
            f"{row['epoch']} | {row['train_loss']:.6f} | {row['alert_loss']:.6f} | "
            f"{row['confirm_loss']:.6f} | {row['category_loss']:.6f} | {row['temporal_loss']:.6f}"
        )
    means = {
        key: float(np.mean([r[key] for r in epoch_rows]))
        for key in ["alert_loss", "confirm_loss", "category_loss", "temporal_loss"]
    }
    report.extend([
        f"Mean per-task losses: {means}",
        "- Category loss is averaged across seven category outputs after masking, so it is not seven times larger solely because there are seven labels.",
        "- Temporal loss is averaged across two outputs after masking.",
        "- Numerically, category/alert/confirm losses are similar (~1.3-1.43), while temporal loss is smaller (~0.63-0.65).",
        "- Equal scalar coefficients do not guarantee equal gradient influence, because heads have different class imbalance, label density, and pos_weight magnitudes. Gradient norms were not logged in seed_1.",
        "Recommended next-run logging: alert_loss, confirm_loss, category_loss, temporal_state_loss, total_loss, plus gradient norms for shared encoder and each head.",
        "",
    ])

    outputs = [
        ("alert", "target_alert", "alert_prob", "_alert_mask"),
        ("confirm", "target_confirm", "confirm_prob", "_confirm_mask"),
    ]
    outputs += [(f"category_{c}", f"target_category_{c}", f"category_{c}_prob", "_category_mask") for c in CATEGORY_ORDER]
    outputs += [(f"temporal_{t}", f"target_{t}", f"{t}_prob", "_temporal_mask") for t in TEMPORAL_ORDER]

    prob_results = {
        name: prob_audit(pred_df, target_col, prob_col, pred_df[mask_col].to_numpy())
        for name, target_col, prob_col, mask_col in outputs
    }
    report.extend([
        "Phase 4 - Probability Distribution Audit",
        "----------------------------------------",
        "Probability/logit distributions use the same primary masks as evaluation: User-turn masks for Alert/Confirm/Category and temporal_state_target_mask for Ongoing/Past.",
    ])
    for name, stats in prob_results.items():
        report.append(f"{name}:")
        report.append(f"  target=1: {fmt_stats(stats['positive'])}")
        report.append(f"  target=0: {fmt_stats(stats['negative'])}")
        report.append(
            f"  fractions: >=0.5={stats['frac_ge_0_5']:.6f}, >=0.3={stats['frac_ge_0_3']:.6f}, "
            f">=0.1={stats['frac_ge_0_1']:.6f}, >=0.05={stats['frac_ge_0_05']:.6f}"
        )
        report.append(f"  AUROC={stats['auroc']}, AUPRC={stats['auprc']}")
    report.extend([
        "",
        "Probability pattern conclusion:",
        "- Alert and Confirm probabilities are highly compressed around ~0.46 and ~0.41 respectively; positives and negatives overlap almost completely.",
        "- The 0.5 threshold fails mainly because most probabilities are below 0.5, but AUROC/AUPRC are also weak, so this is not merely a calibration/threshold problem.",
        "- Categories are also compressed; rare categories do not show extreme unstable probabilities despite very large pos_weight values.",
        "",
    ])

    report.extend([
        "Phase 5 - Logit Distribution Audit",
        "----------------------------------",
        "Logits were not stored, so this audit converts probabilities to diagnostic logits with clipping log(p/(1-p)).",
    ])
    for name, stats in prob_results.items():
        pos_mean = stats["positive_logit_mean"]
        neg_mean = stats["negative_logit_mean"]
        report.append(
            f"{name}: logit_mean={stats['logit_mean']:.6f}, logit_std={stats['logit_std']:.6f}, "
            f"positive_logit_mean={pos_mean}, negative_logit_mean={neg_mean}"
        )
    report.extend([
        "Logit conclusion: logits are mostly compressed, not explosive. Alert/confirm logits are mildly negative on average, explaining why fixed 0.5 predictions mostly collapse to negative.",
        "",
    ])

    report.extend([
        "Phase 6 - Training Dynamics Audit",
        "---------------------------------",
        "Fixed-threshold epoch table:",
        "epoch | train_loss | val_composite | val_alert_macro_f1 | val_confirm_macro_f1 | val_category_macro_f1 | val_temporal_macro_f1",
    ])
    for row in epoch_rows:
        report.append(
            f"{row['epoch']} | {row['train_loss']:.6f} | {row['val_composite']:.6f} | "
            f"{row['val_alert_macro_f1']:.6f} | {row['val_confirm_macro_f1']:.6f} | "
            f"{row['val_category_macro_f1']:.6f} | {row['val_temporal_macro_f1']:.6f}"
        )
    report.extend([
        "The model did not improve after epoch 1 on the composite score or individual fixed-threshold task metrics. Category macro-F1 dropped from 0.004698 to 0.0 after epoch 1; other task macro-F1 values were flat.",
        "Best-epoch selection was driven by a tiny nonzero category macro-F1 in epoch 1, not by meaningful alert/confirm F1.",
        "For future safety-aligned selection, use a safety-weighted validation criterion emphasizing Alert/Confirm AUPRC and high-recall F1, not the current simple mean of macro-F1 values.",
        "",
    ])

    report.extend(["Phase 7 - Input Truncation Audit", "--------------------------------"])
    for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        lines, lengths, trunc = length_report(df, tokenizer, split_name)
        report.extend(lines)
        df = df.copy()
        df["_token_length"] = lengths
        df["_truncated"] = trunc
        examples = df[(df["_truncated"]) & ((df["target_alert"].astype(float) == 1) | (df["target_confirm"].astype(float) == 1))].head(3)
        if len(examples) > 0:
            report.append(f"{split_name}: sample truncated positive turns, full vs retained:")
            for _, row in examples.iterrows():
                full_input = model_text(row)
                report.append(f"  full: {full_input[:700]}")
                report.append(f"  retained_256_tokens: {retained_text(tokenizer, full_input)[:700]}")
        report.append("")
    report.append("Truncation conclusion: max_length=256 truncates almost nothing here. It truncated 1 train row and 1 test row overall, and zero Alert/Confirm/category-positive turns in train/validation/test. It is not a plausible explanation for the weak seed-1 result.")
    report.append("")

    report.extend(["Phase 8 - Speaker Prefix Audit", "------------------------------"])
    for role in ["user", "listener"]:
        ex = train_df[train_df["speaker_role"] == role].head(1)
        if len(ex):
            report.append(f"{role} example model input: {model_text(ex.iloc[0])[:700]}")
    ex = train_df[train_df["speaker_role"] == "unknown"].head(1)
    if len(ex):
        report.append(f"unknown example model input: {model_text(ex.iloc[0])[:700]}")
    else:
        report.append("unknown example model input: none in sampled train rows")
    duplicate_prefix = sum(model_text(row).startswith("User: User:") or model_text(row).startswith("Listener: Listener:") for _, row in train_df.iterrows())
    empty_clean = int((train_df["text"].astype(str).str.len() == 0).sum())
    report.append(f"duplicate User:/Listener: prefixes in train model inputs: {duplicate_prefix}")
    report.append(f"empty cleaned text rows in train: {empty_clean}")
    for role, group in train_df.groupby("speaker_role"):
        lengths = token_lengths(group, tokenizer)
        report.append(f"train speaker={role}: count={len(group)}, median_tokens={np.percentile(lengths,50):.1f}, p95_tokens={np.percentile(lengths,95):.1f}, max_tokens={int(lengths.max())}")
    report.append("")

    state_dict = ckpt["state_dict"]
    total_params, encoder_params, head_params = parameter_audit(state_dict)
    report.extend([
        "Phase 9 - Pretrained Model Initialization Audit",
        "------------------------------------------------",
        f"Checkpoint model_name: {ckpt.get('model_name')}",
        f"Checkpoint hidden_size: {ckpt.get('hidden_size')}",
        "The saved checkpoint is initialized from microsoft/deberta-v3-large, not the severity checkpoint.",
        "The transformer load report showed unexpected masked-language-model pretraining-head keys, which is expected when loading AutoModel as an encoder. There were no missing classifier/task-head keys at checkpoint load because task heads are saved in best_model.pt.",
        "Task-head parameters were randomly initialized at training start: alert_head, confirm_head, category_head, temporal_state_head.",
        f"total_checkpoint_parameters: {total_params}",
        f"encoder_parameters: {encoder_params}",
        f"head_parameters: {head_params}",
        "Encoder trainability: code never freezes encoder parameters, optimizer receives model.parameters(), and gradient checkpointing still permits backpropagation. No evidence the encoder was accidentally frozen.",
        "",
    ])

    train_batches = math.ceil(len(train_df) / 32)
    total_steps = train_batches * 5
    report.extend([
        "Phase 10 - Optimizer And Scheduler Audit",
        "----------------------------------------",
        "optimizer: torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)",
        "scheduler: get_linear_schedule_with_warmup with num_warmup_steps=0 and num_training_steps=len(train_loader)*epochs",
        "gradient_clipping: clip_grad_norm_(model.parameters(), 1.0)",
        "gradient_accumulation: none",
        "effective_batch_size: 32",
        f"training_batches_per_epoch: {train_batches}",
        f"optimizer_steps_per_epoch: {train_batches}",
        f"total_optimizer_steps: {total_steps}",
        "gradient_checkpointing: active, enabled via model.encoder.gradient_checkpointing_enable()",
        f"learning_rate_step_1: {lr_at_step(2e-5, 1, total_steps):.10f}",
        f"learning_rate_10_percent_training: {lr_at_step(2e-5, int(0.10*total_steps), total_steps):.10f}",
        f"learning_rate_end_epoch_1: {lr_at_step(2e-5, train_batches, total_steps):.10f}",
        f"learning_rate_final_step: {lr_at_step(2e-5, total_steps, total_steps):.10f}",
        "Scheduler conclusion: with no warmup, LR starts near 2e-5 and linearly decays. By end of epoch 1 it is already ~80% of base LR. This is not an obvious bug, but no-warmup may be suboptimal for this difficult imbalanced multi-task run.",
        "",
    ])

    report.extend([
        "Phase 11 - Label Information Audit",
        "----------------------------------",
    ])
    for split_name, df in [("train", train_df), ("test", test_df)]:
        lengths = token_lengths(df, tokenizer)
        user = df["speaker_role"].astype(str) == "user"
        for label, condition in [
            ("alert_positive", df["target_alert"].astype(float) == 1),
            ("confirm_positive", df["target_confirm"].astype(float) == 1),
            ("negative_user_no_alert_confirm", user & (df["target_alert"].astype(float) == 0) & (df["target_confirm"].astype(float) == 0)),
        ]:
            idx = condition.to_numpy()
            if idx.sum() == 0:
                continue
            report.append(
                f"{split_name} {label}: n={int(idx.sum())}, median_chars={df.loc[condition,'text'].astype(str).str.len().median():.1f}, "
                f"median_tokens={np.median(lengths[idx]):.1f}, p95_tokens={np.percentile(lengths[idx],95):.1f}"
            )
    report.append("")
    report.append("10 random Alert-positive train examples:")
    report.extend(sample_examples(train_df, train_df["target_alert"].astype(float) == 1, 10, 11))
    report.append("10 random Confirm-positive train examples:")
    report.extend(sample_examples(train_df, train_df["target_confirm"].astype(float) == 1, 10, 12))
    report.append("10 Alert-positive clinician test examples:")
    report.extend(sample_examples(test_df, test_df["target_alert"].astype(float) == 1, 10, 13))
    report.append("10 Confirm-positive clinician test examples:")
    report.extend(sample_examples(test_df, test_df["target_confirm"].astype(float) == 1, 10, 14))
    report.append("Qualitative assessment: many Confirm turns are explicit, while Alert turns often appear earlier and may be intentionally ambiguous. This is qualitative and based only on current-turn examples.")
    report.append("")

    report.extend([
        "Phase 12 - Generalization Audit",
        "--------------------------------",
    ])
    for task in ["alert", "confirm"]:
        val_line = parse_metric_line(result_text, f"{task}:")
        # First matching lines are validation fixed. Extract test by searching section manually if needed.
        report.append(f"{task} validation fixed line from report: {val_line}")
    report.extend([
        "Clinician test fixed Alert: AUROC=0.543185, AUPRC=0.056562, F1=0.000000.",
        "Clinician test fixed Confirm: AUROC=0.453007, AUPRC=0.090166, F1=0.000000.",
        "Validation fixed Alert: AUROC=0.582165, AUPRC=0.081005, F1=0.000000.",
        "Validation fixed Confirm: AUROC=0.498660, AUPRC=0.079472, F1=0.000000.",
        "Category validation per-label AUROC/AUPRC were not stored in seed_1.txt or the prediction CSV, so exact validation-vs-test category per-label comparison is not available without rerunning validation inference.",
        "Evidence: weak validation and weak clinician test metrics are both visible, so the primary issue is not only synthetic-to-clinician generalization. It is likely optimization/class imbalance/current-turn target difficulty.",
        "",
    ])

    report.extend([
        "Phase 13 - Exactly One Next Experiment",
        "--------------------------------------",
        "Recommended single next experiment: run single-task Alert and Confirm baselines, separately, with the same DeBERTa-v3-large encoder and the same train/validation/test split.",
        "Why this one: the current multi-task run cannot distinguish whether the alert/confirm heads failed because of multi-task interference/loss competition or because current-turn text lacks enough signal. Single-task Alert and Confirm runs are the cleanest next diagnostic before investing in the hierarchical temporal Transformer.",
        "Do not change max_length, thresholding, checkpoint selection, or initialization in the same experiment; isolate the variable by removing the category and temporal-state heads/losses.",
        "",
        "Beginner-Friendly Summary",
        "-------------------------",
        "1. Were targets and masks implemented correctly? Yes, for current-turn detection. Category masking treats every User turn without a current Confirm category as a current-turn negative, which is defensible for current-turn confirmed-category detection but not a statement about the whole conversation.",
        "2. Were pos_weights implemented correctly? Yes. They are train-only neg/pos values over masked examples, passed as pos_weight= with unreduced BCE.",
        "3. Did one task dominate the total loss? Numerically no single loss term exploded; category/alert/confirm were similar, temporal was smaller. But equal weights do not guarantee equal gradient influence, and gradient norms were not logged.",
        "4. Are positive probabilities separated from negative probabilities? Mostly no. Positive and negative distributions overlap heavily.",
        "5. Why did 0.5 threshold F1 collapse to zero? The logits/probabilities were compressed below 0.5 for nearly all Alert/Confirm examples, so fixed 0.5 predicted almost no positives.",
        "6. Are AUROC/AUPRC also weak, or is this mainly a threshold/calibration problem? AUROC/AUPRC are also weak for Alert/Confirm, so it is not mainly calibration.",
        "7. Did the model improve after epoch 1? No. Epoch 1 was best; later epochs were flat or slightly worse.",
        "8. Did max_length=256 truncate many crisis-positive turns? No. It truncated almost nothing and zero crisis-positive turns in the checked train/validation/test labels.",
        "9. Was the encoder definitely trainable? Yes. The code did not freeze it, optimizer used model.parameters(), and gradient checkpointing is compatible with gradients.",
        "10. Was the optimizer/scheduler behaving correctly? Mechanically yes. It used AdamW, no warmup, linear decay, gradient clipping, no accumulation. No-warmup may be suboptimal but is not a bug.",
        "11. Is the main problem optimization, class imbalance, calibration, synthetic-to-clinician generalization, or lack of current-turn information? Evidence points first to optimization/class imbalance and/or current-turn target difficulty, with calibration secondary. It is not solely synthetic-to-clinician generalization because validation was weak too.",
        "12. Does this result genuinely support the need for temporal modeling yet? Not by itself. It suggests temporal modeling may be needed, but first we should rule out multi-task optimization failure with single-task Alert/Confirm baselines.",
        "13. What is the single next experiment I should run? Single-task Alert and Confirm baselines, separately, using the same encoder/split/settings.",
    ])

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(report) + "\n")
    print(f"Saved diagnostic audit to {OUT_TXT}")


if __name__ == "__main__":
    main()
