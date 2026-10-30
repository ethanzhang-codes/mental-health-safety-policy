import argparse
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

from module1_benchmark_config import (
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_LEN,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_WARMUP_FRACTION,
    DEFAULT_WEIGHT_DECAY,
    LABEL_NAMES,
    prediction_dir,
    result_path,
    model_save_dir,
    tensorboard_log_dir,
)
from module1_benchmark_utils import (
    EncoderSeverityClassifier,
    MentalHealthSeverityDataset,
    clean_cuda_memory,
    collect_calibrated_metrics,
    collect_prediction_metrics,
    compute_weights,
    device_summary,
    elapsed_seconds,
    ensure_no_completed_run,
    get_or_create_fixed_splits,
    load_clean_dataset,
    make_split_dataframes,
    peak_gpu_memory_mb,
    reset_peak_gpu_memory,
    save_test_predictions,
    set_training_seed,
    write_complete_result,
    write_status_result,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an isolated Module 1 severity encoder benchmark."
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--encoder_short", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--precision_dtype",
        choices=["fp32", "bf16", "fp16", "auto"],
        default="fp32",
        help="Numerical precision for model weights. 'auto' preserves checkpoint default behavior.",
    )
    parser.add_argument(
        "--diagnose_batches",
        type=int,
        default=0,
        help="Run only this many training batches with finite-value diagnostics, then stop.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def torch_dtype_from_precision(precision_dtype):
    if precision_dtype == "fp32":
        return torch.float32
    if precision_dtype == "bf16":
        return torch.bfloat16
    if precision_dtype == "fp16":
        return torch.float16
    return None


def apply_precision(model, precision_dtype, device):
    model = model.to(device)
    if precision_dtype == "fp32":
        model = model.float()
    elif precision_dtype == "bf16":
        model = model.to(dtype=torch.bfloat16)
    elif precision_dtype == "fp16":
        model = model.half()
    return model


def first_parameter_dtype(module):
    return next(module.parameters()).dtype


def finite_status(name, tensor):
    is_finite = torch.isfinite(tensor).all().item()
    print(f"{name}: dtype={tensor.dtype}, shape={tuple(tensor.shape)}, finite={is_finite}")
    return is_finite


def evaluate(model, loader, criterion, device, collect_logits=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits.float(), labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if collect_logits:
                all_logits.append(logits.cpu().numpy())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)
    macro_f1 = macro_f1_score(labels_np, preds_np)
    logits_np = np.concatenate(all_logits, axis=0) if collect_logits else None
    return total_loss / max(len(loader), 1), macro_f1, preds_np, labels_np, logits_np


def macro_f1_score(labels, preds):
    from sklearn.metrics import f1_score

    return float(f1_score(labels, preds, average="macro"))


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits.float(), labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    return total_loss / max(len(loader), 1), macro_f1_score(
        np.array(all_labels), np.array(all_preds)
    )


def run_dry_run(args, device):
    print("Dry run only: no training will be started.")
    print(f"Model: {args.model_name}")
    print(f"Encoder short name: {args.encoder_short}")
    print(f"Batch size: {args.batch_size}")
    print(f"Precision dtype: {args.precision_dtype}")
    print(f"Device: {device}")
    print(f"GPU: {device_summary()['gpu']}")

    print("Loading config...")
    config = AutoConfig.from_pretrained(args.model_name)
    print(f"hidden_size: {config.hidden_size}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading encoder and classifier...")
    model = EncoderSeverityClassifier(
        args.model_name,
        dropout=DEFAULT_DROPOUT,
        torch_dtype=torch_dtype_from_precision(args.precision_dtype),
    )
    model = apply_precision(model, args.precision_dtype, device)
    print(f"Encoder parameter dtype: {first_parameter_dtype(model.encoder)}")
    print(f"Classifier parameter dtype: {first_parameter_dtype(model.classifier)}")

    sample_texts = [
        "I feel stressed but I am safe right now.",
        "I need help because my feelings are getting intense.",
    ]
    while len(sample_texts) < args.batch_size:
        sample_texts.extend(sample_texts)
    sample_texts = sample_texts[: args.batch_size]

    encoded = tokenizer(
        sample_texts,
        max_length=DEFAULT_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    expected_shape = (args.batch_size, 3)
    print(f"Logits dtype: {logits.dtype}")
    print(f"Classifier output shape: {tuple(logits.shape)}")
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Expected classifier output shape {expected_shape}, got {tuple(logits.shape)}"
        )

    print("Dry run succeeded. The model can load and produce Low/Medium/High logits.")


def validate_labels(df, name):
    labels = df["risk_severity"].values
    unique = sorted(set(labels.tolist()))
    print(f"{name} labels unique values: {unique}")
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError(f"{name} contains labels outside [0, 1, 2]")
    if np.isnan(labels.astype(float)).any():
        raise ValueError(f"{name} contains NaN labels")


def gradient_norm_and_finiteness(model):
    total_sq_norm = 0.0
    all_finite = True
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        all_finite = all_finite and torch.isfinite(grad).all().item()
        total_sq_norm += grad.float().norm(2).item() ** 2
    return total_sq_norm ** 0.5, all_finite


def run_diagnostics(args, model, loader, criterion, optimizer, device):
    print("\nDiagnostic mode only: no full training will be started.")
    print("Precision behavior:")
    print("- torch.autocast: not used")
    print("- GradScaler: not used")
    print(f"- requested precision_dtype: {args.precision_dtype}")
    print(f"- encoder parameter dtype: {first_parameter_dtype(model.encoder)}")
    print(f"- classifier parameter dtype: {first_parameter_dtype(model.classifier)}")
    print(f"- learning rate: {DEFAULT_LEARNING_RATE}")

    model.train()
    checked = 0
    for batch_idx, batch in enumerate(loader):
        if checked >= args.diagnose_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        print(f"\nDiagnostic batch {batch_idx}")
        print(f"labels: {labels.detach().cpu().tolist()}")
        print(f"label min/max: {int(labels.min().item())}/{int(labels.max().item())}")
        print(f"input_ids shape: {tuple(input_ids.shape)}, dtype={input_ids.dtype}")
        print(f"attention_mask shape: {tuple(attention_mask.shape)}, dtype={attention_mask.dtype}")
        print(f"CrossEntropyLoss target shape: {tuple(labels.shape)}")

        if not torch.isin(labels, torch.tensor([0, 1, 2], device=device)).all().item():
            print("Invalid labels detected. Stopping diagnostics.")
            return False

        optimizer.zero_grad()
        first_token = model.first_token_embedding(input_ids, attention_mask)
        hidden_finite = finite_status("first-token hidden state", first_token)
        classifier_input = first_token.to(model.classifier[0].weight.dtype)
        classifier_input = model.dropout(classifier_input)
        logits = model.classifier(classifier_input)
        logits_finite = finite_status("classifier logits", logits)
        print(f"CrossEntropyLoss input shape: {tuple(logits.shape)}")

        loss = criterion(logits.float(), labels)
        loss_finite = torch.isfinite(loss).item()
        print(f"loss: dtype={loss.dtype}, value={loss.item()}, finite={loss_finite}")

        if hidden_finite and logits_finite and loss_finite:
            loss.backward()
            grad_norm, grads_finite = gradient_norm_and_finiteness(model)
            print(f"gradient norm: {grad_norm}, gradients finite={grads_finite}")
            if not grads_finite:
                print("First non-finite value appears in gradients.")
                return False
            optimizer.step()
        else:
            if not hidden_finite:
                print("First non-finite value appears in encoder first-token hidden state.")
            elif not logits_finite:
                print("First non-finite value appears in classifier logits.")
            else:
                print("First non-finite value appears in CrossEntropyLoss.")
            return False

        checked += 1

    print(f"\nDiagnostics finished: {checked} batch(es) checked without non-finite values.")
    return True


def main():
    args = parse_args()
    set_training_seed(args.seed)
    clean_cuda_memory()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dry_run:
        run_dry_run(args, device)
        return

    save_path = model_save_dir(args.encoder_short, args.batch_size, args.seed)
    result_file = result_path(args.encoder_short, args.batch_size, args.seed)
    pred_dir = prediction_dir(args.encoder_short, args.batch_size, args.seed)
    tb_log_dir = tensorboard_log_dir(args.encoder_short, args.batch_size, args.seed)

    metadata = {
        "model_name": args.model_name,
        "encoder_short": args.encoder_short,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "batch_size": args.batch_size,
        "precision_dtype": args.precision_dtype,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "dropout": DEFAULT_DROPOUT,
        "epochs": DEFAULT_EPOCHS,
        "max_len": DEFAULT_MAX_LEN,
        "save_path": str(save_path),
        "tb_log_dir": str(tb_log_dir),
    }

    try:
        ensure_no_completed_run(result_file, save_path)
    except FileExistsError as exc:
        print(str(exc))
        sys.exit(1)

    start_time = time.time()

    try:
        reset_peak_gpu_memory()

        df = load_clean_dataset()
        train_indices, val_indices, test_indices, split_checksum, split_source = (
            get_or_create_fixed_splits(df, args.split_seed)
        )
        train_df, val_df, test_df = make_split_dataframes(
            df,
            train_indices,
            val_indices,
            test_indices,
        )

        metadata.update(
            {
                "split_checksum": split_checksum,
                "split_source": split_source,
                "train_count": len(train_df),
                "val_count": len(val_df),
                "test_count": len(test_df),
            }
        )

        print(f"Split checksum: {split_checksum}")
        print(
            f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
        )
        print(f"Labels: {LABEL_NAMES}")
        print(f"Device: {device}")
        print(f"GPU: {device_summary()['gpu']}")
        print(f"Precision dtype: {args.precision_dtype}")

        validate_labels(train_df, "train")
        validate_labels(val_df, "validation")
        validate_labels(test_df, "test")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        writer.add_text("run/model_name", args.model_name)
        writer.add_text("run/encoder_short", args.encoder_short)
        writer.add_text("run/save_path", str(save_path))
        writer.add_scalar("run/seed", args.seed, 0)
        writer.add_scalar("run/split_seed", args.split_seed, 0)
        writer.add_scalar("run/batch_size", args.batch_size, 0)
        writer.add_scalar("run/learning_rate", DEFAULT_LEARNING_RATE, 0)

        train_dataset = MentalHealthSeverityDataset(
            train_df["text"].values,
            train_df["risk_severity"].values,
            tokenizer,
            DEFAULT_MAX_LEN,
        )
        val_dataset = MentalHealthSeverityDataset(
            val_df["text"].values,
            val_df["risk_severity"].values,
            tokenizer,
            DEFAULT_MAX_LEN,
        )
        test_dataset = MentalHealthSeverityDataset(
            test_df["text"].values,
            test_df["risk_severity"].values,
            tokenizer,
            DEFAULT_MAX_LEN,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=DEFAULT_NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=DEFAULT_NUM_WORKERS,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=DEFAULT_NUM_WORKERS,
        )

        class_weights = compute_weights(train_df, device)
        model = EncoderSeverityClassifier(
            args.model_name,
            dropout=DEFAULT_DROPOUT,
            torch_dtype=torch_dtype_from_precision(args.precision_dtype),
        )
        model = apply_precision(model, args.precision_dtype, device)

        print(f"Encoder hidden_size: {model.encoder.config.hidden_size}")
        print(f"Encoder parameter dtype: {first_parameter_dtype(model.encoder)}")
        print(f"Classifier parameter dtype: {first_parameter_dtype(model.classifier)}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=DEFAULT_LEARNING_RATE,
            weight_decay=DEFAULT_WEIGHT_DECAY,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        eval_criterion = nn.CrossEntropyLoss()

        if args.diagnose_batches > 0:
            run_diagnostics(args, model, train_loader, criterion, optimizer, device)
            writer.close()
            return

        total_steps = len(train_loader) * DEFAULT_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * DEFAULT_WARMUP_FRACTION),
            num_training_steps=total_steps,
        )

        best_val_f1 = -1.0
        best_epoch = -1
        epoch_metrics = []

        for epoch in range(DEFAULT_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{DEFAULT_EPOCHS}")
            train_loss, train_f1 = train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                criterion,
                device,
            )
            val_loss, val_f1, _, _, _ = evaluate(
                model,
                val_loader,
                eval_criterion,
                device,
                collect_logits=False,
            )

            print(
                f"Train Loss: {train_loss:.6f} | Train Macro-F1: {train_f1:.6f}"
            )
            print(f"Val Loss: {val_loss:.6f} | Val Macro-F1: {val_f1:.6f}")
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("loss/val", val_loss, epoch + 1)
            writer.add_scalar("macro_f1/train", train_f1, epoch + 1)
            writer.add_scalar("macro_f1/val", val_f1, epoch + 1)

            epoch_metrics.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_macro_f1": train_f1,
                    "val_loss": val_loss,
                    "val_macro_f1": val_f1,
                }
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                save_path.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path / "best_model.pt")
                tokenizer.save_pretrained(save_path)
                print(f"Saved best model for epoch {best_epoch}.")

        model.load_state_dict(torch.load(save_path / "best_model.pt"))

        _, _, _, val_labels, val_logits = evaluate(
            model,
            val_loader,
            eval_criterion,
            device,
            collect_logits=True,
        )
        _, _, _, test_labels, test_logits = evaluate(
            model,
            test_loader,
            eval_criterion,
            device,
            collect_logits=True,
        )

        metrics, test_probs, test_preds = collect_prediction_metrics(
            test_logits,
            test_labels,
        )
        calibrated_metrics = collect_calibrated_metrics(
            val_logits,
            val_labels,
            test_logits,
            test_labels,
        )
        writer.add_scalar("macro_f1/test", metrics["test_macro_f1"], DEFAULT_EPOCHS + 1)
        writer.add_scalar("accuracy/test", metrics["accuracy"], DEFAULT_EPOCHS + 1)
        writer.add_scalar("class_f1/low_risk", metrics["low_f1"], DEFAULT_EPOCHS + 1)
        writer.add_scalar("class_f1/medium_risk", metrics["medium_f1"], DEFAULT_EPOCHS + 1)
        writer.add_scalar("class_f1/high_risk", metrics["high_f1"], DEFAULT_EPOCHS + 1)
        writer.add_scalar("calibration/raw_nll", metrics["raw_test_nll"], DEFAULT_EPOCHS + 1)
        writer.add_scalar("calibration/raw_ece", metrics["raw_ece"], DEFAULT_EPOCHS + 1)
        if calibrated_metrics.get("calibrated_nll") is not None:
            writer.add_scalar(
                "calibration/calibrated_nll",
                calibrated_metrics["calibrated_nll"],
                DEFAULT_EPOCHS + 1,
            )
        if calibrated_metrics.get("calibrated_ece") is not None:
            writer.add_scalar(
                "calibration/calibrated_ece",
                calibrated_metrics["calibrated_ece"],
                DEFAULT_EPOCHS + 1,
            )
        if calibrated_metrics.get("learned_temperature") is not None:
            writer.add_scalar(
                "calibration/temperature",
                calibrated_metrics["learned_temperature"],
                DEFAULT_EPOCHS + 1,
            )
        writer.flush()
        writer.close()

        metadata.update(
            {
                "best_val_macro_f1": best_val_f1,
                "best_epoch": best_epoch,
                "training_duration_seconds": elapsed_seconds(start_time),
                "peak_gpu_memory_mb": peak_gpu_memory_mb(),
            }
        )

        pred_path = pred_dir / "test_predictions.csv"
        save_test_predictions(
            pred_path,
            test_df["text"].values,
            test_labels,
            test_preds,
            test_probs,
        )

        write_complete_result(
            result_file,
            metadata,
            epoch_metrics,
            metrics,
            calibrated_metrics,
        )

        print(f"\nRun complete. Result file: {result_file}")
        print(f"Prediction file: {pred_path}")

    except torch.cuda.OutOfMemoryError as exc:
        clean_cuda_memory()
        metadata.setdefault("split_checksum", "not_available")
        write_status_result(result_file, metadata, "OOM", str(exc))
        print(f"CUDA OOM recorded in {result_file}")
        sys.exit(2)
    except Exception as exc:
        clean_cuda_memory()
        metadata.setdefault("split_checksum", "not_available")
        write_status_result(result_file, metadata, "failed", str(exc))
        print(f"Run failed and status was written to {result_file}")
        raise


if __name__ == "__main__":
    main()
