import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from module1_benchmark_config import (
    DATA_PATH,
    DEFAULT_DROPOUT,
    LABEL_NAMES,
    NUM_CLASSES,
    SPLIT_ROOT,
)


class MentalHealthSeverityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class EncoderSeverityClassifier(nn.Module):
    def __init__(self, model_name, dropout=DEFAULT_DROPOUT, torch_dtype=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=torch_dtype,
        )
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, input_ids, attention_mask):
        first_token = self.first_token_embedding(input_ids, attention_mask)
        first_token = first_token.to(self.classifier[0].weight.dtype)
        first_token = self.dropout(first_token)
        return self.classifier(first_token)

    def first_token_embedding(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]


def set_training_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_clean_dataset(data_path=DATA_PATH):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "risk_severity"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df.copy()
    df["risk_severity"] = df["risk_severity"].astype(int)
    return df


def _checksum_array(array):
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def split_checksum(train_indices, val_indices, test_indices):
    digest = hashlib.sha256()
    for name, indices in [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ]:
        digest.update(name.encode("utf-8"))
        digest.update(str(indices.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(indices).tobytes())
    return digest.hexdigest()


def get_or_create_fixed_splits(df, split_seed):
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    train_path = SPLIT_ROOT / "train_indices.npy"
    val_path = SPLIT_ROOT / "val_indices.npy"
    test_path = SPLIT_ROOT / "test_indices.npy"
    metadata_path = SPLIT_ROOT / "split_metadata.json"

    if train_path.exists() and val_path.exists() and test_path.exists():
        train_indices = np.load(train_path)
        val_indices = np.load(val_path)
        test_indices = np.load(test_path)
        checksum = split_checksum(train_indices, val_indices, test_indices)
        return train_indices, val_indices, test_indices, checksum, "loaded_existing"

    row_indices = df.index.to_numpy()
    labels = df.loc[row_indices, "risk_severity"].to_numpy()

    train_indices, temp_indices = train_test_split(
        row_indices,
        test_size=0.2,
        random_state=split_seed,
        stratify=labels,
    )
    temp_labels = df.loc[temp_indices, "risk_severity"].to_numpy()
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=split_seed,
        stratify=temp_labels,
    )

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    np.save(train_path, train_indices)
    np.save(val_path, val_indices)
    np.save(test_path, test_indices)

    checksum = split_checksum(train_indices, val_indices, test_indices)
    metadata = {
        "split_seed": split_seed,
        "train_count": int(len(train_indices)),
        "val_count": int(len(val_indices)),
        "test_count": int(len(test_indices)),
        "train_indices_sha256": _checksum_array(train_indices),
        "val_indices_sha256": _checksum_array(val_indices),
        "test_indices_sha256": _checksum_array(test_indices),
        "combined_split_sha256": checksum,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return train_indices, val_indices, test_indices, checksum, "created_new"


def make_split_dataframes(df, train_indices, val_indices, test_indices):
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    test_df = df.loc[test_indices].copy()
    return train_df, val_df, test_df


def compute_weights(train_df, device):
    class_weights = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=train_df["risk_severity"].values,
    )
    return torch.tensor(class_weights, dtype=torch.float).to(device)


def expected_calibration_error(probs, labels, n_bins=15):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels
    ece = 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        if not np.any(in_bin):
            continue
        bin_weight = np.mean(in_bin)
        bin_acc = np.mean(accuracies[in_bin])
        bin_conf = np.mean(confidences[in_bin])
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def nll_from_logits(logits, labels, temperature=1.0):
    logits_tensor = torch.tensor(logits, dtype=torch.float32) / float(temperature)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return float(nn.CrossEntropyLoss()(logits_tensor, labels_tensor).item())


def probs_from_logits(logits, temperature=1.0):
    logits_tensor = torch.tensor(logits, dtype=torch.float32) / float(temperature)
    probs = torch.softmax(logits_tensor, dim=1).numpy()
    return probs


def fit_temperature(val_logits, val_labels):
    logits = torch.tensor(val_logits, dtype=torch.float32)
    labels = torch.tensor(val_labels, dtype=torch.long)
    log_temperature = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=0.05, max=10.0)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = torch.exp(log_temperature).detach().clamp(min=0.05, max=10.0)
    return float(temperature.item())


def collect_prediction_metrics(logits, labels):
    probs = probs_from_logits(logits)
    preds = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1, 2],
        zero_division=0,
    )
    metrics = {
        "test_macro_f1": float(f1_score(labels, preds, average="macro")),
        "accuracy": float(accuracy_score(labels, preds)),
        "raw_test_nll": nll_from_logits(logits, labels),
        "raw_ece": expected_calibration_error(probs, labels),
        "average_confidence": float(np.max(probs, axis=1).mean()),
        "classification_report": classification_report(
            labels,
            preds,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            zero_division=0,
        ),
    }

    for idx, label_name in enumerate(["low", "medium", "high"]):
        metrics[f"{label_name}_precision"] = float(precision[idx])
        metrics[f"{label_name}_recall"] = float(recall[idx])
        metrics[f"{label_name}_f1"] = float(f1[idx])
        metrics[f"{label_name}_support"] = int(support[idx])

    return metrics, probs, preds


def collect_calibrated_metrics(val_logits, val_labels, test_logits, test_labels):
    try:
        temperature = fit_temperature(val_logits, val_labels)
        calibrated_probs = probs_from_logits(test_logits, temperature=temperature)
        return {
            "learned_temperature": temperature,
            "calibrated_nll": nll_from_logits(test_logits, test_labels, temperature),
            "calibrated_ece": expected_calibration_error(calibrated_probs, test_labels),
            "calibrated_average_confidence": float(calibrated_probs.max(axis=1).mean()),
        }
    except Exception as exc:
        return {
            "learned_temperature": None,
            "calibrated_nll": None,
            "calibrated_ece": None,
            "calibrated_average_confidence": None,
            "calibration_error": str(exc),
        }


def write_status_result(path, metadata, status, error_message=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Run status: {status}",
        f"Model: {metadata.get('model_name')}",
        f"Encoder short name: {metadata.get('encoder_short')}",
        f"Seed: {metadata.get('seed')}",
        f"Split seed: {metadata.get('split_seed')}",
        f"Split checksum: {metadata.get('split_checksum')}",
        f"Batch size: {metadata.get('batch_size')}",
        f"Precision dtype: {metadata.get('precision_dtype')}",
        f"Learning rate: {metadata.get('learning_rate')}",
        f"Dropout: {metadata.get('dropout')}",
        f"Epochs: {metadata.get('epochs')}",
        f"Max sequence length: {metadata.get('max_len')}",
        f"Save path: {metadata.get('save_path')}",
    ]
    if error_message:
        lines.append(f"Error: {error_message}")
    path.write_text("\n".join(lines) + "\n")


def write_complete_result(path, metadata, epoch_metrics, metrics, calibrated_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Run status: complete",
        f"Model: {metadata['model_name']}",
        f"Encoder short name: {metadata['encoder_short']}",
        f"Seed: {metadata['seed']}",
        f"Split seed: {metadata['split_seed']}",
        f"Split checksum: {metadata['split_checksum']}",
        f"Split source: {metadata['split_source']}",
        f"Batch size: {metadata['batch_size']}",
        f"Precision dtype: {metadata['precision_dtype']}",
        f"Learning rate: {metadata['learning_rate']}",
        f"Dropout: {metadata['dropout']}",
        f"Epochs: {metadata['epochs']}",
        f"Max sequence length: {metadata['max_len']}",
        f"Train count: {metadata['train_count']}",
        f"Validation count: {metadata['val_count']}",
        f"Test count: {metadata['test_count']}",
        f"Best Val Macro-F1: {metadata['best_val_macro_f1']:.6f}",
        f"Best epoch: {metadata['best_epoch']}",
        f"Test Macro-F1: {metrics['test_macro_f1']:.6f}",
        f"Accuracy: {metrics['accuracy']:.6f}",
        f"Low Risk precision: {metrics['low_precision']:.6f}",
        f"Low Risk recall: {metrics['low_recall']:.6f}",
        f"Low Risk F1: {metrics['low_f1']:.6f}",
        f"Medium Risk precision: {metrics['medium_precision']:.6f}",
        f"Medium Risk recall: {metrics['medium_recall']:.6f}",
        f"Medium Risk F1: {metrics['medium_f1']:.6f}",
        f"High-Risk precision: {metrics['high_precision']:.6f}",
        f"High-Risk recall: {metrics['high_recall']:.6f}",
        f"High-Risk F1: {metrics['high_f1']:.6f}",
        f"Raw test NLL: {metrics['raw_test_nll']:.6f}",
        f"Raw ECE: {metrics['raw_ece']:.6f}",
        f"Average confidence: {metrics['average_confidence']:.6f}",
        f"Calibrated NLL: {format_optional_float(calibrated_metrics.get('calibrated_nll'))}",
        f"Calibrated ECE: {format_optional_float(calibrated_metrics.get('calibrated_ece'))}",
        "Calibrated average confidence: "
        f"{format_optional_float(calibrated_metrics.get('calibrated_average_confidence'))}",
        f"Learned temperature: {format_optional_float(calibrated_metrics.get('learned_temperature'))}",
        f"Training duration seconds: {metadata['training_duration_seconds']:.2f}",
        f"Peak GPU memory MB: {format_optional_float(metadata.get('peak_gpu_memory_mb'))}",
        f"Save path: {metadata['save_path']}",
        "",
        "Epoch metrics:",
    ]

    for row in epoch_metrics:
        lines.append(
            f"Epoch {row['epoch']}: "
            f"Train Loss={row['train_loss']:.6f}, "
            f"Train Macro-F1={row['train_macro_f1']:.6f}, "
            f"Val Loss={row['val_loss']:.6f}, "
            f"Val Macro-F1={row['val_macro_f1']:.6f}"
        )

    lines.extend(["", "Test classification report:", metrics["classification_report"]])

    if calibrated_metrics.get("calibration_error"):
        lines.extend(["", f"Calibration error: {calibrated_metrics['calibration_error']}"])

    path.write_text("\n".join(lines) + "\n")


def format_optional_float(value):
    if value is None:
        return "not_available"
    return f"{float(value):.6f}"


def save_test_predictions(path, texts, labels, preds, probs):
    path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(
        {
            "text": texts,
            "true_label": labels,
            "predicted_label": preds,
            "low_prob": probs[:, 0],
            "medium_prob": probs[:, 1],
            "high_prob": probs[:, 2],
            "confidence": probs.max(axis=1),
        }
    )
    output.to_csv(path, index=False)


def elapsed_seconds(start_time):
    return time.time() - start_time


def ensure_no_completed_run(result_path, save_path):
    if result_path.exists():
        text = result_path.read_text(errors="ignore")
        if "Run status: complete" in text:
            raise FileExistsError(
                f"Refusing to overwrite completed result file: {result_path}"
            )
        raise FileExistsError(f"Result file already exists: {result_path}")

    if save_path.exists() and any(save_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite model directory: {save_path}")


def device_summary():
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
        }
    return {"device": "cpu", "gpu": "None"}


def peak_gpu_memory_mb():
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_peak_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
