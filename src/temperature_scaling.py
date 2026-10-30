import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--split-seed", type=int, default=42)
parser.add_argument("--model_name", type=str, default="mental/mental-roberta-base")
parser.add_argument("--save_root", type=str, default="./saved_models")
parser.add_argument("--batch_size", "--batch-size", type=int, default=32)
parser.add_argument("--output_csv", type=str, default="./outputs/ensemble_uncertainty_test_calibrated.csv")
parser.add_argument("--max-samples", type=int, default=None)
args = parser.parse_args()


MODEL_NAME = args.model_name
MODEL_DIRS = [os.path.join(args.save_root, f"seed_{seed}") for seed in range(1, 6)]
DATA_PATH = "./combined_dataset.csv"
OUTPUT_PATH = args.output_csv
MAX_LEN = 256
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
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
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class RobertaRiskClassifier(nn.Module):
    def __init__(self, model_name, num_classes=3, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)


def check_model_dirs():
    for model_dir in MODEL_DIRS:
        checkpoint_path = os.path.join(model_dir, "best_model.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")


def load_splits():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "risk_severity"])
    df = df[df["text"].str.strip() != ""]

    _, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.split_seed,
        stratify=df["risk_severity"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=args.split_seed,
        stratify=temp_df["risk_severity"],
    )

    if args.max_samples is not None:
        val_df = val_df.head(args.max_samples)
        test_df = test_df.head(args.max_samples)

    return val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_dataloader(df, tokenizer):
    dataset = MentalHealthDataset(
        df["text"].values,
        df["risk_severity"].values,
        tokenizer,
        MAX_LEN,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def collect_logits(model_dir, dataloader):
    model = RobertaRiskClassifier(MODEL_NAME, num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint_path = os.path.join(model_dir, "best_model.pt")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    all_logits = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.cat(all_logits, dim=0)


def collect_ensemble_logits(dataloader):
    logits_by_model = []
    for model_dir in MODEL_DIRS:
        print(f"Collecting logits from {model_dir}")
        logits_by_model.append(collect_logits(model_dir, dataloader))
    return torch.stack(logits_by_model, dim=0)


def learn_temperature(validation_logits, validation_labels):
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(validation_logits / temperature.clamp(min=1e-6), validation_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.detach().clamp(min=1e-6).item()


def entropy(probs):
    safe_probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(safe_probs * np.log(safe_probs), axis=-1)


def expected_calibration_error(probs, labels, n_bins=10):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels
    ece = 0.0

    for bin_idx in range(n_bins):
        lower = bin_idx / n_bins
        upper = (bin_idx + 1) / n_bins
        if bin_idx == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        if in_bin.any():
            bin_fraction = in_bin.mean()
            bin_accuracy = accuracies[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            ece += bin_fraction * abs(bin_accuracy - bin_confidence)

    return ece


def compute_metrics(logits, labels, temperature=1.0):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=1).numpy()
    labels_np = labels.numpy()
    nll = nn.CrossEntropyLoss()(scaled_logits, labels).item()
    accuracy = (probs.argmax(axis=1) == labels_np).mean()
    avg_confidence = probs.max(axis=1).mean()
    ece = expected_calibration_error(probs, labels_np)
    return {
        "nll": nll,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "ece": ece,
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print(f"NLL: {metrics['nll']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average confidence: {metrics['avg_confidence']:.4f}")
    print(f"ECE: {metrics['ece']:.4f}")


def save_calibrated_test_csv(test_df, logits_by_model, temperature):
    scaled_logits_by_model = logits_by_model / temperature
    probs_by_model = torch.softmax(scaled_logits_by_model, dim=2).numpy()
    mean_probs = probs_by_model.mean(axis=0)

    predicted_labels = mean_probs.argmax(axis=1)
    confidence = mean_probs.max(axis=1)
    epistemic_uncertainty = probs_by_model.var(axis=0).mean(axis=1)
    aleatoric_uncertainty = entropy(probs_by_model).mean(axis=0)

    output_df = pd.DataFrame({
        "text": test_df["text"].values,
        "true_label": test_df["risk_severity"].values,
        "predicted_label": predicted_labels,
        "mean_low_prob": mean_probs[:, 0],
        "mean_medium_prob": mean_probs[:, 1],
        "mean_high_prob": mean_probs[:, 2],
        "epistemic_uncertainty": epistemic_uncertainty,
        "aleatoric_uncertainty": aleatoric_uncertainty,
        "confidence": confidence,
        "temperature": temperature,
    })
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved calibrated test outputs to {OUTPUT_PATH}")


def main():
    check_model_dirs()
    os.makedirs("./outputs", exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"Model: {MODEL_NAME}")
    print(f"Model root: {args.save_root}")
    print(f"Output path: {OUTPUT_PATH}")

    val_df, test_df = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIRS[0])
    val_loader = make_dataloader(val_df, tokenizer)
    test_loader = make_dataloader(test_df, tokenizer)

    print("\nValidation set: learning temperature")
    val_logits_by_model = collect_ensemble_logits(val_loader)
    val_logits = val_logits_by_model.mean(dim=0)
    val_labels = torch.tensor(val_df["risk_severity"].values, dtype=torch.long)

    before_val = compute_metrics(val_logits, val_labels, temperature=1.0)
    temperature = learn_temperature(val_logits, val_labels)
    after_val = compute_metrics(val_logits, val_labels, temperature=temperature)

    print(f"\nLearned temperature: {temperature:.4f}")
    print_metrics("Validation metrics before calibration", before_val)
    print_metrics("Validation metrics after calibration", after_val)

    print("\nTest set: applying learned temperature")
    test_logits_by_model = collect_ensemble_logits(test_loader)
    test_logits = test_logits_by_model.mean(dim=0)
    test_labels = torch.tensor(test_df["risk_severity"].values, dtype=torch.long)

    before_test = compute_metrics(test_logits, test_labels, temperature=1.0)
    after_test = compute_metrics(test_logits, test_labels, temperature=temperature)
    print_metrics("Test metrics before calibration", before_test)
    print_metrics("Test metrics after calibration", after_test)

    save_calibrated_test_csv(test_df, test_logits_by_model, temperature)


if __name__ == "__main__":
    main()
