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
parser.add_argument("--split", choices=["val", "test"], default="test")
parser.add_argument("--split-seed", type=int, default=42)
parser.add_argument("--model_name", type=str, default="mental/mental-roberta-base")
parser.add_argument("--save_root", type=str, default="./saved_models")
parser.add_argument("--batch_size", "--batch-size", type=int, default=32)
parser.add_argument("--output_csv", type=str, default=None)
parser.add_argument("--max-samples", type=int, default=None)
args = parser.parse_args()


MODEL_NAME = args.model_name
MODEL_DIRS = [os.path.join(args.save_root, f"seed_{seed}") for seed in range(1, 6)]
DATA_PATH = "./combined_dataset.csv"
OUTPUT_PATH = args.output_csv or f"./outputs/ensemble_uncertainty_{args.split}.csv"
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
            "text": str(self.texts[idx]),
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
        logits = self.classifier(cls_output)
        return logits


def check_model_dirs(model_dirs):
    for model_dir in model_dirs:
        model_path = os.path.join(model_dir, "best_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")


def load_eval_split():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "risk_severity"])
    df = df[df["text"].str.strip() != ""]

    train_df, temp_df = train_test_split(
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

    eval_df = val_df if args.split == "val" else test_df
    if args.max_samples is not None:
        eval_df = eval_df.head(args.max_samples)

    return eval_df.reset_index(drop=True)


def predict_with_model(model_dir, dataloader):
    model = RobertaRiskClassifier(MODEL_NAME, num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint_path = os.path.join(model_dir, "best_model.pt")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(all_probs)


def entropy(probs):
    safe_probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(safe_probs * np.log(safe_probs), axis=-1)


def main():
    check_model_dirs(MODEL_DIRS)
    os.makedirs("./outputs", exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"Model: {MODEL_NAME}")
    print(f"Model root: {args.save_root}")
    print(f"Evaluation split: {args.split}")
    print(f"Output path: {OUTPUT_PATH}")

    eval_df = load_eval_split()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIRS[0])
    dataset = MentalHealthDataset(
        eval_df["text"].values,
        eval_df["risk_severity"].values,
        tokenizer,
        MAX_LEN,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model_probabilities = []
    for model_dir in MODEL_DIRS:
        print(f"Running inference with {model_dir}")
        model_probabilities.append(predict_with_model(model_dir, dataloader))

    probs_by_model = np.stack(model_probabilities, axis=0)
    mean_probs = probs_by_model.mean(axis=0)
    predicted_labels = mean_probs.argmax(axis=1)
    confidence = mean_probs.max(axis=1)

    epistemic_uncertainty = probs_by_model.var(axis=0).mean(axis=1)
    aleatoric_uncertainty = entropy(probs_by_model).mean(axis=0)

    output_df = pd.DataFrame({
        "text": eval_df["text"].values,
        "true_label": eval_df["risk_severity"].values,
        "predicted_label": predicted_labels,
        "mean_low_prob": mean_probs[:, 0],
        "mean_medium_prob": mean_probs[:, 1],
        "mean_high_prob": mean_probs[:, 2],
        "epistemic_uncertainty": epistemic_uncertainty,
        "aleatoric_uncertainty": aleatoric_uncertainty,
        "confidence": confidence,
    })
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(output_df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
