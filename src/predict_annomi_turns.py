import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="./data/annomi_turns.csv")
parser.add_argument("--output", default="./outputs/annomi_turn_predictions.csv")
parser.add_argument("--model_name", type=str, default="mental/mental-roberta-base")
parser.add_argument("--save_root", type=str, default="./saved_models")
parser.add_argument("--batch_size", "--batch-size", type=int, default=32)
parser.add_argument("--max-samples", type=int, default=None)
args = parser.parse_args()


MODEL_NAME = args.model_name
MODEL_DIRS = [os.path.join(args.save_root, f"seed_{seed}") for seed in range(1, 6)]
REQUIRED_COLUMNS = ["conversation_id", "turn_id", "speaker_role", "text"]
MAX_LEN = 256
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalTurnDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
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


def check_inputs():
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Missing input file: {args.input}")

    for model_dir in MODEL_DIRS:
        model_path = os.path.join(model_dir, "best_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model checkpoint: {model_path}")


def load_turns():
    turns_df = pd.read_csv(args.input)
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in turns_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Input file is missing required columns: {missing_columns}\n"
            f"Required columns are: {REQUIRED_COLUMNS}"
        )

    turns_df = turns_df.sort_values(["conversation_id", "turn_id"]).reset_index(drop=True)
    if args.max_samples is not None:
        turns_df = turns_df.head(args.max_samples).copy()

    return turns_df


def predict_with_model(model_dir, dataloader):
    model = RobertaRiskClassifier(MODEL_NAME, num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint_path = os.path.join(model_dir, "best_model.pt")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)

    print(f"{model_dir} missing_keys: {missing_keys}")
    print(f"{model_dir} unexpected_keys: {unexpected_keys}")

    classifier_key_problems = [
        key for key in missing_keys + unexpected_keys
        if key.startswith("classifier.")
    ]
    if classifier_key_problems:
        raise RuntimeError(
            "The saved checkpoint does not match the classifier/risk prediction head. "
            f"Problem keys: {classifier_key_problems}"
        )

    if missing_keys or unexpected_keys:
        print(
            "Only non-classifier keys were missing or unexpected. "
            "The classifier/risk prediction head loaded safely."
        )
    else:
        print("All checkpoint keys loaded exactly.")

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
    check_inputs()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("This script does inference only. It does not train or update any model.")
    print(f"Input turns: {args.input}")
    print(f"Output predictions: {args.output}")
    print(f"Model: {MODEL_NAME}")
    print(f"Model root: {args.save_root}")
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

    turns_df = load_turns()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIRS[0])
    dataset = TemporalTurnDataset(turns_df["text"].values, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model_probabilities = []
    for model_dir in MODEL_DIRS:
        print(f"Running inference with {model_dir}")
        model_probabilities.append(predict_with_model(model_dir, dataloader))

    probs_by_model = np.stack(model_probabilities, axis=0)
    mean_probs = probs_by_model.mean(axis=0)

    turns_df["mean_low_prob"] = mean_probs[:, 0]
    turns_df["mean_medium_prob"] = mean_probs[:, 1]
    turns_df["mean_high_prob"] = mean_probs[:, 2]
    turns_df["predicted_label"] = mean_probs.argmax(axis=1)
    turns_df["confidence"] = mean_probs.max(axis=1)
    turns_df["epistemic_uncertainty"] = probs_by_model.var(axis=0).mean(axis=1)
    turns_df["aleatoric_uncertainty"] = entropy(probs_by_model).mean(axis=0)

    output_columns = [
        "conversation_id",
        "turn_id",
        "speaker_role",
        "text",
        "mean_low_prob",
        "mean_medium_prob",
        "mean_high_prob",
        "predicted_label",
        "confidence",
        "epistemic_uncertainty",
        "aleatoric_uncertainty",
    ]
    turns_df[output_columns].to_csv(args.output, index=False)

    print(f"Saved {len(turns_df)} turn predictions to {args.output}")


if __name__ == "__main__":
    main()
