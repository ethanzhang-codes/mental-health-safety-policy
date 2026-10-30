import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "microsoft/deberta-v3-large"
TRAIN_CSV = "cradle_dialogue/data/cradle_train_targets.csv"
VAL_CSV = "cradle_dialogue/data/cradle_validation_targets.csv"
TEST_CSV = "cradle_dialogue/data/cradle_test_targets.csv"
OUTPUT_DIR = "cradle_dialogue/embeddings"


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CRADLE DeBERTa turn embeddings.")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--train_csv", default=TRAIN_CSV)
    parser.add_argument("--val_csv", default=VAL_CSV)
    parser.add_argument("--test_csv", default=TEST_CSV)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow_overwrite", action="store_true")
    return parser.parse_args()


def prefixed_text(row):
    text = str(row["text"])
    role = str(row["speaker_role"]).lower()
    if role == "user":
        return f"User: {text}"
    if role == "listener":
        return f"Listener: {text}"
    return f"Unknown: {text}"


class TurnEmbeddingDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = [prefixed_text(row) for _, row in self.dataframe.iterrows()]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
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
            "dataset_split": str(row["dataset_split"]),
            "dialogue_id": str(row["dialogue_id"]),
            "turn_id": int(row["turn_id"]),
            "speaker_role": str(row["speaker_role"]),
            "is_user_turn": int(row["is_user_turn"]),
            "target_alert": float(row["target_alert"]),
            "alert_target_mask": float(row["alert_target_mask"]),
            "row_index": int(idx),
        }


def collate_batch(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "dataset_split": [item["dataset_split"] for item in batch],
        "dialogue_id": [item["dialogue_id"] for item in batch],
        "turn_id": torch.tensor([item["turn_id"] for item in batch], dtype=torch.long),
        "speaker_role": [item["speaker_role"] for item in batch],
        "is_user_turn": torch.tensor([item["is_user_turn"] for item in batch], dtype=torch.long),
        "target_alert": torch.tensor([item["target_alert"] for item in batch], dtype=torch.float32),
        "alert_target_mask": torch.tensor([item["alert_target_mask"] for item in batch], dtype=torch.float32),
        "row_index": torch.tensor([item["row_index"] for item in batch], dtype=torch.long),
    }


def encode_split(split_name, csv_path, tokenizer, model, args):
    dataframe = pd.read_csv(csv_path, keep_default_na=False)
    dataset = TurnEmbeddingDataset(dataframe, tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

    embeddings = []
    dataset_split = []
    dialogue_id = []
    turn_id = []
    speaker_role = []
    is_user_turn = []
    target_alert = []
    alert_target_mask = []
    row_index = []

    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().float().cpu()
            if not torch.isfinite(cls_embeddings).all():
                raise FloatingPointError(f"Non-finite embeddings detected for {split_name}.")
            embeddings.append(cls_embeddings)
            dataset_split.extend(batch["dataset_split"])
            dialogue_id.extend(batch["dialogue_id"])
            turn_id.append(batch["turn_id"].clone())
            speaker_role.extend(batch["speaker_role"])
            is_user_turn.append(batch["is_user_turn"].clone())
            target_alert.append(batch["target_alert"].clone())
            alert_target_mask.append(batch["alert_target_mask"].clone())
            row_index.append(batch["row_index"].clone())

    embedding_tensor = torch.cat(embeddings, dim=0)
    turn_id_tensor = torch.cat(turn_id, dim=0)
    is_user_turn_tensor = torch.cat(is_user_turn, dim=0)
    target_alert_tensor = torch.cat(target_alert, dim=0)
    alert_target_mask_tensor = torch.cat(alert_target_mask, dim=0)
    row_index_tensor = torch.cat(row_index, dim=0)

    if embedding_tensor.shape[0] != len(dataframe):
        raise ValueError(f"{split_name}: embedding count {embedding_tensor.shape[0]} != row count {len(dataframe)}")
    if row_index_tensor.tolist() != list(range(len(dataframe))):
        raise ValueError(f"{split_name}: row order misalignment detected.")
    if embedding_tensor.ndim != 2:
        raise ValueError(f"{split_name}: unexpected embedding tensor shape {tuple(embedding_tensor.shape)}")

    payload = {
        "dataset_split": dataset_split,
        "dialogue_id": dialogue_id,
        "turn_id": turn_id_tensor,
        "speaker_role": speaker_role,
        "is_user_turn": is_user_turn_tensor,
        "target_alert": target_alert_tensor,
        "alert_target_mask": alert_target_mask_tensor,
        "row_index": row_index_tensor,
        "embeddings": embedding_tensor,
    }
    duration = time.time() - start_time
    return payload, len(dataframe), duration


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "train": output_dir / "deberta_v3_large_train_turn_embeddings.pt",
        "validation": output_dir / "deberta_v3_large_validation_turn_embeddings.pt",
        "test": output_dir / "deberta_v3_large_test_turn_embeddings.pt",
    }
    metadata_path = output_dir / "deberta_v3_large_embedding_metadata.json"

    if not args.allow_overwrite:
        for path in list(output_paths.values()) + [metadata_path]:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    print(f"Device: {args.device}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)
    model.eval()
    hidden_size = int(model.config.hidden_size)

    source_paths = {
        "train": str(Path(args.train_csv)),
        "validation": str(Path(args.val_csv)),
        "test": str(Path(args.test_csv)),
    }

    split_payloads = {}
    split_counts = {}
    split_durations = {}
    total_start = time.time()
    for split_name, csv_path in [
        ("train", args.train_csv),
        ("validation", args.val_csv),
        ("test", args.test_csv),
    ]:
        payload, row_count, duration = encode_split(split_name, csv_path, tokenizer, model, args)
        if payload["embeddings"].shape[1] != hidden_size:
            raise ValueError(
                f"{split_name}: embedding dim {payload['embeddings'].shape[1]} != hidden_size {hidden_size}"
            )
        torch.save(payload, output_paths[split_name])
        split_payloads[split_name] = payload
        split_counts[split_name] = row_count
        split_durations[split_name] = duration
        print(
            f"{split_name}: rows={row_count}, embeddings={payload['embeddings'].shape[0]}, "
            f"dim={payload['embeddings'].shape[1]}, seconds={duration:.2f}"
        )

    metadata = {
        "model_name": args.model_name,
        "hidden_size": hidden_size,
        "max_length": args.max_length,
        "embedding_type": "deberta_cls_first_token",
        "precision_dtype": "fp32",
        "device": args.device,
        "batch_size": args.batch_size,
        "split_row_counts": split_counts,
        "embeddings_per_split": {
            split_name: int(split_payloads[split_name]["embeddings"].shape[0])
            for split_name in split_payloads
        },
        "embedding_dimension": hidden_size,
        "created_timestamp_epoch": time.time(),
        "created_timestamp_readable": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source_csv_paths": source_paths,
        "split_precompute_seconds": split_durations,
        "total_precompute_seconds": time.time() - total_start,
        "validation_checks": {
            "row_count_matches": True,
            "embedding_dim_matches_hidden_size": True,
            "no_nan_or_inf_embeddings": True,
            "dialogue_id_and_turn_id_preserved": True,
            "row_order_alignment_preserved": True,
            "no_label_or_future_metadata_used_to_create_embeddings": True,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
