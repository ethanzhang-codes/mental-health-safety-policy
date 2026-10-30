import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import os
import argparse

# ── Argument parser (for ensemble seed) ─────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--model_name', type=str, default="mental/mental-roberta-base")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--save_root', type=str, default="./saved_models")
parser.add_argument('--results_root', type=str, default="./results")
parser.add_argument('--tb_log_root', type=str, default="./tensorboard_logs")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME    = args.model_name
MAX_LEN       = 256
BATCH_SIZE    = args.batch_size
EPOCHS        = 5
LEARNING_RATE = 2e-5
SAVE_PATH     = os.path.join(args.save_root, f"seed_{args.seed}")
RESULTS_PATH  = os.path.join(args.results_root, f"seed_{args.seed}.txt")
TB_LOG_DIR    = os.path.join(
    args.tb_log_root,
    MODEL_NAME.replace("/", "__"),
    f"bs_{BATCH_SIZE}",
    f"seed_{args.seed}",
)
DATA_PATH     = "./combined_dataset.csv"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(SAVE_PATH) and os.listdir(SAVE_PATH):
    raise FileExistsError(
        f"Refusing to overwrite existing model directory: {SAVE_PATH}\n"
        "Use a different --seed, move the existing directory, or delete it yourself if you are sure."
    )

if os.path.exists(RESULTS_PATH):
    raise FileExistsError(
        f"Refusing to overwrite existing results file: {RESULTS_PATH}\n"
        "Use a different --seed, move the existing file, or delete it yourself if you are sure."
    )

os.makedirs(args.save_root, exist_ok=True)
os.makedirs(args.results_root, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)

print(f"Seed: {args.seed}")
print(f"Model: {MODEL_NAME}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Save path: {SAVE_PATH}")
print(f"Results path: {RESULTS_PATH}")
print(f"TensorBoard log dir: {TB_LOG_DIR}")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── Dataset ──────────────────────────────────────────────────────────────────
class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Model ────────────────────────────────────────────────────────────────────
class RobertaRiskClassifier(nn.Module):
    def __init__(self, model_name, num_classes=3, dropout=0.3):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden_size     = self.encoder.config.hidden_size

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits     = self.classifier(cls_output)
        return logits

# ── Load and split data ───────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=['text', 'risk_severity'])
df = df[df['text'].str.strip() != '']

train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=args.seed, stratify=df['risk_severity']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df['risk_severity']
)

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Label distribution:\n{df['risk_severity'].value_counts().sort_index()}")

# ── Tokenizer and datasets ────────────────────────────────────────────────────
tokenizer     = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = MentalHealthDataset(train_df['text'].values, train_df['risk_severity'].values, tokenizer, MAX_LEN)
val_dataset   = MentalHealthDataset(val_df['text'].values,   val_df['risk_severity'].values,   tokenizer, MAX_LEN)
test_dataset  = MentalHealthDataset(test_df['text'].values,  test_df['risk_severity'].values,  tokenizer, MAX_LEN)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ── Class weights ─────────────────────────────────────────────────────────────
class_weights = compute_class_weight(
    'balanced',
    classes=np.array([0, 1, 2]),
    y=train_df['risk_severity'].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
print(f"\nClass weights: {class_weights}")

# ── Model, optimizer, scheduler ───────────────────────────────────────────────
model     = RobertaRiskClassifier(MODEL_NAME).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(weight=class_weights)

total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)
writer = SummaryWriter(log_dir=TB_LOG_DIR)
writer.add_text("run/model_name", MODEL_NAME)
writer.add_text("run/save_path", SAVE_PATH)
writer.add_scalar("run/seed", args.seed, 0)
writer.add_scalar("run/batch_size", BATCH_SIZE, 0)
writer.add_scalar("run/learning_rate", LEARNING_RATE, 0)

# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for batch in loader:
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels         = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), f1_score(all_labels, all_preds, average='macro')

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), f1_score(all_labels, all_preds, average='macro'), all_preds, all_labels

# ── Run training ──────────────────────────────────────────────────────────────
best_val_f1 = 0
epoch_metrics = []

for epoch in range(EPOCHS):
    print(f"\n── Epoch {epoch + 1}/{EPOCHS} ──")
    train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler, criterion)
    val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)

    print(f"Train Loss: {train_loss:.4f} | Train Macro-F1: {train_f1:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Macro-F1: {val_f1:.4f}")
    writer.add_scalar("loss/train", train_loss, epoch + 1)
    writer.add_scalar("loss/val", val_loss, epoch + 1)
    writer.add_scalar("macro_f1/train", train_f1, epoch + 1)
    writer.add_scalar("macro_f1/val", val_f1, epoch + 1)
    epoch_metrics.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_f1": val_f1,
    })

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(model.state_dict(), f"{SAVE_PATH}/best_model.pt")
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"  ✓ Best model saved (val F1: {val_f1:.4f})")

# ── Final test evaluation ──────────────────────────────────────────────────────
print("\n── Test Set Evaluation ──")
model.load_state_dict(torch.load(f"{SAVE_PATH}/best_model.pt"))
_, test_f1, test_preds, test_labels = evaluate(model, test_loader, criterion)

test_report = classification_report(
    test_labels, test_preds,
    target_names=['Low Risk', 'Medium Risk', 'High Risk']
)
print(test_report)
print(f"Final Test Macro-F1: {test_f1:.4f}")
report_dict = classification_report(
    test_labels,
    test_preds,
    target_names=['Low Risk', 'Medium Risk', 'High Risk'],
    output_dict=True,
)
writer.add_scalar("macro_f1/test", test_f1, EPOCHS + 1)
writer.add_scalar("accuracy/test", report_dict["accuracy"], EPOCHS + 1)
writer.add_scalar("class_f1/low_risk", report_dict["Low Risk"]["f1-score"], EPOCHS + 1)
writer.add_scalar("class_f1/medium_risk", report_dict["Medium Risk"]["f1-score"], EPOCHS + 1)
writer.add_scalar("class_f1/high_risk", report_dict["High Risk"]["f1-score"], EPOCHS + 1)
writer.flush()
writer.close()

with open(RESULTS_PATH, "w") as results_file:
    results_file.write(f"Seed: {args.seed}\n")
    results_file.write(f"Model: {MODEL_NAME}\n")
    results_file.write(f"Batch size: {BATCH_SIZE}\n")
    results_file.write(f"Learning rate: {LEARNING_RATE}\n")
    results_file.write(f"Epochs: {EPOCHS}\n")
    results_file.write(f"Saved model directory: {SAVE_PATH}\n")
    results_file.write(f"Device: {DEVICE}\n")
    results_file.write(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\n")
    results_file.write("\nDataset split:\n")
    results_file.write(f"Train: {len(train_df)}\n")
    results_file.write(f"Val: {len(val_df)}\n")
    results_file.write(f"Test: {len(test_df)}\n")
    results_file.write("\nEpoch metrics:\n")
    for metrics in epoch_metrics:
        results_file.write(
            f"Epoch {metrics['epoch']}: "
            f"Train Loss={metrics['train_loss']:.4f}, "
            f"Train Macro-F1={metrics['train_f1']:.4f}, "
            f"Val Loss={metrics['val_loss']:.4f}, "
            f"Val Macro-F1={metrics['val_f1']:.4f}\n"
        )
    results_file.write(f"\nBest Val Macro-F1: {best_val_f1:.4f}\n")
    results_file.write("\nTest classification report:\n")
    results_file.write(test_report)
    results_file.write(f"\nFinal Test Macro-F1: {test_f1:.4f}\n")

print(f"Results saved to {RESULTS_PATH}")
