from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = PROJECT_DIR / "module1_encoder_benchmark"
MODEL_ROOT = BENCHMARK_ROOT / "models"
RESULTS_ROOT = BENCHMARK_ROOT / "results"
PREDICTIONS_ROOT = BENCHMARK_ROOT / "predictions"
SPLIT_ROOT = BENCHMARK_ROOT / "splits"
TENSORBOARD_ROOT = BENCHMARK_ROOT / "tensorboard_logs"

DATA_PATH = PROJECT_DIR / "combined_dataset.csv"

LABEL_NAMES = ["Low Risk", "Medium Risk", "High Risk"]
LABEL_TO_ID = {
    "low": 0,
    "medium": 1,
    "high": 2,
}
NUM_CLASSES = 3

DEFAULT_SPLIT_SEED = 1
DEFAULT_MAX_LEN = 256
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_DROPOUT = 0.3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_FRACTION = 0.10
DEFAULT_NUM_WORKERS = 2

ENCODER_CONFIGS = {
    "mental_roberta_base": {
        "model_name": "mental/mental-roberta-base",
        "screening_batch_sizes": [32],
    },
    "mental_bert_large": {
        "model_name": "AIMH/mental-bert-large-cased",
        "screening_batch_sizes": [8],
    },
    "mental_roberta_large": {
        "model_name": "AIMH/mental-roberta-large",
        "screening_batch_sizes": [32, 8, 4],
    },
    "deberta_v3_large": {
        "model_name": "microsoft/deberta-v3-large",
        "screening_batch_sizes": [32, 8, 4],
    },
    "modernbert_large": {
        "model_name": "answerdotai/ModernBERT-large",
        "screening_batch_sizes": [32, 8, 4],
    },
}


def model_save_dir(encoder_short, batch_size, seed):
    return MODEL_ROOT / encoder_short / f"bs{batch_size}" / f"seed_{seed}"


def result_path(encoder_short, batch_size, seed):
    return RESULTS_ROOT / encoder_short / f"bs{batch_size}" / f"seed_{seed}.txt"


def prediction_dir(encoder_short, batch_size, seed):
    return PREDICTIONS_ROOT / encoder_short / f"bs{batch_size}" / f"seed_{seed}"


def tensorboard_log_dir(encoder_short, batch_size, seed):
    return TENSORBOARD_ROOT / encoder_short / f"bs{batch_size}" / f"seed_{seed}"
