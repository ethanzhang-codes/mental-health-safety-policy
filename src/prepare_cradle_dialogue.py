"""
Prepare CRADLE-Dialogue turn-level files for future modeling.

Scientific notes:
- CRADLE-Dialogue train and validation are synthetic/structured conversational data.
- CRADLE-Dialogue test is the clinician-annotated benchmark.
- The released Hugging Face table currently contains turn_id, text, and labels; dialogue_id is
  inferred here from turn_id == 0 resets.
- The dataset does not directly contain Low/Medium/High severity labels.
- Alert is an early type-agnostic crisis signal.
- Confirm identifies explicit crisis categories.
- Category targets are multi-label.
- Progression-stage and landmark columns are ground-truth evaluation metadata. They must not be
  used as future-information model inputs during causal turn-by-turn prediction.

This script does not train a model.
"""

from collections import Counter
from pathlib import Path
import re

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "SungJoo/Cradle-Dialogue"
DATA_DIR = Path("cradle_dialogue/data")
RESULTS_DIR = Path("cradle_dialogue/results")
SPLITS = ["train", "validation", "test"]
EXPECTED_DIALOGUES = {"train": 3058, "validation": 420, "test": 600}
EXPECTED_ROWS = {"train": 48557, "validation": 6679, "test": 8975}
CATEGORIES = ["CA", "DV", "RA", "SHA", "SH", "SI_active", "SI_passive"]
LABEL_VOCAB = {
    "alert_ongoing",
    "alert_past",
    "confirm_CA_ongoing",
    "confirm_CA_past",
    "confirm_DV_ongoing",
    "confirm_DV_past",
    "confirm_RA_ongoing",
    "confirm_RA_past",
    "confirm_SHA_ongoing",
    "confirm_SHA_past",
    "confirm_SH_ongoing",
    "confirm_SH_past",
    "confirm_SI_active_ongoing",
    "confirm_SI_active_past",
    "confirm_SI_passive_ongoing",
    "confirm_SI_passive_past",
}


def split_labels(value):
    value = "" if value is None else str(value)
    value = value.strip()
    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def infer_dialogue_ids(df, split):
    ids = []
    current = 0
    for i, turn_id in enumerate(df["turn_id"].astype(int).tolist()):
        if turn_id == 0:
            current += 1
        elif current == 0:
            raise ValueError(f"{split}: first row did not start with turn_id 0")
        ids.append(f"{split}_dialogue_{current:06d}")
    out = df.copy()
    out.insert(0, "dataset_split", split)
    out.insert(1, "dialogue_id", ids)
    out.insert(2, "dialogue_id_source", "inferred_from_turn_id_reset")
    return out


def parse_speaker(raw_text):
    text = str(raw_text)
    if text.startswith("User:"):
        return "user", text[len("User:") :].lstrip()
    if text.startswith("Listener:"):
        return "listener", text[len("Listener:") :].lstrip()
    return "unknown", text


def label_event_types(labels):
    has_alert = any(label.startswith("alert_") for label in labels)
    has_confirm = any(label.startswith("confirm_") for label in labels)
    parts = []
    if has_alert:
        parts.append("alert")
    if has_confirm:
        parts.append("confirm")
    return has_alert, has_confirm, ";".join(parts) if parts else "none"


def label_temporal_states(labels):
    states = []
    if any(label.endswith("_ongoing") for label in labels):
        states.append("ongoing")
    if any(label.endswith("_past") for label in labels):
        states.append("past")
    return "ongoing" in states, "past" in states, ";".join(states) if states else "none"


def category_from_confirm(label):
    if not label.startswith("confirm_"):
        return None
    for state in ("_ongoing", "_past"):
        if label.endswith(state):
            category = label[len("confirm_") : -len(state)]
            return category if category in CATEGORIES else None
    return None


def add_label_columns(df):
    parsed = df["labels"].map(split_labels)
    df["labels_raw"] = df["labels"].astype(str)
    df["label_count"] = parsed.map(len).astype(int)
    df["has_any_crisis_label"] = (df["label_count"] > 0).astype(int)

    event_values = parsed.map(label_event_types)
    df["has_alert"] = event_values.map(lambda x: int(x[0]))
    df["has_confirm"] = event_values.map(lambda x: int(x[1]))
    df["event_types"] = event_values.map(lambda x: x[2])

    for category in CATEGORIES:
        df[f"category_{category}"] = parsed.map(
            lambda labels, c=category: int(any(category_from_confirm(label) == c for label in labels))
        )
    category_cols = [f"category_{category}" for category in CATEGORIES]
    df["category_count"] = df[category_cols].sum(axis=1).astype(int)
    df["has_confirmed_category"] = (df["category_count"] > 0).astype(int)

    temporal_values = parsed.map(label_temporal_states)
    df["has_ongoing_label"] = temporal_values.map(lambda x: int(x[0]))
    df["has_past_label"] = temporal_values.map(lambda x: int(x[1]))
    df["temporal_states"] = temporal_values.map(lambda x: x[2])
    return df


def first_or_na(values):
    if len(values) == 0:
        return pd.NA
    return int(min(values))


def add_landmarks(df):
    landmark_rows = []
    for dialogue_id, group in df.groupby("dialogue_id", sort=False):
        alert_turns = group.loc[group["has_alert"] == 1, "turn_id"].astype(int).tolist()
        confirm_turns = group.loc[group["has_confirm"] == 1, "turn_id"].astype(int).tolist()
        labeled_turns = group.loc[group["has_any_crisis_label"] == 1, "turn_id"].astype(int).tolist()
        row = {
            "dialogue_id": dialogue_id,
            "first_alert_turn_id": first_or_na(alert_turns),
            "first_confirm_turn_id": first_or_na(confirm_turns),
            "first_crisis_labeled_turn_id": first_or_na(labeled_turns),
            "last_crisis_labeled_turn_id": int(max(labeled_turns)) if labeled_turns else pd.NA,
        }
        for category in CATEGORIES:
            turns = group.loc[group[f"category_{category}"] == 1, "turn_id"].astype(int).tolist()
            row[f"first_confirm_{category}_turn_id"] = first_or_na(turns)
        landmark_rows.append(row)

    landmarks = pd.DataFrame(landmark_rows)
    out = df.merge(landmarks, on="dialogue_id", how="left")
    landmark_cols = [c for c in landmarks.columns if c != "dialogue_id"]
    for col in landmark_cols:
        out[col] = out[col].astype("Int64")
    return out


def nullable_diff(a, b):
    if pd.isna(a) or pd.isna(b):
        return pd.NA
    return int(a) - int(b)


def progression_stage(row):
    turn_id = int(row["turn_id"])
    alert = row["first_alert_turn_id"]
    confirm = row["first_confirm_turn_id"]
    has_alert = not pd.isna(alert)
    has_confirm = not pd.isna(confirm)
    if has_alert and has_confirm:
        alert = int(alert)
        confirm = int(confirm)
        if turn_id < alert:
            return "pre_alert"
        if turn_id == alert:
            return "alert"
        if alert < turn_id < confirm:
            return "between_alert_confirm"
        if turn_id == confirm:
            return "confirm"
        if turn_id > confirm:
            return "post_confirm"
    if has_alert and not has_confirm:
        return "alert_only" if turn_id >= int(alert) else "unlabeled_no_progression"
    if has_confirm and not has_alert:
        return "confirm_only" if turn_id >= int(confirm) else "unlabeled_no_progression"
    return "unlabeled_no_progression"


def add_temporal_targets(df):
    df["turns_until_first_alert"] = df.apply(
        lambda r: nullable_diff(r["first_alert_turn_id"], r["turn_id"]), axis=1
    ).astype("Int64")
    df["turns_until_first_confirm"] = df.apply(
        lambda r: nullable_diff(r["first_confirm_turn_id"], r["turn_id"]), axis=1
    ).astype("Int64")
    df["turns_since_first_alert"] = df.apply(
        lambda r: pd.NA if pd.isna(r["first_alert_turn_id"]) or int(r["turn_id"]) < int(r["first_alert_turn_id"]) else int(r["turn_id"]) - int(r["first_alert_turn_id"]),
        axis=1,
    ).astype("Int64")
    df["turns_since_first_confirm"] = df.apply(
        lambda r: pd.NA if pd.isna(r["first_confirm_turn_id"]) or int(r["turn_id"]) < int(r["first_confirm_turn_id"]) else int(r["turn_id"]) - int(r["first_confirm_turn_id"]),
        axis=1,
    ).astype("Int64")
    df["is_pre_alert"] = df.apply(lambda r: int(not pd.isna(r["first_alert_turn_id"]) and int(r["turn_id"]) < int(r["first_alert_turn_id"])), axis=1)
    df["is_alert_turn"] = df.apply(lambda r: int(not pd.isna(r["first_alert_turn_id"]) and int(r["turn_id"]) == int(r["first_alert_turn_id"])), axis=1)
    df["is_between_alert_and_confirm"] = df.apply(
        lambda r: int(not pd.isna(r["first_alert_turn_id"]) and not pd.isna(r["first_confirm_turn_id"]) and int(r["first_alert_turn_id"]) < int(r["turn_id"]) < int(r["first_confirm_turn_id"])),
        axis=1,
    )
    df["is_confirm_turn"] = df.apply(lambda r: int(not pd.isna(r["first_confirm_turn_id"]) and int(r["turn_id"]) == int(r["first_confirm_turn_id"])), axis=1)
    df["is_post_confirm"] = df.apply(lambda r: int(not pd.isna(r["first_confirm_turn_id"]) and int(r["turn_id"]) > int(r["first_confirm_turn_id"])), axis=1)
    df["progression_stage"] = df.apply(progression_stage, axis=1)
    return df


def ordered_columns():
    return [
        "dataset_split", "dialogue_id", "dialogue_id_source", "turn_id", "speaker_role",
        "is_user_turn", "is_listener_turn", "raw_text", "text", "labels_raw", "label_count",
        "has_any_crisis_label", "has_alert", "has_confirm", "event_types",
        *[f"category_{category}" for category in CATEGORIES],
        "category_count", "has_confirmed_category", "has_ongoing_label", "has_past_label",
        "temporal_states", "first_alert_turn_id", "first_confirm_turn_id",
        "first_crisis_labeled_turn_id", "last_crisis_labeled_turn_id",
        *[f"first_confirm_{category}_turn_id" for category in CATEGORIES],
        "turns_until_first_alert", "turns_until_first_confirm", "turns_since_first_alert",
        "turns_since_first_confirm", "is_pre_alert", "is_alert_turn",
        "is_between_alert_and_confirm", "is_confirm_turn", "is_post_confirm", "progression_stage",
    ]


def process_split(raw_df, split):
    df = infer_dialogue_ids(raw_df, split)
    df["raw_text"] = df["text"].astype(str)
    speaker = df["raw_text"].map(parse_speaker)
    df["speaker_role"] = speaker.map(lambda x: x[0])
    df["text"] = speaker.map(lambda x: x[1])
    df["is_user_turn"] = (df["speaker_role"] == "user").astype(int)
    df["is_listener_turn"] = (df["speaker_role"] == "listener").astype(int)
    df = add_label_columns(df)
    df = add_landmarks(df)
    df = add_temporal_targets(df)
    return df[ordered_columns()]


def validate(df, split):
    errors = []
    if len(df) != EXPECTED_ROWS[split]:
        errors.append(f"row count {len(df)} != expected {EXPECTED_ROWS[split]}")
    if df["dialogue_id"].nunique() != EXPECTED_DIALOGUES[split]:
        errors.append(f"dialogue count {df['dialogue_id'].nunique()} != expected {EXPECTED_DIALOGUES[split]}")
    if df["raw_text"].isna().any():
        errors.append("missing raw_text found")
    if df.duplicated(["dialogue_id", "turn_id"]).any():
        errors.append("duplicate dialogue_id + turn_id found")
    labels = {label for labels in df["labels_raw"] for label in split_labels(labels)}
    unknown = labels - LABEL_VOCAB
    if unknown:
        errors.append(f"unknown labels found: {sorted(unknown)}")
    category_cols = [f"category_{category}" for category in CATEGORIES]
    for col in category_cols:
        if not df[col].isin([0, 1]).all():
            errors.append(f"{col} is not binary")
    if not (df["category_count"] == df[category_cols].sum(axis=1)).all():
        errors.append("category_count mismatch")
    if not (df["has_confirmed_category"] == (df["category_count"] > 0).astype(int)).all():
        errors.append("has_confirmed_category mismatch")
    if not (df["has_alert"] == df["labels_raw"].map(lambda x: int(any(l.startswith("alert_") for l in split_labels(x))))).all():
        errors.append("has_alert mismatch")
    if not (df["has_confirm"] == df["labels_raw"].map(lambda x: int(any(l.startswith("confirm_") for l in split_labels(x))))).all():
        errors.append("has_confirm mismatch")
    for dialogue_id, group in df.groupby("dialogue_id"):
        turns = group["turn_id"].astype(int).tolist()
        if turns[0] != 0 or turns != list(range(len(turns))):
            errors.append(f"bad turn sequence in {dialogue_id}")
            break
        for category in CATEGORIES:
            first_col = f"first_confirm_{category}_turn_id"
            leaked_rows = group[(group["has_confirm"] == 0) & (group[f"category_{category}"] == 1)]
            if not leaked_rows.empty:
                errors.append(f"confirm category present without exact confirm label in {dialogue_id}")
                break
    return errors


def audit_lines(processed, validation_errors):
    lines = [
        "CRADLE-Dialogue Preprocessing Audit",
        "===================================",
        "",
        "No model was trained.",
        "Train/validation are synthetic/structured conversational data; test is the clinician-annotated benchmark.",
        "Dialogue IDs are inferred from turn_id == 0 resets because dialogue_id is absent from the loaded release.",
        "",
    ]
    for split, df in processed.items():
        category_cols = [f"category_{category}" for category in CATEGORIES]
        grouped = df.groupby("dialogue_id")
        alert_dialogues = set(df.loc[df["has_alert"] == 1, "dialogue_id"])
        confirm_dialogues = set(df.loc[df["has_confirm"] == 1, "dialogue_id"])
        lines.extend([
            f"Split: {split}",
            "-" * (7 + len(split)),
            f"turns: {len(df)}",
            f"dialogues: {df['dialogue_id'].nunique()}",
            f"user turns: {int(df['is_user_turn'].sum())}",
            f"listener turns: {int(df['is_listener_turn'].sum())}",
            f"unknown speaker turns: {int((df['speaker_role'] == 'unknown').sum())}",
            f"labeled turns: {int(df['has_any_crisis_label'].sum())}",
            f"Alert turns: {int(df['has_alert'].sum())}",
            f"Confirm turns: {int(df['has_confirm'].sum())}",
            f"confirmed category distribution: {df[category_cols].sum().to_dict()}",
            f"multi-category turn count: {int((df['category_count'] > 1).sum())}",
            f"ongoing label count: {int(df['has_ongoing_label'].sum())}",
            f"past label count: {int(df['has_past_label'].sum())}",
            f"progression-stage distribution: {dict(Counter(df['progression_stage']))}",
            f"Alert-only dialogue count: {len(alert_dialogues - confirm_dialogues)}",
            f"Confirm-only dialogue count: {len(confirm_dialogues - alert_dialogues)}",
            f"Alert-to-Confirm dialogue count: {len(alert_dialogues & confirm_dialogues)}",
            f"crisis labels on User turns: {int(df.loc[df['is_user_turn'] == 1, 'has_any_crisis_label'].sum())}",
            f"crisis labels on Listener turns: {int(df.loc[df['is_listener_turn'] == 1, 'has_any_crisis_label'].sum())}",
            f"Alert labels by speaker: {df[df['has_alert'] == 1].groupby('speaker_role').size().to_dict()}",
            f"Confirm labels by speaker: {df[df['has_confirm'] == 1].groupby('speaker_role').size().to_dict()}",
            f"validation errors: {validation_errors[split] if validation_errors[split] else 'none'}",
            "",
        ])
    if any(validation_errors.values()):
        lines.append("Serious validation inconsistencies were found. Review before modeling.")
    else:
        lines.append("All validation checks passed.")
    return "\n".join(lines) + "\n"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME)
    processed = {}
    validation_errors = {}
    for split in SPLITS:
        df = dataset[split].to_pandas()
        processed[split] = process_split(df, split)
        validation_errors[split] = validate(processed[split], split)
        if validation_errors[split]:
            raise ValueError(f"{split} validation failed: {validation_errors[split]}")
        processed[split].to_csv(DATA_DIR / f"cradle_{split}_turns.csv", index=False)
    report = audit_lines(processed, validation_errors)
    (RESULTS_DIR / "cradle_preprocessing_audit.txt").write_text(report)
    print(report)
    print("Saved:")
    for split in SPLITS:
        print(f"- {DATA_DIR / f'cradle_{split}_turns.csv'}")
    print(f"- {RESULTS_DIR / 'cradle_preprocessing_audit.txt'}")


if __name__ == "__main__":
    main()
