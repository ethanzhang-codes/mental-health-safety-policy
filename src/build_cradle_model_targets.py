"""
Build causal prediction targets for CRADLE-Dialogue.

This script does not train a model.

Design decisions:
- CRADLE is not used to create Low/Medium/High severity labels.
- Severity remains separate in the existing Module 1A severity benchmark.
- Alert, Confirm, category, temporal-state, and forecasting columns are target/evaluation
  metadata, not model inputs.
- Forecasting targets may use future labels to define the supervised outcome, but the future
  labels must never be included in model inputs.
"""

from collections import Counter
from pathlib import Path

import pandas as pd


DATA_DIR = Path("cradle_dialogue/data")
RESULTS_DIR = Path("cradle_dialogue/results")
SPLITS = ["train", "validation", "test"]
CATEGORIES = ["CA", "DV", "RA", "SHA", "SH", "SI_active", "SI_passive"]
FORECAST_HORIZONS = [1, 2, 3, 5]

SAFE_TURN_INPUT_COLUMNS = [
    "dialogue_id",  # grouping only; do not feed as semantic text feature
    "turn_id",
    "speaker_role",
    "is_user_turn",
    "is_listener_turn",
    "text",
]

SAFE_HISTORY_DERIVED_COLUMNS = [
    # Placeholder for future causal features computed only from turns <= current turn.
    # None are created in this target-building step.
]

EVALUATION_ONLY_FUTURE_COLUMNS = [
    "labels_raw",
    "label_count",
    "has_any_crisis_label",
    "has_alert",
    "has_confirm",
    "event_types",
    "category_count",
    "has_confirmed_category",
    "has_ongoing_label",
    "has_past_label",
    "temporal_states",
    "first_alert_turn_id",
    "first_confirm_turn_id",
    "first_crisis_labeled_turn_id",
    "last_crisis_labeled_turn_id",
    *[f"first_confirm_{category}_turn_id" for category in CATEGORIES],
    "turns_until_first_alert",
    "turns_until_first_confirm",
    "turns_since_first_alert",
    "turns_since_first_confirm",
    "is_pre_alert",
    "is_alert_turn",
    "is_between_alert_and_confirm",
    "is_confirm_turn",
    "is_post_confirm",
    "progression_stage",
    "target_alert",
    "target_confirm",
    *[f"target_category_{category}" for category in CATEGORIES],
    "target_ongoing",
    "target_past",
    "user_turn_index",
    "user_turns_until_first_confirm",
    *[f"confirm_within_{k}_user_turn{'s' if k > 1 else ''}" for k in FORECAST_HORIZONS],
]


def split_labels(value):
    if pd.isna(value):
        return []
    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def exact_has_alert(value):
    return int(any(label.startswith("alert_") for label in split_labels(value)))


def exact_has_confirm(value):
    return int(any(label.startswith("confirm_") for label in split_labels(value)))


def exact_has_category(value, category):
    labels = split_labels(value)
    return int(
        any(
            label == f"confirm_{category}_ongoing" or label == f"confirm_{category}_past"
            for label in labels
        )
    )


def exact_has_ongoing(value):
    return int(any(label.endswith("_ongoing") for label in split_labels(value)))


def exact_has_past(value):
    return int(any(label.endswith("_past") for label in split_labels(value)))


def add_current_turn_targets(df):
    df = df.copy()
    df["target_alert"] = df["labels_raw"].map(exact_has_alert).astype(int)
    df["alert_target_mask"] = (df["speaker_role"] == "user").astype(int)
    df["target_confirm"] = df["labels_raw"].map(exact_has_confirm).astype(int)
    df["confirm_target_mask"] = (df["speaker_role"] == "user").astype(int)

    for category in CATEGORIES:
        df[f"target_category_{category}"] = df["labels_raw"].map(
            lambda value, c=category: exact_has_category(value, c)
        ).astype(int)
    df["category_target_mask"] = (df["speaker_role"] == "user").astype(int)

    df["target_ongoing"] = df["labels_raw"].map(exact_has_ongoing).astype(int)
    df["target_past"] = df["labels_raw"].map(exact_has_past).astype(int)
    df["temporal_state_target_mask"] = (
        (df["target_ongoing"] == 1) | (df["target_past"] == 1)
    ).astype(int)
    return df


def add_user_turn_indexes_and_forecasts(df):
    df = df.copy()
    df["user_turn_index"] = pd.NA
    df["user_turns_until_first_confirm"] = pd.NA
    for k in FORECAST_HORIZONS:
        suffix = "user_turn" if k == 1 else "user_turns"
        df[f"confirm_within_{k}_{suffix}"] = pd.NA
    df["forecast_target_mask"] = (df["speaker_role"] == "user").astype(int)

    pieces = []
    for _, group in df.groupby("dialogue_id", sort=False):
        group = group.copy()
        user_positions = group.index[group["speaker_role"] == "user"].tolist()
        index_by_position = {idx: n for n, idx in enumerate(user_positions)}
        for idx, n in index_by_position.items():
            group.loc[idx, "user_turn_index"] = n

        user_confirm_positions = [
            idx for idx in user_positions if int(group.loc[idx, "target_confirm"]) == 1
        ]
        first_user_confirm_index = (
            index_by_position[user_confirm_positions[0]] if user_confirm_positions else None
        )
        if first_user_confirm_index is not None:
            for idx, n in index_by_position.items():
                group.loc[idx, "user_turns_until_first_confirm"] = first_user_confirm_index - n

        for idx, n in index_by_position.items():
            for k in FORECAST_HORIZONS:
                suffix = "user_turn" if k == 1 else "user_turns"
                future_positions = [
                    pos
                    for pos in user_positions
                    if n < index_by_position[pos] <= n + k
                ]
                target = int(any(int(group.loc[pos, "target_confirm"]) == 1 for pos in future_positions))
                group.loc[idx, f"confirm_within_{k}_{suffix}"] = target
        pieces.append(group)

    out = pd.concat(pieces).sort_index()
    nullable_cols = ["user_turn_index", "user_turns_until_first_confirm"] + [
        f"confirm_within_{k}_{'user_turn' if k == 1 else 'user_turns'}"
        for k in FORECAST_HORIZONS
    ]
    for col in nullable_cols:
        out[col] = out[col].astype("Int64")
    return out


def build_targets(df):
    df = add_current_turn_targets(df)
    df = add_user_turn_indexes_and_forecasts(df)
    return df


def validate_targets(df, split):
    errors = []
    if not (df["target_alert"] == df["labels_raw"].map(exact_has_alert)).all():
        errors.append("target_alert mismatch")
    if not (df["target_confirm"] == df["labels_raw"].map(exact_has_confirm)).all():
        errors.append("target_confirm mismatch")
    for category in CATEGORIES:
        target_col = f"target_category_{category}"
        if not (df[target_col] == df["labels_raw"].map(lambda v, c=category: exact_has_category(v, c))).all():
            errors.append(f"{target_col} mismatch")
    target_category_cols = [f"target_category_{category}" for category in CATEGORIES]
    if not (df[target_category_cols].sum(axis=1) == df["category_count"]).all():
        errors.append("target category sum does not match category_count")
    if not (df["target_ongoing"] == df["labels_raw"].map(exact_has_ongoing)).all():
        errors.append("target_ongoing mismatch")
    if not (df["target_past"] == df["labels_raw"].map(exact_has_past)).all():
        errors.append("target_past mismatch")
    if set(EVALUATION_ONLY_FUTURE_COLUMNS) & set(SAFE_TURN_INPUT_COLUMNS):
        errors.append("future/evaluation columns included in safe turn inputs")
    for _, group in df.groupby("dialogue_id", sort=False):
        user = group[group["speaker_role"] == "user"]
        expected_indexes = list(range(len(user)))
        actual = user["user_turn_index"].dropna().astype(int).tolist()
        if actual != expected_indexes:
            errors.append(f"user_turn_index mismatch in {group['dialogue_id'].iloc[0]}")
            break
        for k in FORECAST_HORIZONS:
            suffix = "user_turn" if k == 1 else "user_turns"
            col = f"confirm_within_{k}_{suffix}"
            for _, row in user.iterrows():
                n = int(row["user_turn_index"])
                future = user[(user["user_turn_index"] > n) & (user["user_turn_index"] <= n + k)]
                expected = int((future["target_confirm"] == 1).any())
                if int(row[col]) != expected:
                    errors.append(f"{col} crosses boundary or mismatch in {row['dialogue_id']}")
                    break
            if errors:
                break
        if errors:
            break
    return errors


def target_distribution(df, target, mask=None):
    sub = df if mask is None else df[df[mask] == 1]
    positive = int(sub[target].fillna(0).astype(int).sum())
    total = len(sub)
    negative = total - positive
    prevalence = positive / total if total else 0.0
    ratio = positive / negative if negative else None
    return {
        "positive": positive,
        "negative": negative,
        "prevalence": prevalence,
        "positive_to_negative_ratio": ratio,
    }


def format_dist(name, dist):
    ratio = "undefined" if dist["positive_to_negative_ratio"] is None else f"{dist['positive_to_negative_ratio']:.6f}"
    return (
        f"- {name}: positive={dist['positive']}, negative={dist['negative']}, "
        f"prevalence={dist['prevalence']:.6f}, pos:neg={ratio}"
    )


def audit_report(processed, validation_errors):
    lines = [
        "CRADLE-Dialogue Model Target Audit",
        "==================================",
        "",
        "No model was trained.",
        "",
        "Target definitions:",
        "- target_alert: 1 only when the current turn contains an exact alert_* label.",
        "- target_confirm: 1 only when the current turn contains any exact confirm_* label.",
        "- target_category_*: seven independent multi-label category targets from exact confirm labels.",
        "- target_ongoing / target_past: binary temporal-state targets from exact labels ending in _ongoing or _past.",
        "- confirm_within_K_user_turns: for User turns only, whether a future User turn in the same dialogue confirms crisis within K subsequent User turns.",
        "",
        "Target masks:",
        "- alert_target_mask, confirm_target_mask, category_target_mask: 1 for User turns, 0 for Listener/unknown turns.",
        "- temporal_state_target_mask: 1 only for turns with an exact temporal-state crisis label.",
        "- forecast_target_mask: 1 for User turns only.",
        "",
        "Forbidden / evaluation-only future columns:",
        *[f"- {col}" for col in EVALUATION_ONLY_FUTURE_COLUMNS],
        "",
        "Safe turn input columns:",
        *[f"- {col}" for col in SAFE_TURN_INPUT_COLUMNS],
        "",
        "Safe history-derived causal columns:",
        "- none created yet; future features must use only turns <= current turn.",
        "",
    ]

    for split, df in processed.items():
        lines.extend([f"Split: {split}", "-" * (7 + len(split))])
        primary_targets = [
            ("target_alert", "alert_target_mask"),
            ("target_confirm", "confirm_target_mask"),
            *[(f"target_category_{category}", "category_target_mask") for category in CATEGORIES],
            ("target_ongoing", "temporal_state_target_mask"),
            ("target_past", "temporal_state_target_mask"),
        ]
        for target, mask in primary_targets:
            lines.append(format_dist(target, target_distribution(df, target, mask)))
        for k in FORECAST_HORIZONS:
            suffix = "user_turn" if k == 1 else "user_turns"
            target = f"confirm_within_{k}_{suffix}"
            lines.append(format_dist(target, target_distribution(df, target, "forecast_target_mask")))
        category_cols = [f"target_category_{category}" for category in CATEGORIES]
        lines.append(f"multi-label category prevalence by turn count: {dict(Counter(df[category_cols].sum(axis=1)))}")
        user_until = df.loc[df["forecast_target_mask"] == 1, "user_turns_until_first_confirm"].dropna().astype(int)
        lines.append(f"user_turns_until_first_confirm distribution: {dict(Counter(user_until.tolist()))}")
        listener_labeled = df[(df["speaker_role"] != "user") & (df["has_any_crisis_label"] == 1)]
        lines.append(f"listener/unknown labeled anomalies: {len(listener_labeled)}")
        if len(listener_labeled):
            lines.append(listener_labeled[["dialogue_id", "turn_id", "speaker_role", "labels_raw"]].to_string(index=False))
        lines.append(f"validation errors: {validation_errors[split] if validation_errors[split] else 'none'}")
        lines.append("")

    lines.extend(
        [
            "Progression Target Design Audit",
            "-------------------------------",
            "Do not train directly on progression_stage yet because it uses full-dialogue future knowledge.",
            "Current-turn Alert and Confirm detection are causal turn-level tasks.",
            "Early-warning forecasting is valid as a supervised outcome because future labels define the target, but future labels must not be model inputs.",
            "",
            "Future Model Head Recommendation",
            "--------------------------------",
            "Use a shared DeBERTa encoder with sigmoid/BCE-style heads conceptually:",
            "- alert_head: hidden_size -> 256 -> 1",
            "- confirm_head: hidden_size -> 256 -> 1",
            "- category_head: hidden_size -> 256 -> 7",
            "- temporal_state_head: hidden_size -> 256 -> 2",
            "",
            "Forecasting Recommendation",
            "--------------------------",
            "Handle early-warning forecasting primarily in the future hierarchical conversation Transformer, because it is a temporal/history-dependent task.",
            "A turn-level forecasting head can be added later as an ablation.",
            "",
            "Relationship To Severity DeBERTa",
            "--------------------------------",
            "Option A: start independently from pretrained microsoft/deberta-v3-large.",
            "- Advantage: clean separation between severity provenance and CRADLE synthetic/clinician benchmark provenance.",
            "- Disadvantage: does not reuse severity fine-tuning.",
            "",
            "Option B: initialize from the severity-trained DeBERTa checkpoint.",
            "- Advantage: may transfer general mental-health risk features.",
            "- Disadvantage: severity labels and CRADLE labels differ; transfer may bias the CRADLE task.",
            "",
            "Option C: jointly train one encoder across severity and CRADLE tasks.",
            "- Advantage: most integrated multi-dimensional model.",
            "- Disadvantage: harder to audit and may mix weak/separate label provenances.",
            "",
            "Recommendation: start the CRADLE model independently from pretrained microsoft/deberta-v3-large first, then compare severity-checkpoint initialization as a controlled ablation later.",
            "",
            "Validation Summary",
            "------------------",
        ]
    )
    if any(validation_errors.values()):
        lines.append("Some validation checks failed. Do not model until resolved.")
    else:
        lines.append("All validation checks passed.")
    return "\n".join(lines) + "\n"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    processed = {}
    validation_errors = {}
    for split in SPLITS:
        df = pd.read_csv(DATA_DIR / f"cradle_{split}_turns.csv", keep_default_na=False)
        targets = build_targets(df)
        errors = validate_targets(targets, split)
        validation_errors[split] = errors
        if errors:
            raise ValueError(f"{split} target validation failed: {errors}")
        out_path = DATA_DIR / f"cradle_{split}_targets.csv"
        targets.to_csv(out_path, index=False)
        processed[split] = targets

    report = audit_report(processed, validation_errors)
    (RESULTS_DIR / "cradle_model_target_audit.txt").write_text(report)
    print(report)
    print("Saved target files:")
    for split in SPLITS:
        print(f"- {DATA_DIR / f'cradle_{split}_targets.csv'}")
    print(f"- {RESULTS_DIR / 'cradle_model_target_audit.txt'}")


if __name__ == "__main__":
    main()
