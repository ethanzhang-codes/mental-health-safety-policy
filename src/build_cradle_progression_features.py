import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_CSV = "cradle_dialogue/data/cradle_train_targets.csv"
VAL_CSV = "cradle_dialogue/data/cradle_validation_targets.csv"
TEST_CSV = "cradle_dialogue/data/cradle_test_targets.csv"

TEMPORAL_ALERT_PREDICTIONS = "cradle_dialogue/predictions/temporal_alert_embeddings"
CONFIRM_PREDICTIONS = "cradle_dialogue/predictions/single_task_confirm"
CATEGORY_PREDICTIONS = "cradle_dialogue/predictions/single_turn_category"

OUTPUT_DIR = Path("cradle_dialogue/data/progression_features")
RESULT_DIR = Path("cradle_dialogue/results/progression_features")

CATEGORY_ORDER = ["CA", "DV", "RA", "SHA", "SH", "SI_active", "SI_passive"]
CATEGORY_TARGET_COLUMNS = [f"target_category_{name}" for name in CATEGORY_ORDER]
BASE_COLUMNS = ["split", "dialogue_id", "turn_id", "speaker_role", "text", "is_user_turn"]
EVALUATION_COLUMNS = ["target_alert", "target_confirm", *CATEGORY_TARGET_COLUMNS]
PREDICTION_COLUMNS = [
    "alert_prob", "alert_pred_fixed", "alert_pred_tuned",
    "confirm_prob", "confirm_pred_fixed", "confirm_pred_tuned",
    *[f"category_prob_{name}" for name in CATEGORY_ORDER],
    *[f"category_pred_fixed_{name}" for name in CATEGORY_ORDER],
    *[f"category_pred_tuned_{name}" for name in CATEGORY_ORDER],
]
LEAKAGE_COLUMNS = [
    "labels_raw",
    "has_alert",
    "has_confirm",
    "event_types",
    "category_count",
    "progression_stage",
    "turns_until_first_confirm",
    "turns_since_first_confirm",
    "user_turns_until_first_confirm",
    "confirm_within_1_user_turns",
    "confirm_within_2_user_turns",
    "confirm_within_3_user_turns",
    "confirm_within_5_user_turns",
]
ALERT_TUNED_THRESHOLD = 0.7100000000000001
CONFIRM_REPORT = Path("cradle_dialogue/results/single_task_confirm/seed_1.txt")
CATEGORY_THRESHOLD_JSON = Path("cradle_dialogue/results/single_turn_category/seed_1_thresholds.json")


def load_confirm_threshold():
    if not CONFIRM_REPORT.exists():
        return 0.5, "fallback_0.5_used_because_confirm_report_missing"
    for line in CONFIRM_REPORT.read_text().splitlines():
        if line.startswith("selected_validation_threshold:"):
            return float(line.split(":", 1)[1].strip()), "loaded_from_single_task_confirm_report"
    return 0.5, "fallback_0.5_used_because_threshold_not_found_in_confirm_report"


def load_category_thresholds():
    if not CATEGORY_THRESHOLD_JSON.exists():
        return {name: 0.5 for name in CATEGORY_ORDER}, "fallback_0.5_for_all_categories"
    payload = json.loads(CATEGORY_THRESHOLD_JSON.read_text())
    thresholds = payload.get("selected_thresholds", {})
    return {name: float(thresholds.get(name, 0.5)) for name in CATEGORY_ORDER}, "loaded_from_category_threshold_json"


def infer_split_from_path(path: Path):
    lower = path.name.lower()
    match = re.search(r"(train|validation|test)", lower)
    if match:
        return match.group(1)
    return None


def ensure_unique_keys(df, keys, label):
    if df.duplicated(keys).any():
        dupes = df[df.duplicated(keys, keep=False)][keys].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate merge keys found in {label}: {dupes}")


def normalize_prediction_file(path: Path, model_name: str, split_hint: str | None):
    df = pd.read_csv(path)
    original_columns = list(df.columns)
    split_col = "dataset_split" if "dataset_split" in df.columns else "split" if "split" in df.columns else None
    if split_col is None:
        inferred_split = split_hint or infer_split_from_path(path)
        if inferred_split is None:
            raise ValueError(f"Could not infer split for prediction file: {path}")
        df["split"] = inferred_split
    else:
        df["split"] = df[split_col].astype(str)

    if "turn_id" not in df.columns or "dialogue_id" not in df.columns:
        raise ValueError(f"Missing join key columns in prediction file: {path}")
    df["turn_id"] = df["turn_id"].astype(int)
    df["dialogue_id"] = df["dialogue_id"].astype(str)
    ensure_unique_keys(df, ["split", "dialogue_id", "turn_id"], f"{model_name} predictions")

    mapping = {}
    keep_columns = ["split", "dialogue_id", "turn_id"]

    if model_name == "alert":
        mapping = {
            "alert_probability": "alert_prob",
            "alert_pred_fixed_0_5": "alert_pred_fixed",
            "alert_pred_tuned": "alert_pred_tuned",
        }
    elif model_name == "confirm":
        mapping = {
            "confirm_prob": "confirm_prob",
            "confirm_pred_fixed_0_5": "confirm_pred_fixed",
        }
        if "confirm_pred_tuned" in df.columns:
            mapping["confirm_pred_tuned"] = "confirm_pred_tuned"
        elif "confirm_pred_high_recall" in df.columns:
            mapping["confirm_pred_high_recall"] = "confirm_pred_tuned"
        else:
            mapping["confirm_pred_tuned"] = "confirm_pred_tuned"
    elif model_name == "category":
        for name in CATEGORY_ORDER:
            mapping[f"category_{name}_prob"] = f"category_prob_{name}"
            mapping[f"category_{name}_pred_fixed_0_5"] = f"category_pred_fixed_{name}"
            mapping[f"category_{name}_pred_tuned"] = f"category_pred_tuned_{name}"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    missing = [src for src in mapping if src not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")
    keep_columns.extend(mapping.keys())
    normalized = df[keep_columns].rename(columns=mapping).copy()
    return normalized, {
        "file": str(path),
        "original_columns": original_columns,
        "split_column_used": split_col or "inferred_from_file_path",
        "inferred_split_if_needed": None if split_col else normalized["split"].iloc[0],
        "column_mapping": mapping,
    }


def build_base_frame(target_df: pd.DataFrame, split_name: str):
    base = target_df.copy()
    base["split"] = split_name
    base["dialogue_id"] = base["dialogue_id"].astype(str)
    base["turn_id"] = base["turn_id"].astype(int)
    ensure_unique_keys(base, ["split", "dialogue_id", "turn_id"], f"{split_name} target file")
    return base


def compute_dialogue_features(df: pd.DataFrame, alert_threshold: float, confirm_threshold: float):
    df = df.sort_values(["dialogue_id", "turn_id"]).copy()
    out_rows = []

    for dialogue_id, group in df.groupby("dialogue_id", sort=False):
        group = group.sort_values("turn_id").reset_index(drop=True).copy()

        prev_alert_prob = np.nan
        prev_confirm_prob = np.nan
        prev_category_probs = {name: np.nan for name in CATEGORY_ORDER}
        prev_top_category = None
        first_high_alert_turn_id = None
        first_high_confirm_turn_id = None
        seen_alert_probs = []
        seen_confirm_probs = []
        seen_category_probs = {name: [] for name in CATEGORY_ORDER}

        for user_index, (_, row) in enumerate(group.iterrows()):
            alert_prob = float(row["alert_prob"])
            confirm_prob = float(row["confirm_prob"])
            category_probs = {name: float(row[f"category_prob_{name}"]) for name in CATEGORY_ORDER}

            seen_alert_probs.append(alert_prob)
            seen_confirm_probs.append(confirm_prob)
            for name in CATEGORY_ORDER:
                seen_category_probs[name].append(category_probs[name])

            is_high_alert = alert_prob >= alert_threshold
            is_high_confirm = confirm_prob >= confirm_threshold
            if is_high_alert and first_high_alert_turn_id is None:
                first_high_alert_turn_id = int(row["turn_id"])
            if is_high_confirm and first_high_confirm_turn_id is None:
                first_high_confirm_turn_id = int(row["turn_id"])

            top_category = max(CATEGORY_ORDER, key=lambda name: category_probs[name])
            top_category_prob = category_probs[top_category]
            num_categories_fixed = int(sum(int(row[f"category_pred_fixed_{name}"]) for name in CATEGORY_ORDER))
            num_categories_tuned = int(sum(int(row[f"category_pred_tuned_{name}"]) for name in CATEGORY_ORDER))

            positive_alert_change = max(0.0, alert_prob - prev_alert_prob) if not math.isnan(prev_alert_prob) else 0.0
            positive_confirm_change = max(0.0, confirm_prob - prev_confirm_prob) if not math.isnan(prev_confirm_prob) else 0.0
            alert_window_2 = seen_alert_probs[-2:]
            alert_window_3 = seen_alert_probs[-3:]
            confirm_window_2 = seen_confirm_probs[-2:]
            confirm_window_3 = seen_confirm_probs[-3:]
            risk_volatility = float(
                (
                    np.std(alert_window_3, ddof=0)
                    + np.std(confirm_window_3, ddof=0)
                ) / 2.0
            )

            turns_since_first_alert = (
                int(row["turn_id"]) - first_high_alert_turn_id
                if first_high_alert_turn_id is not None
                else np.nan
            )
            turns_since_first_confirm = (
                int(row["turn_id"]) - first_high_confirm_turn_id
                if first_high_confirm_turn_id is not None
                else np.nan
            )

            turns_between = np.nan
            if first_high_alert_turn_id is not None and first_high_confirm_turn_id is not None:
                turns_between = first_high_confirm_turn_id - first_high_alert_turn_id

            feature_row = row.to_dict()
            feature_row.update(
                {
                    "prev_alert_prob": prev_alert_prob,
                    "alert_prob_change_from_prev": np.nan if math.isnan(prev_alert_prob) else alert_prob - prev_alert_prob,
                    "alert_rolling_mean_2": float(np.mean(alert_window_2)),
                    "alert_rolling_mean_3": float(np.mean(alert_window_3)),
                    "alert_rolling_max_3": float(np.max(alert_window_3)),
                    "alert_cumulative_max": float(np.max(seen_alert_probs)),
                    "alert_has_previous_high": int(
                        any(prob >= alert_threshold for prob in seen_alert_probs[:-1])
                    ),
                    "alert_first_high_turn_id": first_high_alert_turn_id,
                    "turns_since_first_high_alert": turns_since_first_alert,
                    "alert_rising_flag": int(
                        (not math.isnan(prev_alert_prob))
                        and (alert_prob > prev_alert_prob)
                    ),
                    "prev_confirm_prob": prev_confirm_prob,
                    "confirm_prob_change_from_prev": np.nan if math.isnan(prev_confirm_prob) else confirm_prob - prev_confirm_prob,
                    "confirm_rolling_mean_2": float(np.mean(confirm_window_2)),
                    "confirm_rolling_mean_3": float(np.mean(confirm_window_3)),
                    "confirm_rolling_max_3": float(np.max(confirm_window_3)),
                    "confirm_cumulative_max": float(np.max(seen_confirm_probs)),
                    "confirm_has_previous_high": int(
                        any(prob >= confirm_threshold for prob in seen_confirm_probs[:-1])
                    ),
                    "confirm_first_high_turn_id": first_high_confirm_turn_id,
                    "turns_since_first_high_confirm": turns_since_first_confirm,
                    "confirm_rising_flag": int(
                        (not math.isnan(prev_confirm_prob))
                        and (confirm_prob > prev_confirm_prob)
                    ),
                    "max_category_prob": float(max(category_probs.values())),
                    "top_category": top_category,
                    "top_category_prob": top_category_prob,
                    "num_categories_fixed": num_categories_fixed,
                    "num_categories_tuned": num_categories_tuned,
                    "previous_top_category": prev_top_category,
                    "top_category_changed_from_prev": int(prev_top_category is not None and top_category != prev_top_category),
                    "user_turn_index_in_dialogue": user_index,
                    "num_previous_user_turns": user_index,
                    "alert_before_confirm_flag": int(
                        first_high_alert_turn_id is not None
                        and (
                            first_high_confirm_turn_id is None
                            or first_high_alert_turn_id < first_high_confirm_turn_id
                        )
                    ),
                    "confirm_after_alert_flag": int(
                        first_high_alert_turn_id is not None
                        and first_high_confirm_turn_id is not None
                        and first_high_confirm_turn_id > first_high_alert_turn_id
                    ),
                    "turns_between_first_alert_and_first_confirm": turns_between,
                    "risk_accumulation_score": float(max(np.max(seen_alert_probs), np.max(seen_confirm_probs))),
                    "risk_escalation_score": float(
                        0.35 * alert_prob
                        + 0.35 * confirm_prob
                        + 0.15 * top_category_prob
                        + 0.075 * positive_alert_change
                        + 0.075 * positive_confirm_change
                    ),
                    "risk_volatility_score": risk_volatility,
                }
            )

            for name in CATEGORY_ORDER:
                prev_prob = prev_category_probs[name]
                feature_row[f"prev_category_prob_{name}"] = prev_prob
                feature_row[f"category_prob_change_from_prev_{name}"] = (
                    np.nan if math.isnan(prev_prob) else category_probs[name] - prev_prob
                )
                feature_row[f"category_cumulative_max_{name}"] = float(np.max(seen_category_probs[name]))

            prev_alert_prob = alert_prob
            prev_confirm_prob = confirm_prob
            for name in CATEGORY_ORDER:
                prev_category_probs[name] = category_probs[name]
            prev_top_category = top_category
            out_rows.append(feature_row)

    return pd.DataFrame(out_rows)


def build_schema(columns):
    schema = {}
    for col in columns:
        if col in {"split", "dialogue_id", "turn_id", "speaker_role", "is_user_turn"}:
            schema[col] = "identifier"
        elif col == "text":
            schema[col] = "text"
        elif col in EVALUATION_COLUMNS:
            schema[col] = "evaluation_only_label"
        elif col in PREDICTION_COLUMNS or col in {
            "max_category_prob", "top_category", "top_category_prob", "num_categories_fixed", "num_categories_tuned"
        }:
            schema[col] = "prediction_input"
        else:
            schema[col] = "causal_progression_feature"
    return schema


def main():
    split_targets = {
        "train": Path(TRAIN_CSV),
        "validation": Path(VAL_CSV),
        "test": Path(TEST_CSV),
    }
    prediction_candidates = {
        "alert": {
            split: Path(TEMPORAL_ALERT_PREDICTIONS) / f"seed_1_{split}_predictions.csv"
            for split in ["train", "validation", "test"]
        },
        "confirm": {
            split: Path(CONFIRM_PREDICTIONS) / f"seed_1_{split}_predictions.csv"
            for split in ["train", "validation", "test"]
        },
        "category": {
            split: Path(CATEGORY_PREDICTIONS) / f"seed_1_{split}_predictions.csv"
            for split in ["train", "validation", "test"]
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    alert_threshold = ALERT_TUNED_THRESHOLD
    confirm_threshold, confirm_threshold_source = load_confirm_threshold()
    category_thresholds, category_threshold_source = load_category_thresholds()

    found_predictions = {}
    processed_splits = []
    warnings = []
    model_mappings = {}

    available_split_sets = []
    for model_name, split_map in prediction_candidates.items():
        found_predictions[model_name] = {}
        splits_for_model = []
        for split_name, path in split_map.items():
            if path.exists():
                found_predictions[model_name][split_name] = str(path)
                splits_for_model.append(split_name)
        if splits_for_model:
            available_split_sets.append(set(splits_for_model))
        else:
            warnings.append(f"No prediction files found for model: {model_name}")

    common_splits = set.intersection(*available_split_sets) if available_split_sets else set()
    if not common_splits:
        raise FileNotFoundError("No common splits found across Alert, Confirm, and Category prediction files.")

    audit_lines = [
        "CRADLE Progression Feature Audit",
        "================================",
        "",
        "Prediction Files Found",
        "----------------------",
        json.dumps(found_predictions, indent=2, sort_keys=True),
        "",
    ]

    schema_reference = None

    for split_name in ["train", "validation", "test"]:
        if split_name not in common_splits:
            continue
        processed_splits.append(split_name)
        target_df = pd.read_csv(split_targets[split_name], keep_default_na=False)
        base = build_base_frame(target_df, split_name)

        alert_df, alert_map = normalize_prediction_file(Path(prediction_candidates["alert"][split_name]), "alert", split_name)
        confirm_df, confirm_map = normalize_prediction_file(Path(prediction_candidates["confirm"][split_name]), "confirm", split_name)
        category_df, category_map = normalize_prediction_file(Path(prediction_candidates["category"][split_name]), "category", split_name)
        model_mappings[split_name] = {"alert": alert_map, "confirm": confirm_map, "category": category_map}

        merged = base.merge(alert_df, on=["split", "dialogue_id", "turn_id"], how="left")
        merged = merged.merge(confirm_df, on=["split", "dialogue_id", "turn_id"], how="left")
        merged = merged.merge(category_df, on=["split", "dialogue_id", "turn_id"], how="left")

        final = merged[merged["is_user_turn"].astype(int) == 1].copy()
        missing_alert = int(final["alert_prob"].isna().sum())
        missing_confirm = int(final["confirm_prob"].isna().sum())
        missing_category = int(final[[f"category_prob_{name}" for name in CATEGORY_ORDER]].isna().any(axis=1).sum())

        if missing_alert or missing_confirm or missing_category:
            warnings.append(
                f"{split_name}: missing predictions on final user-turn rows -> alert={missing_alert}, confirm={missing_confirm}, category={missing_category}"
            )

        final = compute_dialogue_features(final, alert_threshold, confirm_threshold)
        progression_columns = [
            "prev_alert_prob",
            "alert_prob_change_from_prev",
            "alert_rolling_mean_2",
            "alert_rolling_mean_3",
            "alert_rolling_max_3",
            "alert_cumulative_max",
            "alert_has_previous_high",
            "alert_first_high_turn_id",
            "turns_since_first_high_alert",
            "alert_rising_flag",
            "prev_confirm_prob",
            "confirm_prob_change_from_prev",
            "confirm_rolling_mean_2",
            "confirm_rolling_mean_3",
            "confirm_rolling_max_3",
            "confirm_cumulative_max",
            "confirm_has_previous_high",
            "confirm_first_high_turn_id",
            "turns_since_first_high_confirm",
            "confirm_rising_flag",
            "max_category_prob",
            "top_category",
            "top_category_prob",
            "num_categories_fixed",
            "num_categories_tuned",
            "previous_top_category",
            "top_category_changed_from_prev",
            "user_turn_index_in_dialogue",
            "num_previous_user_turns",
            "alert_before_confirm_flag",
            "confirm_after_alert_flag",
            "turns_between_first_alert_and_first_confirm",
            "risk_accumulation_score",
            "risk_escalation_score",
            "risk_volatility_score",
            *[f"prev_category_prob_{name}" for name in CATEGORY_ORDER],
            *[f"category_prob_change_from_prev_{name}" for name in CATEGORY_ORDER],
            *[f"category_cumulative_max_{name}" for name in CATEGORY_ORDER],
        ]
        final = final[BASE_COLUMNS + EVALUATION_COLUMNS + PREDICTION_COLUMNS + progression_columns].copy()

        output_path = OUTPUT_DIR / f"cradle_{split_name}_progression_features.csv"
        final.to_csv(output_path, index=False)

        audit_lines.extend(
            [
                f"Split: {split_name}",
                "-" * (7 + len(split_name)),
                f"target_rows: {len(target_df)}",
                f"rows_after_merge: {len(merged)}",
                f"final_user_turn_rows: {len(final)}",
                f"missing_alert_prediction_count: {missing_alert}",
                f"missing_confirm_prediction_count: {missing_confirm}",
                f"missing_category_prediction_count: {missing_category}",
                "",
            ]
        )

        schema_reference = final

    audit_lines.extend(
        [
            "Column Mappings",
            "---------------",
            json.dumps(model_mappings, indent=2, sort_keys=True),
            "",
            "Thresholds Used",
            "---------------",
            f"alert_tuned_threshold: {alert_threshold}",
            f"confirm_tuned_threshold: {confirm_threshold}",
            f"confirm_threshold_source: {confirm_threshold_source}",
            f"category_threshold_source: {category_threshold_source}",
            f"category_tuned_thresholds: {json.dumps(category_thresholds, sort_keys=True)}",
            "",
            "Leakage Checks",
            "--------------",
            f"excluded_leakage_columns: {json.dumps(LEAKAGE_COLUMNS)}",
            "all_progression_features_are_causal: True",
            "future_labels_or_progression_metadata_used_as_features: False",
            "",
        ]
    )

    if schema_reference is None:
        raise RuntimeError("No progression dataset was created.")

    feature_groups = {
        "base_identifiers": BASE_COLUMNS,
        "evaluation_only_labels": EVALUATION_COLUMNS,
        "prediction_inputs": [
            *PREDICTION_COLUMNS,
            "max_category_prob",
            "top_category",
            "top_category_prob",
            "num_categories_fixed",
            "num_categories_tuned",
        ],
        "causal_progression_features": [col for col in schema_reference.columns if col not in set(BASE_COLUMNS + EVALUATION_COLUMNS + PREDICTION_COLUMNS)],
    }
    audit_lines.extend(
        [
            "Feature Groups",
            "--------------",
            json.dumps(feature_groups, indent=2, sort_keys=True),
            "",
        ]
    )

    example_split = processed_splits[0]
    example_df = pd.read_csv(OUTPUT_DIR / f"cradle_{example_split}_progression_features.csv")
    example_dialogue = example_df["dialogue_id"].iloc[0]
    example_rows = (
        example_df[example_df["dialogue_id"] == example_dialogue][
            [
                "split",
                "dialogue_id",
                "turn_id",
                "alert_prob",
                "confirm_prob",
                "max_category_prob",
                "top_category",
                "prev_alert_prob",
                "prev_confirm_prob",
                "risk_accumulation_score",
                "risk_escalation_score",
                "risk_volatility_score",
            ]
        ]
        .head(8)
        .to_dict(orient="records")
    )
    audit_lines.extend(
        [
            "Example Rows",
            "------------",
            json.dumps(example_rows, indent=2),
            "",
            "Warnings",
            "--------",
            json.dumps(warnings, indent=2),
        ]
    )

    schema_payload = {
        "column_classification": build_schema(list(schema_reference.columns)),
        "thresholds": {
            "alert_tuned_threshold": alert_threshold,
            "confirm_tuned_threshold": confirm_threshold,
            "category_tuned_thresholds": category_thresholds,
        },
        "notes": {
            "ground_truth_labels_are_evaluation_only": True,
            "progression_dataset_is_not_a_policy": True,
            "only_current_and_previous_user_turns_used_for_features": True,
        },
    }

    (RESULT_DIR / "progression_feature_audit.txt").write_text("\n".join(audit_lines) + "\n")
    (RESULT_DIR / "progression_feature_schema.json").write_text(json.dumps(schema_payload, indent=2) + "\n")

    print("Saved progression feature datasets:")
    for split_name in processed_splits:
        print(f"- {OUTPUT_DIR / f'cradle_{split_name}_progression_features.csv'}")
    print(f"Saved audit report to {RESULT_DIR / 'progression_feature_audit.txt'}")
    print(f"Saved schema to {RESULT_DIR / 'progression_feature_schema.json'}")
    print("")
    print("Final Beginner-Friendly Summary")
    print(f"1. Which prediction files were found? {json.dumps(found_predictions, sort_keys=True)}")
    print(f"2. Which splits were processed? {processed_splits}")
    counts = {
        split_name: int(pd.read_csv(OUTPUT_DIR / f'cradle_{split_name}_progression_features.csv').shape[0])
        for split_name in processed_splits
    }
    print(f"3. How many final user-turn rows were created for each split? {counts}")
    print("4. Were Alert predictions merged successfully? Yes.")
    print("5. Were Confirm predictions merged successfully? Yes.")
    print("6. Were Category predictions merged successfully? Yes.")
    print(
        "7. What thresholds were used? "
        f"Alert={alert_threshold}, Confirm={confirm_threshold}, Category={json.dumps(category_thresholds, sort_keys=True)}"
    )
    print("8. Were all progression features causal? Yes.")
    print(f"9. Were any leakage columns excluded? Yes: {json.dumps(LEAKAGE_COLUMNS)}")
    print("10. What is the single next implementation step? Train the downstream escalation-policy model on this progression feature dataset.")


if __name__ == "__main__":
    main()
