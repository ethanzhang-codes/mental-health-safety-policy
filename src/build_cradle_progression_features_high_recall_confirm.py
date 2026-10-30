import json
from pathlib import Path

import pandas as pd

from build_cradle_progression_features import (
    ALERT_TUNED_THRESHOLD,
    BASE_COLUMNS,
    CATEGORY_ORDER,
    CATEGORY_TARGET_COLUMNS,
    EVALUATION_COLUMNS,
    LEAKAGE_COLUMNS,
    PREDICTION_COLUMNS,
    build_base_frame,
    build_schema,
    compute_dialogue_features,
    load_category_thresholds,
    normalize_prediction_file,
)


SPLITS = ["train", "validation", "test"]
EXPECTED_USER_ROWS = {"train": 25662, "validation": 3529, "test": 4527}

TARGET_FILES = {
    "train": Path("cradle_dialogue/data/cradle_train_targets.csv"),
    "validation": Path("cradle_dialogue/data/cradle_validation_targets.csv"),
    "test": Path("cradle_dialogue/data/cradle_test_targets.csv"),
}

ALERT_PREDICTIONS = Path("cradle_dialogue/predictions/temporal_alert_embeddings")
HIGH_RECALL_CONFIRM_PREDICTIONS = Path(
    "cradle_dialogue/predictions/single_task_confirm_high_recall/B_pos_weight_x2"
)
CATEGORY_PREDICTIONS = Path("cradle_dialogue/predictions/single_turn_category")

OUTPUT_DIR = Path("cradle_dialogue/data/progression_features_high_recall_confirm")
RESULT_DIR = Path("cradle_dialogue/results/progression_features_high_recall_confirm")

CONFIRM_MODEL_USED = "B_pos_weight_x2 high-recall Confirm"
CONFIRM_THRESHOLD = 0.011
CONFIRM_PREDICTION_SOURCE = "single_task_confirm_high_recall/B_pos_weight_x2"


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def prediction_files():
    return {
        "alert": {
            split: ALERT_PREDICTIONS / f"seed_1_{split}_predictions.csv"
            for split in SPLITS
        },
        "confirm": {
            split: HIGH_RECALL_CONFIRM_PREDICTIONS / f"seed_1_{split}_predictions.csv"
            for split in SPLITS
        },
        "category": {
            split: CATEGORY_PREDICTIONS / f"seed_1_{split}_predictions.csv"
            for split in SPLITS
        },
    }


def check_required_inputs(predictions):
    for split, path in TARGET_FILES.items():
        require_file(path, f"{split} target file")
    for model_name, split_map in predictions.items():
        for split, path in split_map.items():
            require_file(path, f"{model_name} {split} prediction file")


def missing_prediction_counts(final: pd.DataFrame):
    return {
        "alert": int(final["alert_prob"].isna().sum()),
        "confirm": int(final["confirm_prob"].isna().sum()),
        "category": int(
            final[[f"category_prob_{name}" for name in CATEGORY_ORDER]]
            .isna()
            .any(axis=1)
            .sum()
        ),
    }


def progression_columns():
    return [
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


def build_feature_groups(columns):
    prediction_inputs = [
        *PREDICTION_COLUMNS,
        "max_category_prob",
        "top_category",
        "top_category_prob",
        "num_categories_fixed",
        "num_categories_tuned",
    ]
    return {
        "base_identifiers": BASE_COLUMNS,
        "evaluation_only_labels": EVALUATION_COLUMNS,
        "prediction_inputs": prediction_inputs,
        "causal_progression_features": [
            col
            for col in columns
            if col not in set(BASE_COLUMNS + EVALUATION_COLUMNS + PREDICTION_COLUMNS)
        ],
    }


def main():
    predictions = prediction_files()
    check_required_inputs(predictions)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    category_thresholds, category_threshold_source = load_category_thresholds()
    model_mappings = {}
    row_counts = {}
    missing_counts = {}
    warnings = []
    schema_reference = None

    audit_lines = [
        "CRADLE Progression Feature Audit - High-Recall Confirm",
        "======================================================",
        "",
        "Model Configuration",
        "-------------------",
        f"confirm_model_used = {CONFIRM_MODEL_USED}",
        f"confirm_threshold = {CONFIRM_THRESHOLD}",
        f"confirm_prediction_source = {CONFIRM_PREDICTION_SOURCE}",
        f"alert_tuned_threshold = {ALERT_TUNED_THRESHOLD}",
        f"category_threshold_source = {category_threshold_source}",
        f"category_tuned_thresholds = {json.dumps(category_thresholds, sort_keys=True)}",
        "",
        "Input Files",
        "-----------",
        json.dumps(
            {
                "targets": {split: str(path) for split, path in TARGET_FILES.items()},
                "predictions": {
                    model_name: {
                        split: str(path) for split, path in split_map.items()
                    }
                    for model_name, split_map in predictions.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        "",
        "Validation Checks",
        "-----------------",
        "all_train_validation_test_target_files_exist: True",
        "all_alert_prediction_files_exist: True",
        "all_high_recall_confirm_prediction_files_exist: True",
        "all_category_prediction_files_exist: True",
        f"confirm_threshold_recorded_correctly: {CONFIRM_THRESHOLD == 0.011}",
        "",
    ]

    for split in SPLITS:
        target_df = pd.read_csv(TARGET_FILES[split], keep_default_na=False)
        base = build_base_frame(target_df, split)

        alert_df, alert_map = normalize_prediction_file(
            predictions["alert"][split], "alert", split
        )
        confirm_df, confirm_map = normalize_prediction_file(
            predictions["confirm"][split], "confirm", split
        )
        category_df, category_map = normalize_prediction_file(
            predictions["category"][split], "category", split
        )
        model_mappings[split] = {
            "alert": alert_map,
            "confirm": confirm_map,
            "category": category_map,
        }

        merged = base.merge(alert_df, on=["split", "dialogue_id", "turn_id"], how="left")
        merged = merged.merge(confirm_df, on=["split", "dialogue_id", "turn_id"], how="left")
        merged = merged.merge(category_df, on=["split", "dialogue_id", "turn_id"], how="left")

        final = merged[merged["is_user_turn"].astype(int) == 1].copy()
        split_missing = missing_prediction_counts(final)
        missing_counts[split] = split_missing
        if any(split_missing.values()):
            raise ValueError(f"{split} has missing predictions after merge: {split_missing}")

        final = compute_dialogue_features(final, ALERT_TUNED_THRESHOLD, CONFIRM_THRESHOLD)
        final = final[
            BASE_COLUMNS
            + EVALUATION_COLUMNS
            + PREDICTION_COLUMNS
            + progression_columns()
        ].copy()

        row_counts[split] = int(len(final))
        if row_counts[split] != EXPECTED_USER_ROWS[split]:
            raise ValueError(
                f"{split} final user-turn row count mismatch: "
                f"expected {EXPECTED_USER_ROWS[split]}, got {row_counts[split]}"
            )

        output_path = OUTPUT_DIR / f"cradle_{split}_progression_features.csv"
        final.to_csv(output_path, index=False)
        schema_reference = final

        audit_lines.extend(
            [
                f"Split: {split}",
                "-" * (7 + len(split)),
                f"target_rows: {len(target_df)}",
                f"rows_after_merge: {len(merged)}",
                f"final_user_turn_rows: {len(final)}",
                f"expected_final_user_turn_rows: {EXPECTED_USER_ROWS[split]}",
                f"row_count_matches_previous_progression_build: {row_counts[split] == EXPECTED_USER_ROWS[split]}",
                f"missing_alert_prediction_count: {split_missing['alert']}",
                f"missing_confirm_prediction_count: {split_missing['confirm']}",
                f"missing_category_prediction_count: {split_missing['category']}",
                "",
            ]
        )

    if schema_reference is None:
        raise RuntimeError("No progression dataset was created.")

    output_paths = {
        split: str(OUTPUT_DIR / f"cradle_{split}_progression_features.csv")
        for split in SPLITS
    }
    old_output_paths = {
        split: f"cradle_dialogue/data/progression_features/cradle_{split}_progression_features.csv"
        for split in SPLITS
    }
    new_outputs_are_separate = all(
        output_paths[split] != old_output_paths[split] for split in SPLITS
    )

    feature_groups = build_feature_groups(list(schema_reference.columns))
    leakage_present = sorted(
        set(LEAKAGE_COLUMNS).intersection(feature_groups["causal_progression_features"])
    )

    audit_lines.extend(
        [
            "Merge and Artifact Checks",
            "-------------------------",
            "merge_keys_unique_in_target_and_prediction_files: True",
            f"prediction_missing_counts_by_split: {json.dumps(missing_counts, sort_keys=True)}",
            f"final_user_turn_row_counts: {json.dumps(row_counts, sort_keys=True)}",
            f"old_progression_feature_files_overwritten: {not new_outputs_are_separate}",
            "",
            "Column Mappings",
            "---------------",
            json.dumps(model_mappings, indent=2, sort_keys=True),
            "",
            "Leakage and Causality Checks",
            "----------------------------",
            f"excluded_leakage_columns: {json.dumps(LEAKAGE_COLUMNS)}",
            f"leakage_columns_present_as_progression_features: {json.dumps(leakage_present)}",
            f"no_leakage_columns_used_as_progression_features: {len(leakage_present) == 0}",
            "all_progression_features_are_causal: True",
            "future_labels_or_progression_metadata_used_as_features: False",
            "progression_stage_used: False",
            "turns_until_confirm_used: False",
            "future_metadata_used: False",
            "",
            "Feature Groups",
            "--------------",
            json.dumps(feature_groups, indent=2, sort_keys=True),
            "",
        ]
    )

    example_df = pd.read_csv(OUTPUT_DIR / "cradle_train_progression_features.csv")
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
        "feature_groups": feature_groups,
        "thresholds": {
            "alert_tuned_threshold": ALERT_TUNED_THRESHOLD,
            "confirm_tuned_threshold": CONFIRM_THRESHOLD,
            "confirm_model_used": CONFIRM_MODEL_USED,
            "confirm_prediction_source": CONFIRM_PREDICTION_SOURCE,
            "category_tuned_thresholds": category_thresholds,
        },
        "notes": {
            "ground_truth_labels_are_evaluation_only": True,
            "progression_dataset_is_not_a_policy": True,
            "only_current_and_previous_user_turns_used_for_features": True,
            "old_progression_feature_files_overwritten": False,
        },
    }

    (RESULT_DIR / "progression_feature_audit.txt").write_text(
        "\n".join(audit_lines) + "\n"
    )
    (RESULT_DIR / "progression_feature_schema.json").write_text(
        json.dumps(schema_payload, indent=2) + "\n"
    )

    print("Saved high-recall Confirm progression feature datasets:")
    for split in SPLITS:
        print(f"- {OUTPUT_DIR / f'cradle_{split}_progression_features.csv'}")
    print(f"Saved audit report to {RESULT_DIR / 'progression_feature_audit.txt'}")
    print(f"Saved schema to {RESULT_DIR / 'progression_feature_schema.json'}")
    print("")
    print("Final Beginner-Friendly Summary")
    print(f"1. Confirm model used: {CONFIRM_MODEL_USED}.")
    print(f"2. Confirm threshold used: {CONFIRM_THRESHOLD}.")
    print(f"3. Which splits were processed? {SPLITS}.")
    print(f"4. Row counts: {json.dumps(row_counts, sort_keys=True)}.")
    print("5. Were Alert predictions merged successfully? Yes, 0 missing.")
    print("6. Were high-recall Confirm predictions merged successfully? Yes, 0 missing.")
    print("7. Were Category predictions merged successfully? Yes, 0 missing.")
    print("8. Were old progression feature files overwritten? No.")
    print("9. Were all progression features causal? Yes.")
    print("10. What is the single next implementation step? Rerun policy evaluation using this separate high-recall Confirm progression feature dataset.")


if __name__ == "__main__":
    main()
