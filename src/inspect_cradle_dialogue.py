import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download


DATASET_NAME = "SungJoo/Cradle-Dialogue"
ROOT = Path("cradle_dialogue")
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SPLITS = ["train", "validation", "test"]
TEMPORAL_STATES = {"ongoing", "past"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict structural, label, temporal, and leakage audit for CRADLE-Dialogue."
    )
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--save_split_csv", action="store_true", default=True)
    return parser.parse_args()


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).lower().strip())


def stable_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_labels(value):
    if value is None:
        return []
    value = str(value).strip()
    if value == "":
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def infer_dialogue_ids(df, split):
    inferred = []
    current = -1
    for turn_id in df["turn_id"].tolist():
        if int(turn_id) == 0:
            current += 1
        if current < 0:
            current = 0
        inferred.append(f"{split}_inferred_dialogue_{current:05d}")
    out = df.copy()
    out.insert(0, "dialogue_id", inferred)
    out.insert(1, "dialogue_index", [int(x.rsplit("_", 1)[-1]) for x in inferred])
    return out


def speaker_prefix(text):
    match = re.match(r"^\s*([^:\n]{1,40}):", str(text))
    return match.group(1).strip() if match else "UNKNOWN"


def speaker_role(prefix):
    if prefix == "User":
        return "User"
    if prefix == "Listener":
        return "Listener"
    return "UNKNOWN"


def parse_label(label):
    if label in {"alert_ongoing", "alert_past"}:
        _, temporal = label.split("_", 1)
        return {
            "exact_label": label,
            "event_type": "alert",
            "crisis_category": "",
            "temporal_state": temporal,
            "parse_status": "parsed",
        }
    if label.startswith("confirm_"):
        parts = label.split("_")
        temporal = parts[-1] if parts else ""
        category = "_".join(parts[1:-1])
        if temporal in TEMPORAL_STATES and category:
            return {
                "exact_label": label,
                "event_type": "confirm",
                "crisis_category": category,
                "temporal_state": temporal,
                "parse_status": "parsed",
            }
    return {
        "exact_label": label,
        "event_type": "",
        "crisis_category": "",
        "temporal_state": "",
        "parse_status": "unparsed",
    }


def row_duplicate_count(df):
    return int(df.duplicated().sum())


def turn_stats(df):
    grouped = df.groupby("dialogue_id")["turn_id"].agg(list)
    records = []
    for dialogue_id, turns in grouped.items():
        int_turns = [int(x) for x in turns]
        sorted_turns = sorted(int_turns)
        expected = list(range(len(sorted_turns)))
        records.append(
            {
                "dialogue_id": dialogue_id,
                "begins_at_0": bool(int_turns[0] == 0),
                "contiguous": sorted_turns == expected,
                "has_gaps": sorted_turns != expected,
                "duplicate_turn_indices": len(set(int_turns)) != len(int_turns),
                "sortable_as_integer": True,
                "turn_count": len(int_turns),
                "min_turn_id": min(int_turns),
                "max_turn_id": max(int_turns),
            }
        )
    return pd.DataFrame(records)


def speaker_audit(df):
    df = df.copy()
    df["speaker_prefix"] = df["text"].map(speaker_prefix)
    df["speaker_role"] = df["speaker_prefix"].map(speaker_role)
    unusual = []
    final_roles = Counter()
    first_roles = Counter()
    alternation_ok = 0
    for dialogue_id, group in df.groupby("dialogue_id", sort=False):
        roles = group["speaker_role"].tolist()
        first_roles[roles[0]] += 1
        final_roles[roles[-1]] += 1
        ok = all(roles[i] != roles[i - 1] for i in range(1, len(roles)))
        if ok:
            alternation_ok += 1
        else:
            unusual.append(dialogue_id)
    return {
        "prefix_counts": Counter(df["speaker_prefix"].tolist()),
        "role_counts": Counter(df["speaker_role"].tolist()),
        "unknown_count": int((df["speaker_role"] == "UNKNOWN").sum()),
        "alternation_ok_dialogues": alternation_ok,
        "alternation_problem_dialogues": unusual,
        "first_roles": first_roles,
        "final_roles": final_roles,
    }


def label_rows(all_data):
    rows = []
    dialogue_by_label = defaultdict(set)
    for split, df in all_data.items():
        for _, row in df.iterrows():
            labels = split_labels(row["labels"])
            for label in labels:
                rows.append(
                    {
                        "split": split,
                        "dialogue_id": row["dialogue_id"],
                        "turn_id": int(row["turn_id"]),
                        "exact_label": label,
                    }
                )
                dialogue_by_label[label].add(row["dialogue_id"])
    return pd.DataFrame(rows), dialogue_by_label


def write_label_distribution(all_data):
    labels_df, dialogue_by_label = label_rows(all_data)
    vocab = sorted(labels_df["exact_label"].unique()) if not labels_df.empty else []
    rows = []
    for label in vocab:
        row = {"exact_label": label}
        total = int((labels_df["exact_label"] == label).sum())
        row["total_turn_count"] = total
        for split in SPLITS:
            row[f"{split}_count"] = int(
                ((labels_df["exact_label"] == label) & (labels_df["split"] == split)).sum()
            )
        row["unique_dialogues_containing_label"] = len(dialogue_by_label[label])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "cradle_dialogue_label_distribution.csv", index=False)
    return out


def write_label_schema(label_distribution):
    rows = [parse_label(label) for label in label_distribution["exact_label"].tolist()]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "cradle_dialogue_label_schema.csv", index=False)
    return out


def label_cardinality_stats(df):
    counts = df["labels"].map(lambda x: len(split_labels(x)))
    return {
        "unlabeled_turns": int((counts == 0).sum()),
        "single_label_turns": int((counts == 1).sum()),
        "multi_label_turns": int((counts > 1).sum()),
        "max_labels_on_one_turn": int(counts.max()) if len(counts) else 0,
        "label_count_distribution": Counter(counts.tolist()),
    }


def dialogue_label_events(df, schema_map):
    events = []
    for _, row in df.iterrows():
        for label in split_labels(row["labels"]):
            parsed = schema_map.get(label, parse_label(label))
            events.append(
                {
                    "dialogue_id": row["dialogue_id"],
                    "turn_id": int(row["turn_id"]),
                    "exact_label": label,
                    "event_type": parsed["event_type"],
                    "crisis_category": parsed["crisis_category"],
                    "temporal_state": parsed["temporal_state"],
                }
            )
    return pd.DataFrame(events)


def alert_confirm_statistics(all_data, schema_df):
    schema_map = {row["exact_label"]: row for _, row in schema_df.iterrows()}
    dialogue_rows = []
    summary_rows = []

    for split, df in all_data.items():
        events = dialogue_label_events(df, schema_map)
        split_distances = []
        unlabeled_before_alert = []
        unlabeled_before_confirm = []
        counts = Counter()

        for dialogue_id, group in df.groupby("dialogue_id", sort=False):
            ev = events[events["dialogue_id"] == dialogue_id].copy()
            if ev.empty:
                continue
            ev = ev.sort_values("turn_id")
            alert_turns = ev[ev["event_type"] == "alert"]["turn_id"].tolist()
            confirm_turns = ev[ev["event_type"] == "confirm"]["turn_id"].tolist()
            categories = sorted([c for c in ev["crisis_category"].unique().tolist() if c])
            temporal_states = sorted([t for t in ev["temporal_state"].unique().tolist() if t])
            first_alert = min(alert_turns) if alert_turns else None
            first_confirm = min(confirm_turns) if confirm_turns else None
            first_labeled = int(ev["turn_id"].min())
            final_labeled = int(ev["turn_id"].max())

            if first_alert is not None and first_confirm is None:
                counts["alert_only"] += 1
            if first_alert is None and first_confirm is not None:
                counts["confirm_only"] += 1
            if first_alert is not None and first_confirm is not None:
                if first_alert < first_confirm:
                    counts["alert_followed_by_confirm"] += 1
                    split_distances.append(first_confirm - first_alert)
                elif first_confirm < first_alert:
                    counts["confirm_before_alert"] += 1
                else:
                    counts["alert_and_confirm_same_turn"] += 1
            if len(categories) > 1:
                counts["multiple_crisis_categories"] += 1
            if "ongoing" in temporal_states and "past" in temporal_states:
                counts["both_ongoing_and_past"] += 1

            if first_alert is not None:
                unlabeled_before_alert.append(
                    int((group[group["turn_id"] < first_alert]["labels"].map(lambda x: len(split_labels(x))) == 0).sum())
                )
            if first_confirm is not None:
                unlabeled_before_confirm.append(
                    int((group[group["turn_id"] < first_confirm]["labels"].map(lambda x: len(split_labels(x))) == 0).sum())
                )

            for category in categories:
                cat_confirms = ev[(ev["event_type"] == "confirm") & (ev["crisis_category"] == category)]
                first_category_confirm = int(cat_confirms["turn_id"].min()) if not cat_confirms.empty else None
                dialogue_rows.append(
                    {
                        "row_type": "dialogue_category",
                        "split": split,
                        "dialogue_id": dialogue_id,
                        "category": category,
                        "first_labeled_turn": first_labeled,
                        "first_alert_turn": first_alert,
                        "first_confirm_turn": first_confirm,
                        "first_confirm_turn_for_category": first_category_confirm,
                        "final_labeled_turn": final_labeled,
                    }
                )

        def stat(values, fn):
            return float(fn(values)) if values else np.nan

        summary = {
            "row_type": "summary",
            "split": split,
            "dialogue_id": "",
            "category": "",
            "alert_only_dialogues": counts["alert_only"],
            "confirm_only_dialogues": counts["confirm_only"],
            "alert_followed_by_confirm_dialogues": counts["alert_followed_by_confirm"],
            "confirm_before_alert_dialogues": counts["confirm_before_alert"],
            "alert_and_confirm_same_turn_dialogues": counts["alert_and_confirm_same_turn"],
            "multiple_crisis_category_dialogues": counts["multiple_crisis_categories"],
            "both_ongoing_and_past_dialogues": counts["both_ongoing_and_past"],
            "alert_to_confirm_mean": stat(split_distances, np.mean),
            "alert_to_confirm_median": stat(split_distances, np.median),
            "alert_to_confirm_min": stat(split_distances, np.min),
            "alert_to_confirm_max": stat(split_distances, np.max),
            "unlabeled_before_first_alert_mean": stat(unlabeled_before_alert, np.mean),
            "unlabeled_before_first_confirm_mean": stat(unlabeled_before_confirm, np.mean),
            "alert_to_confirm_distance_distribution": json.dumps(dict(Counter(split_distances)), sort_keys=True),
        }
        summary_rows.append(summary)

    out = pd.DataFrame(summary_rows + dialogue_rows)
    out.to_csv(RESULTS_DIR / "cradle_dialogue_alert_confirm_statistics.csv", index=False)
    return out


def category_distribution_and_cooccurrence(all_data, schema_df):
    schema_map = {row["exact_label"]: row for _, row in schema_df.iterrows()}
    category_rows = []
    co_turn = Counter()
    co_dialogue = Counter()
    all_categories = sorted([c for c in schema_df["crisis_category"].dropna().unique().tolist() if c])

    for split, df in all_data.items():
        events = dialogue_label_events(df, schema_map)
        confirm_events = events[events["event_type"] == "confirm"]
        for category in all_categories:
            cat = confirm_events[confirm_events["crisis_category"] == category]
            category_rows.append(
                {
                    "row_type": "category_distribution",
                    "split": split,
                    "category_a": category,
                    "category_b": "",
                    "total_confirm_turns": len(cat),
                    "dialogues": cat["dialogue_id"].nunique(),
                    "ongoing_count": int((cat["temporal_state"] == "ongoing").sum()),
                    "past_count": int((cat["temporal_state"] == "past").sum()),
                    "same_turn_count": "",
                    "same_dialogue_count": "",
                }
            )

        for _, row in df.iterrows():
            cats = sorted(
                {
                    schema_map[label]["crisis_category"]
                    for label in split_labels(row["labels"])
                    if schema_map.get(label, {}).get("event_type") == "confirm"
                    and schema_map.get(label, {}).get("crisis_category")
                }
            )
            for a, b in combinations(cats, 2):
                co_turn[(split, a, b)] += 1

        for dialogue_id, group in df.groupby("dialogue_id"):
            cats = sorted(
                {
                    schema_map[label]["crisis_category"]
                    for labels in group["labels"]
                    for label in split_labels(labels)
                    if schema_map.get(label, {}).get("event_type") == "confirm"
                    and schema_map.get(label, {}).get("crisis_category")
                }
            )
            for a, b in combinations(cats, 2):
                co_dialogue[(split, a, b)] += 1

    co_rows = []
    for split in SPLITS:
        for a, b in combinations(all_categories, 2):
            co_rows.append(
                {
                    "row_type": "category_cooccurrence",
                    "split": split,
                    "category_a": a,
                    "category_b": b,
                    "total_confirm_turns": "",
                    "dialogues": "",
                    "ongoing_count": "",
                    "past_count": "",
                    "same_turn_count": co_turn[(split, a, b)],
                    "same_dialogue_count": co_dialogue[(split, a, b)],
                }
            )

    out = pd.DataFrame(category_rows + co_rows)
    out.to_csv(RESULTS_DIR / "cradle_dialogue_category_cooccurrence.csv", index=False)
    return out


def leakage_audit(all_data):
    lines = [
        "CRADLE-Dialogue Leakage / Overlap Audit",
        "=======================================",
        "",
        "Important caveat:",
        "The released Hugging Face table does not include the documented dialogue_id column.",
        "Dialogue IDs in this audit are inferred from turn_id resets within each split and are not official IDs.",
        "",
    ]

    split_texts = {}
    split_norm_texts = {}
    split_dialogue_hashes = {}
    split_dialogue_norm_hashes = {}
    split_dialogue_ids = {}

    for split, df in all_data.items():
        split_texts[split] = set(df["text"].astype(str))
        split_norm_texts[split] = set(df["text"].map(normalize_text))
        split_dialogue_ids[split] = set(df["dialogue_id"])
        raw_hashes = set()
        norm_hashes = set()
        for _, group in df.groupby("dialogue_id", sort=False):
            raw = "\n".join(group.sort_values("turn_id")["text"].astype(str).tolist())
            norm = "\n".join(group.sort_values("turn_id")["text"].map(normalize_text).tolist())
            raw_hashes.add(stable_hash(raw))
            norm_hashes.add(stable_hash(norm))
        split_dialogue_hashes[split] = raw_hashes
        split_dialogue_norm_hashes[split] = norm_hashes

    for a, b in combinations(SPLITS, 2):
        lines.extend(
            [
                f"{a} vs {b}",
                "-" * (len(a) + len(b) + 4),
                f"Exact text overlap: {len(split_texts[a] & split_texts[b])}",
                f"Exact normalized-text overlap: {len(split_norm_texts[a] & split_norm_texts[b])}",
                f"Entire-dialogue text-hash overlap: {len(split_dialogue_hashes[a] & split_dialogue_hashes[b])}",
                f"Normalized dialogue-hash overlap: {len(split_dialogue_norm_hashes[a] & split_dialogue_norm_hashes[b])}",
                f"Inferred dialogue ID overlap: {len(split_dialogue_ids[a] & split_dialogue_ids[b])}",
                "",
            ]
        )

    lines.extend(
        [
            "Source Scenario Overlap",
            "-----------------------",
            "The official dataset card says train/validation are synthetic/structured, while test is clinician-annotated.",
            "It also says dialogues are generated from real Reddit posts sourced from CRADLE-Bench via GPT-5, conditioned on clinical scenarios and reveal timing.",
            "The released fields do not include source post IDs, scenario IDs, or generation prompt IDs.",
            "Therefore exact source-scenario leakage cannot be verified from released fields alone.",
            "Do not claim zero source leakage solely because exact text hashes differ.",
            "",
            "CRADLE Bench Overlap Consideration",
            "----------------------------------",
            "The official card says CRADLE-Dialogue dialogues are generated from real Reddit posts sourced from CRADLE-Bench.",
            "Using both CRADLE-Dialogue and CRADLE-Bench could create conceptual or source overlap.",
            "Instance-level overlap would require source post IDs, original CRADLE-Bench IDs, or a formal overlap audit after downloading CRADLE-Bench.",
            "Recommendation: do not mix CRADLE-Bench into training/evaluation yet. If it is needed, download it only for a formal overlap audit or use it as a separate static benchmark.",
        ]
    )
    path = RESULTS_DIR / "cradle_dialogue_leakage_audit.txt"
    path.write_text("\n".join(lines) + "\n")
    return "\n".join(lines) + "\n"


def split_summary(split, df, features):
    tstats = turn_stats(df)
    turns_per_dialogue = tstats["turn_count"]
    speaker = speaker_audit(df)
    missing = df.isna().sum().to_dict()
    duplicate_pairs = int(df.duplicated(subset=["dialogue_id", "turn_id"]).sum())
    card = label_cardinality_stats(df)
    return {
        "split": split,
        "rows": len(df),
        "unique_dialogues": df["dialogue_id"].nunique(),
        "turns_min": int(turns_per_dialogue.min()),
        "turns_max": int(turns_per_dialogue.max()),
        "turns_mean": float(turns_per_dialogue.mean()),
        "turns_median": float(turns_per_dialogue.median()),
        "columns": list(df.columns),
        "features": features,
        "missing": missing,
        "duplicate_rows": row_duplicate_count(df),
        "duplicate_dialogue_turn_pairs": duplicate_pairs,
        "turn_all_begin_0": bool(tstats["begins_at_0"].all()),
        "turn_all_contiguous": bool(tstats["contiguous"].all()),
        "turn_dialogues_with_gaps": int(tstats["has_gaps"].sum()),
        "turn_dialogues_with_duplicate_indices": int(tstats["duplicate_turn_indices"].sum()),
        "speaker": speaker,
        "label_cardinality": card,
    }


def fmt_counter(counter):
    return ", ".join(f"{k}: {v}" for k, v in counter.items()) if counter else "none"


def write_main_audit(summaries, label_distribution, schema_df, category_df, leakage_text):
    lines = [
        "CRADLE-Dialogue Dataset Audit",
        "=============================",
        "",
        "Dataset: SungJoo/Cradle-Dialogue",
        "Official page: https://huggingface.co/datasets/SungJoo/Cradle-Dialogue",
        "",
        "Critical split distinction:",
        "- train: synthetic/structured conversational training data",
        "- validation: synthetic/structured conversational data",
        "- test: clinician-annotated benchmark",
        "",
        "Important released-field caveat:",
        "The official dataset card documents a dialogue_id field, but the loaded Hugging Face splits contain only turn_id, text, and labels.",
        "This audit reconstructs inferred dialogue IDs from turn_id resets. These inferred IDs are useful for structure checks but are not official identifiers.",
        "",
    ]

    for s in summaries:
        lines.extend(
            [
                f"Split: {s['split']}",
                "-" * (7 + len(s["split"])),
                f"Rows/turns: {s['rows']}",
                f"Unique inferred dialogues: {s['unique_dialogues']}",
                f"Turns per dialogue: min={s['turns_min']}, max={s['turns_max']}, mean={s['turns_mean']:.2f}, median={s['turns_median']:.2f}",
                f"Column names: {', '.join(s['columns'])}",
                f"Feature schema: {json.dumps({k: repr(v) for k, v in s['features'].items()}, ensure_ascii=False)}",
                f"Missing values: {s['missing']}",
                f"Duplicate rows: {s['duplicate_rows']}",
                f"Duplicate dialogue_id + turn_id pairs: {s['duplicate_dialogue_turn_pairs']}",
                f"Turn IDs begin at 0 in all inferred dialogues: {s['turn_all_begin_0']}",
                f"Turn IDs contiguous in all inferred dialogues: {s['turn_all_contiguous']}",
                f"Dialogues with turn gaps: {s['turn_dialogues_with_gaps']}",
                f"Dialogues with duplicate turn indices: {s['turn_dialogues_with_duplicate_indices']}",
                "",
                "Speaker audit:",
                f"- detected prefixes: {fmt_counter(s['speaker']['prefix_counts'])}",
                f"- role counts: {fmt_counter(s['speaker']['role_counts'])}",
                f"- unknown/unrecognized speaker prefixes: {s['speaker']['unknown_count']}",
                f"- dialogues with consistent alternation: {s['speaker']['alternation_ok_dialogues']} / {s['unique_dialogues']}",
                f"- dialogues with unusual ordering: {len(s['speaker']['alternation_problem_dialogues'])}",
                f"- first-turn roles: {fmt_counter(s['speaker']['first_roles'])}",
                f"- final-turn roles: {fmt_counter(s['speaker']['final_roles'])}",
                "",
                "Label cardinality:",
                f"- unlabeled turns: {s['label_cardinality']['unlabeled_turns']}",
                f"- single-label turns: {s['label_cardinality']['single_label_turns']}",
                f"- multi-label turns: {s['label_cardinality']['multi_label_turns']}",
                f"- max labels on one turn: {s['label_cardinality']['max_labels_on_one_turn']}",
                f"- number-of-labels distribution: {fmt_counter(s['label_cardinality']['label_count_distribution'])}",
                "",
            ]
        )

    lines.extend(
        [
            "Exact Released Label Vocabulary",
            "------------------------------",
        ]
    )
    for label in label_distribution["exact_label"].tolist():
        lines.append(f"- {label}")

    unparsed = schema_df[schema_df["parse_status"] != "parsed"]["exact_label"].tolist()
    categories = sorted([c for c in schema_df["crisis_category"].dropna().unique().tolist() if c])
    lines.extend(
        [
            "",
            "Parsed Label Structure",
            "----------------------",
            f"Crisis categories parsed from exact labels: {', '.join(categories)}",
            f"Unparsed labels: {', '.join(unparsed) if unparsed else 'none'}",
            "",
            "Category Head Recommendation",
            "----------------------------",
            "CRADLE labels are multi-label at the turn level and can contain multiple confirm categories on one turn/dialogue.",
            "The scientifically appropriate category formulation is multi-label classification, not forced single-label multiclass.",
            "Alert labels are type-agnostic and should not be forced into a crisis category.",
            "",
            "Speaker Structure Recommendation",
            "--------------------------------",
            "Use both structures later: preserve the speaker prefix in text for compatibility with language encoders, and also create a separate speaker_role column for hierarchical temporal modeling.",
            "Do not strip prefixes before an ablation confirms whether removing them helps.",
            "",
            "Dataset-To-Module Mapping",
            "-------------------------",
            "1. Severity: CRADLE does not directly provide Low/Medium/High severity labels. Do not invent a severity mapping without a separate design decision.",
            "2. Category: Confirm labels can supervise crisis category, and should be treated as multi-label.",
            "3. Progression: Alert and Confirm turns directly support temporal progression modeling and detection-delay evaluation.",
            "4. Uncertainty: uncertainty must come from the model/ensemble/calibration, not from CRADLE labels.",
            "5. Escalation: CRADLE can evaluate crisis detection and timing, but mapping Alert/Confirm to policy actions requires a justified policy design.",
            "6. Safety-centered evaluation: detection delay, early Alert detection, Confirm detection, missed escalation, unsafe continuation, premature escalation, and over-escalation burden can be built from labels plus a policy interpretation.",
            "",
            "Directly grounded metrics:",
            "- first Alert detection",
            "- first Confirm detection",
            "- Alert-to-Confirm delay",
            "- missed Confirm event",
            "- detection delay relative to clinician labels",
            "",
            "Metrics requiring policy interpretation:",
            "- unsafe continuation",
            "- premature escalation",
            "- over-escalation burden",
            "- mapping continue/resources/human support/crisis protocol to Alert/Confirm states",
            "",
            "AnnoMI Comparison",
            "-----------------",
            "AnnoMI remains useful as an out-of-domain conversational robustness and demonstration dataset, but its evaluation is pseudo-label based in the current pipeline.",
            "CRADLE-Dialogue is stronger for temporal crisis evaluation because it has turn-level crisis progression labels and a clinician-annotated test benchmark.",
            "Do not delete AnnoMI; keep it as a secondary/generalization dataset.",
            "",
            "CRADLE Bench Overlap",
            "--------------------",
            "CRADLE-Dialogue is related to CRADLE-Bench according to the official card: dialogues are generated from Reddit posts sourced from CRADLE-Bench.",
            "CRADLE-Bench should be excluded for now unless a formal overlap audit is performed.",
            "",
            "Output Files",
            "------------",
            "- cradle_dialogue/results/cradle_dialogue_dataset_audit.txt",
            "- cradle_dialogue/results/cradle_dialogue_label_distribution.csv",
            "- cradle_dialogue/results/cradle_dialogue_label_schema.csv",
            "- cradle_dialogue/results/cradle_dialogue_alert_confirm_statistics.csv",
            "- cradle_dialogue/results/cradle_dialogue_category_cooccurrence.csv",
            "- cradle_dialogue/results/cradle_dialogue_leakage_audit.txt",
        ]
    )
    path = RESULTS_DIR / "cradle_dialogue_dataset_audit.txt"
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    readme_path = hf_hub_download(args.dataset_name, "README.md", repo_type="dataset")
    ds = load_dataset(args.dataset_name)
    all_data = {}
    summaries = []
    for split in SPLITS:
        raw = ds[split]
        df = raw.to_pandas()
        df = infer_dialogue_ids(df, split)
        all_data[split] = df
        if args.save_split_csv:
            df.to_csv(DATA_DIR / f"{split}_with_inferred_dialogue_id.csv", index=False)
        summaries.append(split_summary(split, df, raw.features))

    label_distribution = write_label_distribution(all_data)
    schema_df = write_label_schema(label_distribution)
    alert_confirm_statistics(all_data, schema_df)
    category_distribution_and_cooccurrence(all_data, schema_df)
    leakage_text = leakage_audit(all_data)
    write_main_audit(summaries, label_distribution, schema_df, None, leakage_text)

    print(f"Loaded official dataset: {args.dataset_name}")
    print(f"Read official README: {readme_path}")
    print("Saved audit outputs under cradle_dialogue/results/")
    print("Important: dialogue_id was not present in loaded fields; inferred IDs were created from turn_id resets.")


if __name__ == "__main__":
    main()
