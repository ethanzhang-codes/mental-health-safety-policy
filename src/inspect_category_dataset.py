import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import ClassLabel, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from datasets.exceptions import DatasetGenerationCastError, DatasetNotFoundError
from huggingface_hub import HfApi


AUDIT_DIR = Path("advanced_module1/category_dataset_audits")

LABEL_COLUMN_HINTS = (
    "label",
    "labels",
    "class",
    "category",
    "categories",
    "disorder",
    "diagnosis",
    "diagnostic",
    "symptom",
    "risk",
    "severity",
    "crisis",
    "suicide",
    "self_harm",
    "self-harm",
    "abuse",
    "trauma",
    "mental",
    "intent",
    "target",
    "annotation",
    "annotated",
)

TEXT_COLUMN_HINTS = (
    "text",
    "post",
    "content",
    "utterance",
    "message",
    "comment",
    "context",
    "question",
    "sentence",
    "body",
    "input",
    "prompt",
)

CATEGORY_CONCEPT_HINTS = {
    "mental health category": ("category", "class", "label", "mental", "condition"),
    "disorder/category": ("disorder", "diagnosis", "condition", "category"),
    "symptom type": ("symptom", "symptoms"),
    "suicide/self-harm category": ("suicide", "suicidal", "self_harm", "self-harm", "selfharm"),
    "anxiety/panic": ("anxiety", "panic", "fear"),
    "depression/distress": ("depression", "depressive", "distress", "sadness"),
    "trauma/abuse": ("trauma", "abuse", "ptsd", "violence"),
    "crisis/risk type": ("crisis", "risk", "urgency", "emergency"),
}

SEVERITY_HINTS = ("severity", "risk", "level", "crisis", "label")
URGENCY_HINTS = ("urgency", "urgent", "immediate", "emergency", "priority")
TEMPORAL_HINTS = (
    "conversation",
    "conversation_id",
    "thread",
    "dialog",
    "dialogue",
    "turn",
    "turn_id",
    "timestamp",
    "time",
    "created",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit a Hugging Face dataset for genuine category supervision candidates. "
            "This script does not merge data or create heuristic training labels."
        )
    )
    parser.add_argument("--dataset_name", required=True, help="Hugging Face dataset name.")
    parser.add_argument(
        "--dataset_config",
        default=None,
        help="Optional Hugging Face dataset configuration/subset name.",
    )
    parser.add_argument(
        "--data_file",
        default=None,
        help=(
            "Optional single file inside the Hugging Face dataset repo to inspect, "
            "for dataset repos whose files have different columns."
        ),
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split to inspect. If omitted, the first available split is used.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=20000,
        help="Maximum rows to materialize for counts/examples from the selected split.",
    )
    parser.add_argument(
        "--examples_per_label",
        type=int,
        default=5,
        help="Number of text examples to show per label value where possible.",
    )
    return parser.parse_args()


def safe_filename(dataset_name, dataset_config, data_file, split):
    parts = [dataset_name]
    if dataset_config:
        parts.append(dataset_config)
    if data_file:
        parts.append(Path(data_file).name)
    if split:
        parts.append(split)
    name = "__".join(parts)
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return f"{name}_audit.txt"


def load_selected_dataset(dataset_name, dataset_config, data_file, split):
    placeholder_names = {"DATASET_NAME", "dataset_name", "some/dataset-name"}
    if dataset_name in placeholder_names:
        raise ValueError(
            "--dataset_name must be a real Hugging Face dataset ID, not the placeholder "
            f"'{dataset_name}'. Example: --dataset_name dair-ai/emotion"
        )

    load_kwargs = {}
    if data_file:
        load_kwargs["data_files"] = data_file

    if dataset_config:
        loaded = load_dataset(dataset_name, dataset_config, **load_kwargs)
    else:
        loaded = load_dataset(dataset_name, **load_kwargs)

    if isinstance(loaded, (IterableDataset,)):
        selected_split = split or "streaming_iterable"
        return loaded, selected_split, loaded

    if isinstance(loaded, (DatasetDict, IterableDatasetDict)):
        available_splits = list(loaded.keys())
        selected_split = split or available_splits[0]
        if selected_split not in loaded:
            raise ValueError(
                f"Split '{selected_split}' was not found. Available splits: {available_splits}"
            )
        return loaded, selected_split, loaded[selected_split]

    selected_split = split or "train"
    return loaded, selected_split, loaded


def materialize_dataframe(dataset, max_rows):
    if isinstance(dataset, IterableDataset):
        rows = []
        for idx, row in enumerate(dataset):
            if idx >= max_rows:
                break
            rows.append(row)
        return pd.DataFrame(rows), True

    count = min(len(dataset), max_rows)
    return pd.DataFrame(dataset.select(range(count))), count < len(dataset)


def feature_to_text(feature):
    try:
        return repr(feature)
    except Exception:
        return str(type(feature))


def class_names_from_feature(feature):
    if isinstance(feature, ClassLabel):
        return list(feature.names)
    if hasattr(feature, "feature") and isinstance(feature.feature, ClassLabel):
        return list(feature.feature.names)
    return None


def is_label_like_column(column, series, feature):
    lower = column.lower()
    has_name_hint = any(hint in lower for hint in LABEL_COLUMN_HINTS)
    if any(id_hint in lower for id_hint in ("id", "index")) and not has_name_hint:
        return False
    if pd.api.types.is_numeric_dtype(series) and not has_name_hint and class_names_from_feature(feature) is None:
        return False
    if any(hint in lower for hint in LABEL_COLUMN_HINTS):
        return True
    if class_names_from_feature(feature) is not None:
        return True
    nunique = series.dropna().nunique()
    if nunique <= 50 and not looks_like_text(series):
        return True
    return False


def looks_like_text(series):
    non_null = series.dropna()
    if non_null.empty:
        return False
    sample = non_null.astype(str).head(100)
    avg_len = sample.str.len().mean()
    long_fraction = (sample.str.len() > 40).mean()
    return avg_len > 30 or long_fraction > 0.25


def find_text_column(df):
    for column in df.columns:
        lower = column.lower()
        if any(hint == lower or hint in lower for hint in TEXT_COLUMN_HINTS):
            if looks_like_text(df[column]):
                return column
    text_like = [column for column in df.columns if looks_like_text(df[column])]
    return text_like[0] if text_like else None


def value_to_label(value, class_names):
    if pd.isna(value):
        return "missing"
    if class_names is not None:
        try:
            idx = int(value)
            if 0 <= idx < len(class_names):
                return class_names[idx]
        except Exception:
            pass
    return str(value)


def counts_for_column(series, class_names):
    labels = [value_to_label(value, class_names) for value in series]
    return Counter(labels)


def sample_examples(df, label_column, text_column, class_names, examples_per_label):
    examples = defaultdict(list)
    if text_column is None:
        return examples

    for _, row in df[[label_column, text_column]].dropna(subset=[text_column]).iterrows():
        label = value_to_label(row[label_column], class_names)
        if len(examples[label]) >= examples_per_label:
            continue
        text = str(row[text_column]).replace("\n", " ").strip()
        if len(text) > 500:
            text = text[:497] + "..."
        examples[label].append(text)
    return examples


def concept_matches(column, labels):
    searchable = " ".join([column] + [str(label) for label in labels]).lower()
    matches = []
    for concept, hints in CATEGORY_CONCEPT_HINTS.items():
        if any(hint in searchable for hint in hints):
            matches.append(concept)
    return matches


def annotation_source_assessment(column, feature, labels):
    lower = column.lower()
    if isinstance(feature, ClassLabel):
        return "explicitly annotated by dataset schema"
    if any(hint in lower for hint in ("label", "category", "class", "diagnosis", "disorder", "symptom")):
        return "possibly explicitly annotated; manual dataset-card review required"
    if labels and len(labels) > 1:
        return "unknown from available schema; manual dataset-card review required"
    return "unknown from available schema"


def suitability_notes(df, label_reports):
    all_columns = " ".join(df.columns).lower()
    label_text = " ".join(
        [report["column"].lower() + " " + " ".join(map(str, report["labels"])).lower() for report in label_reports]
    )
    combined = all_columns + " " + label_text

    has_category_candidate = any(report["category_concept_candidates"] for report in label_reports)
    has_severity_candidate = any(hint in combined for hint in SEVERITY_HINTS)
    has_urgency_candidate = any(hint in combined for hint in URGENCY_HINTS)
    has_temporal_candidate = any(hint in all_columns for hint in TEMPORAL_HINTS)

    return {
        "severity supervision": (
            "possible candidate columns found; manual review required"
            if has_severity_candidate
            else "not apparent from schema"
        ),
        "category supervision": (
            "possible semantic category candidates found; use only if labels are explicit/source-provided"
            if has_category_candidate
            else "not apparent from schema"
        ),
        "urgency supervision": (
            "possible urgency/risk-priority candidates found; manual review required"
            if has_urgency_candidate
            else "not apparent from schema"
        ),
        "temporal progression supervision": (
            "possible temporal/conversation columns found"
            if has_temporal_candidate
            else "not apparent from schema"
        ),
    }


def inspect_dataset(args):
    loaded, selected_split, dataset = load_selected_dataset(
        args.dataset_name,
        args.dataset_config,
        args.data_file,
        args.split,
    )
    df, truncated = materialize_dataframe(dataset, args.max_rows)
    features = getattr(dataset, "features", {})
    text_column = find_text_column(df)

    label_reports = []
    for column in df.columns:
        feature = features.get(column) if hasattr(features, "get") else None
        series = df[column]
        if not is_label_like_column(column, series, feature):
            continue
        class_names = class_names_from_feature(feature)
        counts = counts_for_column(series, class_names)
        labels = list(counts.keys())
        report = {
            "column": column,
            "dtype": str(series.dtype),
            "feature": feature_to_text(feature),
            "class_names": class_names,
            "labels": labels,
            "counts": counts,
            "missing_count": int(series.isna().sum()),
            "category_concept_candidates": concept_matches(column, labels),
            "annotation_source_assessment": annotation_source_assessment(column, feature, labels),
            "examples": sample_examples(
                df,
                column,
                text_column,
                class_names,
                args.examples_per_label,
            ),
        }
        label_reports.append(report)

    return {
        "loaded": loaded,
        "selected_split": selected_split,
        "dataset": dataset,
        "df": df,
        "truncated": truncated,
        "features": features,
        "text_column": text_column,
        "label_reports": label_reports,
        "suitability": suitability_notes(df, label_reports),
    }


def format_count_items(counter, limit=100):
    items = counter.most_common(limit)
    lines = [f"  - {label}: {count}" for label, count in items]
    if len(counter) > limit:
        lines.append(f"  - ... {len(counter) - limit} more labels omitted")
    return "\n".join(lines) if lines else "  none"


def format_schema(features):
    try:
        parsed = {key: feature_to_text(value) for key, value in features.items()}
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        return feature_to_text(features)


def make_report(args, audit):
    df = audit["df"]
    lines = [
        "Category Dataset Integration Audit",
        "==================================",
        "",
        "Scientific rule: this report identifies possible category-supervision candidates for manual review.",
        "It does not create labels, merge data, or treat dataset origin/keyword matches as ground truth.",
        "",
        "Dataset Overview",
        "----------------",
        f"Dataset name: {args.dataset_name}",
        f"Dataset configuration: {args.dataset_config or 'not_provided'}",
        f"Data file: {args.data_file or 'not_provided'}",
        f"Split used: {audit['selected_split']}",
        f"Rows inspected: {len(df)}",
        f"Rows truncated by --max_rows: {'yes' if audit['truncated'] else 'no'}",
        f"Column names: {', '.join(map(str, df.columns)) if len(df.columns) else 'none'}",
        f"Detected text column for examples: {audit['text_column'] or 'none'}",
        "",
        "Hugging Face Feature / Schema Information",
        "-----------------------------------------",
        format_schema(audit["features"]),
        "",
        "Possible Label-Like Columns",
        "---------------------------",
    ]

    if not audit["label_reports"]:
        lines.append("No obvious label-like columns were detected from schema/column names/cardinality.")
    else:
        for report in audit["label_reports"]:
            lines.extend(
                [
                    f"Column: {report['column']}",
                    f"- dtype: {report['dtype']}",
                    f"- feature: {report['feature']}",
                    f"- missing-value count: {report['missing_count']}",
                    f"- class names: {', '.join(report['class_names']) if report['class_names'] else 'not_available'}",
                    f"- possible category concepts: {', '.join(report['category_concept_candidates']) if report['category_concept_candidates'] else 'none_detected'}",
                    f"- label source assessment: {report['annotation_source_assessment']}",
                    "- label counts:",
                    format_count_items(report["counts"]),
                    "- example texts by label:",
                ]
            )
            if not report["examples"]:
                lines.append("  text examples unavailable")
            else:
                for label, examples in report["examples"].items():
                    lines.append(f"  Label: {label}")
                    for idx, text in enumerate(examples, start=1):
                        lines.append(f"    {idx}. {text}")
            lines.append("")

    lines.extend(
        [
            "Suitability Summary",
            "-------------------",
        ]
    )
    for key, value in audit["suitability"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "Manual Review Checklist",
            "-----------------------",
            "- Use a column for category-head training only if it is an explicit/source-provided semantic category label.",
            "- Do not treat dataset origin, subset name, or keywords as ground truth.",
            "- Confirm the dataset license, intended use, label definitions, and annotation process before merging.",
            "- Keep missing category labels as missing; they can still be useful for severity-only training.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    try:
        audit = inspect_dataset(args)
    except DatasetNotFoundError as exc:
        print("\nDataset could not be loaded.")
        print("----------------------------")
        print(f"Hugging Face error: {exc}")
        print("\nWhat to check:")
        print("- Make sure --dataset_name is a real Hugging Face dataset ID.")
        print("- If the dataset is private or gated, make sure you have access and are logged in.")
        print("- If the dataset requires a config/subset, add --dataset_config CONFIG_NAME.")
        print("\nExample:")
        print("python inspect_category_dataset.py --dataset_name dair-ai/emotion --split train")
        raise SystemExit(2)
    except DatasetGenerationCastError as exc:
        print("\nDataset files could not be combined automatically.")
        print("-----------------------------------------------")
        print("This usually means the Hugging Face dataset repo contains multiple CSV/Parquet files with different columns.")
        print("For a category audit, inspect one file at a time with --data_file.")
        print(f"\nOriginal error: {exc}")
        try:
            files = HfApi().list_repo_files(args.dataset_name, repo_type="dataset")
            data_files = [
                f for f in files
                if f.lower().endswith((".csv", ".json", ".jsonl", ".parquet", ".txt"))
            ]
            if data_files:
                print("\nAvailable data files in this dataset repo:")
                for filename in data_files:
                    print(f"- {filename}")
        except Exception:
            pass
        print("\nExample:")
        print(
            "python inspect_category_dataset.py "
            f"--dataset_name {args.dataset_name} "
            "--data_file FILE_NAME "
            "--split train"
        )
        raise SystemExit(2)
    except ValueError as exc:
        print("\nInvalid audit command.")
        print("----------------------")
        print(str(exc))
        print("\nExample:")
        print("python inspect_category_dataset.py --dataset_name dair-ai/emotion --split train")
        raise SystemExit(2)

    report = make_report(args, audit)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AUDIT_DIR / safe_filename(
        args.dataset_name,
        args.dataset_config,
        args.data_file,
        audit["selected_split"],
    )
    output_path.write_text(report)

    print(report)
    print(f"Saved audit report: {output_path}")


if __name__ == "__main__":
    main()
