import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DEFAULT_OUTPUT = Path("advanced_module1/data/multitask_dataset_audit.csv")
DEFAULT_REPORT = Path("advanced_module1/data/multitask_dataset_audit_report.txt")

SEVERITY_LABELS = {
    0: "low",
    1: "medium",
    2: "high",
}

CATEGORY_TO_ID = {
    "depressive_general_distress": 0,
    "fear_anxiety": 1,
}

EMOTION_LABEL_TO_NAME = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

EMOTION_TO_SEVERITY = {
    0: 1,  # sadness -> medium
    1: 0,  # joy -> low
    2: 0,  # love -> low
    3: 0,  # anger -> low in the existing baseline mapping
    4: 1,  # fear -> medium
    5: 0,  # surprise -> low
}

EMOTION_TO_CATEGORY = {
    0: "depressive_general_distress",
    4: "fear_anxiety",
}

SUICIDE_SELF_HARM_PATTERN = re.compile(
    r"\b("
    r"suicide|suicidal|kill myself|killing myself|end my life|"
    r"take my life|self harm|self-harm|hurt myself|harm myself|"
    r"cut myself|overdose"
    r")\b",
    flags=re.IGNORECASE,
)

ABUSE_PATTERN = re.compile(
    r"\b("
    r"abuse|abused|abusive|assault|assaulted|violence|violent|"
    r"hit me|beat me|molested|rape|raped"
    r")\b",
    flags=re.IGNORECASE,
)

PANIC_PATTERN = re.compile(
    r"\b(panic attack|panic|anxiety attack|terrified|fearful)\b",
    flags=re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an auditable multi-task dataset without inventing category labels. "
            "Trainable category labels are assigned only when supported by explicit source labels."
        )
    )
    parser.add_argument("--output_csv", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report_txt", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def clean_text(text):
    text = str(text)
    text = re.sub(r"^Post:\s*", "", text)
    return text.strip()


def weak_category_candidate(text):
    matches = []
    if SUICIDE_SELF_HARM_PATTERN.search(text):
        matches.append(("suicidality_self_harm", "keyword_audit_only"))
    if ABUSE_PATTERN.search(text):
        matches.append(("abuse_disclosure", "keyword_audit_only"))
    if PANIC_PATTERN.search(text):
        matches.append(("panic_or_anxiety", "keyword_audit_only"))

    if not matches:
        return "missing", "none"
    labels = ";".join(label for label, _ in matches)
    reasons = ";".join(reason for _, reason in matches)
    return labels, reasons


def base_row(
    text,
    risk_severity,
    severity_label_quality,
    source_dataset,
    source_split,
    source_row_id,
    source_label,
    source_label_name,
):
    text = clean_text(text)
    weak_label, weak_reason = weak_category_candidate(text)
    return {
        "text": text,
        "risk_severity": int(risk_severity),
        "risk_severity_name": SEVERITY_LABELS[int(risk_severity)],
        "has_severity_label": 1,
        "severity_label_quality": severity_label_quality,
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_row_id": int(source_row_id),
        "source_label": source_label,
        "source_label_name": source_label_name,
        "category_label": "missing",
        "category_id": -1,
        "has_category_label": 0,
        "category_label_quality": "missing",
        "category_label_basis": "not_enough_source_information",
        "weak_category_candidate": weak_label,
        "weak_category_reason": weak_reason,
        "urgency_score": pd.NA,
        "has_urgency_label": 0,
        "urgency_label_quality": "missing",
        "weak_urgency_candidate": float({0: 0.1, 1: 0.5, 2: 0.9}[int(risk_severity)]),
        "weak_urgency_reason": "derived_from_severity_audit_only",
    }


def load_dreaddit_rows():
    dataset = load_dataset("asmaab/dreaddit_train")["train"]
    rows = []
    for idx, example in enumerate(dataset):
        severity = {"no": 0, "yes": 1}[example["label"]]
        rows.append(
            base_row(
                text=example["post"],
                risk_severity=severity,
                severity_label_quality="source_derived_stress_binary",
                source_dataset="asmaab/dreaddit_train",
                source_split="train",
                source_row_id=idx,
                source_label=example["label"],
                source_label_name=example["label"],
            )
        )
    return rows


def load_emotion_rows():
    dataset = load_dataset("dair-ai/emotion")["train"]
    rows = []
    for idx, example in enumerate(dataset):
        source_label = int(example["label"])
        source_label_name = EMOTION_LABEL_TO_NAME[source_label]
        row = base_row(
            text=example["text"],
            risk_severity=EMOTION_TO_SEVERITY[source_label],
            severity_label_quality="source_derived_emotion_mapping",
            source_dataset="dair-ai/emotion",
            source_split="train",
            source_row_id=idx,
            source_label=source_label,
            source_label_name=source_label_name,
        )

        category = EMOTION_TO_CATEGORY.get(source_label)
        if category is not None:
            row["category_label"] = category
            row["category_id"] = CATEGORY_TO_ID[category]
            row["has_category_label"] = 1
            row["category_label_quality"] = "source_derived"
            row["category_label_basis"] = f"emotion_label_{source_label_name}"
        rows.append(row)
    return rows


def load_counseling_rows():
    dataset = load_dataset("Amod/mental_health_counseling_conversations")["train"]
    rows = []
    for idx, example in enumerate(dataset):
        rows.append(
            base_row(
                text=example["Context"],
                risk_severity=2,
                severity_label_quality="dataset_origin_weak_high_risk_baseline",
                source_dataset="Amod/mental_health_counseling_conversations",
                source_split="train",
                source_row_id=idx,
                source_label="missing",
                source_label_name="missing",
            )
        )
    return rows


def build_dataset():
    rows = []
    rows.extend(load_dreaddit_rows())
    rows.extend(load_emotion_rows())
    rows.extend(load_counseling_rows())
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["text", "risk_severity"])
    df = df[df["text"].astype(str).str.len() >= 10].copy()
    df["risk_severity"] = df["risk_severity"].astype(int)
    df["has_severity_label"] = df["has_severity_label"].astype(int)
    df["has_category_label"] = df["has_category_label"].astype(int)
    df["category_id"] = df["category_id"].astype(int)
    df["has_urgency_label"] = df["has_urgency_label"].astype(int)
    return df


def format_counts(series):
    counts = series.value_counts(dropna=False).sort_index()
    return "\n".join(f"- {idx}: {value}" for idx, value in counts.items())


def make_report(df):
    total = len(df)
    category_count = int(df["has_category_label"].sum())
    missing_category_count = total - category_count
    urgency_count = int(df["has_urgency_label"].sum())
    weak_category_count = int((df["weak_category_candidate"] != "missing").sum())

    lines = [
        "Advanced Module 1 Multi-Task Dataset Audit",
        "==========================================",
        "",
        "Purpose",
        "-------",
        "This file audits recoverable supervision for a future multi-task model.",
        "It does not convert heuristic keyword matches into trainable category labels.",
        "",
        "Strict Category Rule",
        "--------------------",
        "A trainable category label is assigned only when an explicit original source label directly supports it.",
        "Dataset origin alone is not treated as a semantic risk category.",
        "",
        "Summary",
        "-------",
        f"Total examples: {total}",
        f"Examples with severity labels: {int(df['has_severity_label'].sum())}",
        f"Examples with genuine/source-derived category labels: {category_count}",
        f"Examples missing category labels: {missing_category_count}",
        f"Examples with true urgency labels: {urgency_count}",
        f"Examples with keyword audit-only category candidates: {weak_category_count}",
        "",
        "Source Dataset Counts",
        "---------------------",
        format_counts(df["source_dataset"]),
        "",
        "Severity Counts",
        "---------------",
        format_counts(df["risk_severity_name"]),
        "",
        "Category Label Counts",
        "---------------------",
        format_counts(df["category_label"]),
        "",
        "Category Label Quality Counts",
        "-----------------------------",
        format_counts(df["category_label_quality"]),
        "",
        "Weak Category Candidate Counts",
        "------------------------------",
        format_counts(df["weak_category_candidate"]),
        "",
        "Important Limitation",
        "--------------------",
        "Only dair-ai/emotion sadness and fear provide defensible source-derived category labels here.",
        "Dreaddit stress labels and counseling dataset origin do not provide enough information for semantic category supervision.",
        "Urgency has no true label in these sources; weak urgency candidates are audit-only until explicitly approved.",
        "",
        "Recommended Next Decision",
        "-------------------------",
        "Decide whether the available source-derived category supervision is sufficient, or add a category-annotated dataset before training.",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output_csv = Path(args.output_csv)
    report_txt = Path(args.report_txt)

    df = build_dataset()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_txt.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)
    report = make_report(df)
    report_txt.write_text(report)

    print(report)
    print(f"Saved dataset: {output_csv}")
    print(f"Saved report: {report_txt}")


if __name__ == "__main__":
    main()
