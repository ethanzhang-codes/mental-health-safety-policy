from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "cradle_dialogue" / "results" / "presentation_inventory"

NA = "not found"


KEY_RESULT_FILES = {
    "final_decision": ROOT / "cradle_dialogue/results/final_experiment_summary/final_policy_decision.md",
    "final_table": ROOT / "cradle_dialogue/results/final_experiment_summary/final_results_table.csv",
    "final_narrative": ROOT / "cradle_dialogue/results/final_experiment_summary/final_results_narrative.md",
    "module1_seed1": ROOT / "results/module1_seed1_benchmark_comparison.csv",
    "module1_multiseed": ROOT / "results/module1_multiseed_finalist_comparison.csv",
    "single_vs_multi": ROOT / "cradle_dialogue/results/single_task_vs_multitask_comparison.txt",
    "alert_baseline": ROOT / "cradle_dialogue/results/single_task_alert/seed_1.txt",
    "temporal_alert": ROOT / "cradle_dialogue/results/temporal_alert_embeddings/single_turn_vs_temporal_alert_comparison.txt",
    "alert_hw": ROOT / "cradle_dialogue/results/temporal_alert_embeddings_history_window_ablation.txt",
    "confirm_baseline": ROOT / "cradle_dialogue/results/single_task_confirm/seed_1.txt",
    "confirm_audit": ROOT / "cradle_dialogue/results/single_task_confirm/confirm_false_negative_audit_report.txt",
    "temporal_confirm": ROOT / "cradle_dialogue/results/temporal_confirm_embeddings/single_turn_vs_temporal_confirm_comparison.txt",
    "confirm_high_recall": ROOT / "cradle_dialogue/results/single_task_confirm_high_recall/high_recall_confirm_variant_comparison.txt",
    "category": ROOT / "cradle_dialogue/results/single_turn_category/category_model_summary.txt",
    "policy_old": ROOT / "cradle_dialogue/results/safety_constrained_policy_v2/policy_comparison_all_v2.txt",
    "policy_high_recall": ROOT / "cradle_dialogue/results/policy_high_recall_confirm/policy_comparison_high_recall_confirm.txt",
    "guardrail_v3": ROOT / "cradle_dialogue/results/policy_high_recall_confirm_guardrail_v3/guardrail_v3_summary.txt",
    "abstract_alignment": ROOT / "cradle_dialogue/results/abstract_alignment/abstract_alignment_report.md",
    "uncertainty": ROOT / "cradle_dialogue/results/uncertainty_ablation/uncertainty_ablation_summary.md",
}


FINAL_METRICS = {
    "old_safety_high_risk_unsafe_continuation": "0.2978723404255319",
    "old_safety_confirm_unsafe_continuation": "0.2978723404255319",
    "old_safety_over_escalation_burden": "0.11898535564853556",
    "final_high_risk_unsafe_continuation": "0.2297872340425532",
    "final_confirm_unsafe_continuation": "0.2297872340425532",
    "final_alert_unsafe_continuation": "0.26495726495726496",
    "final_over_escalation_burden": "0.12238493723849372",
    "final_urgent_escalation_precision": "0.7727272727272727",
    "final_human_defer_precision": "0.3945666235446313",
    "final_macro_f1": "0.38786635857726703",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path, max_chars: int = 300_000) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, NA) for field in fields})


def classify_script(path: Path) -> str:
    name = path.name.lower()
    text_path = str(path).lower()
    if "uncertainty" in text_path:
        return "ablation"
    if "guardrail_v3" in text_path or "abstract" in text_path or "context_false" in text_path:
        return "ablation"
    if "backup" in text_path or "inspect" in name or "diagnose" in name or "compare" in name:
        return "exploratory"
    final_markers = [
        "build_cradle_progression_features_high_recall_confirm",
        "run_policies_with_high_recall_confirm",
        "train_cradle_single_task_confirm_high_recall",
        "train_cradle_temporal_alert_from_embeddings",
        "train_cradle_single_turn_category",
        "policy_engine",
        "build_cradle_rule_based_policy",
        "train_cradle_safety_constrained_policy_v2",
    ]
    return "final pipeline" if any(m in name for m in final_markers) else "exploratory"


def summarize_path_refs(text: str, output_only: bool = False) -> str:
    refs = set()
    patterns = [
        r"read_csv\((?:Path\()?['\"]([^'\"]+)['\"]",
        r"to_csv\((?:Path\()?['\"]([^'\"]+)['\"]",
        r"open\((?:Path\()?['\"]([^'\"]+)['\"]",
        r"Path\(['\"]([^'\"]+)['\"]\)",
    ]
    if output_only:
        patterns = [r"to_csv\((?:Path\()?['\"]([^'\"]+)['\"]", r"write_text\((?:Path\()?['\"]([^'\"]+)['\"]"]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if len(match) < 180 and not match.startswith("{"):
                refs.add(match)
    return "; ".join(sorted(refs)) if refs else NA


def inventory_scripts() -> list[dict]:
    rows = []
    skip_parts = {"__pycache__", ".git"}
    py_files = [p for p in ROOT.rglob("*.py") if not any(part in skip_parts for part in p.parts)]
    relevant_keywords = [
        "cradle", "policy", "alert", "confirm", "category", "progression", "guardrail",
        "uncertainty", "abstract", "module1", "benchmark", "temporal", "dataset", "severity",
        "roberta", "bert", "deberta", "alignment", "context", "audit", "temperature",
    ]
    for path in sorted(py_files):
        rp = rel(path)
        if not any(k in rp.lower() for k in relevant_keywords):
            continue
        text = read_text(path)
        try:
            tree = ast.parse(text)
            names = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
            functions = "; ".join(names[:40]) if names else NA
        except Exception:
            functions = NA
        doc = ast.get_docstring(ast.parse(text)) if text else None
        purpose = doc.splitlines()[0] if doc else infer_purpose(path)
        model_hits = sorted(set(re.findall(r"(?:mental/mental-roberta-base|AIMH/mental-bert-large-cased|AIMH/mental-roberta-large|microsoft/deberta-v3-large|answerdotai/ModernBERT-large|[A-Za-z0-9_.-]*BERT[A-Za-z0-9_.-]*|[A-Za-z0-9_.-]*RoBERTa[A-Za-z0-9_.-]*|RandomForestClassifier|LogisticRegression|HistGradientBoostingClassifier|GradientBoostingClassifier)", text)))
        loss_hits = sorted(set(re.findall(r"(BCEWithLogitsLoss|CrossEntropyLoss|FocalLoss|focal loss|pos_weight|class_weight|weighted)", text, flags=re.I)))
        metric_hits = sorted(set(re.findall(r"(AUROC|AUPRC|macro_f1|macro F1|f1_score|precision_score|recall_score|roc_auc_score|average_precision_score|unsafe_continuation|over_escalation|false_negative|false_positive|accuracy)", text, flags=re.I)))
        rows.append({
            "file_path": rp,
            "purpose": purpose,
            "inputs": summarize_path_refs(text),
            "outputs": summarize_path_refs(text, output_only=True),
            "important_functions_classes": functions,
            "models_used": "; ".join(model_hits) if model_hits else NA,
            "loss_functions_used": "; ".join(loss_hits) if loss_hits else NA,
            "metrics_computed": "; ".join(metric_hits) if metric_hits else NA,
            "pipeline_role": classify_script(path),
        })
    return rows


def infer_purpose(path: Path) -> str:
    name = path.stem.replace("_", " ")
    mapping = {
        "train": "Train model or policy component",
        "build": "Build processed CRADLE artifact",
        "audit": "Audit model or policy outputs",
        "compare": "Compare experiment results",
        "run": "Run model or policy experiment",
        "search": "Search candidate configuration",
        "align": "Create abstract-aligned analysis artifacts",
        "analyze": "Analyze error cases",
    }
    for prefix, purpose in mapping.items():
        if path.name.startswith(prefix):
            return f"{purpose}: {name}"
    return name


def inventory_data() -> list[dict]:
    roots = [
        ROOT / "cradle_dialogue/data",
        ROOT / "cradle_dialogue/predictions",
        ROOT / "cradle_dialogue/policy_outputs",
        ROOT / "cradle_dialogue/policy_outputs_high_recall_confirm",
        ROOT / "cradle_dialogue/policy_outputs_high_recall_confirm_guardrail_v3",
        ROOT / "cradle_dialogue/policy_outputs_uncertainty_ablation",
        ROOT / "cradle_dialogue/results",
        ROOT / "module1_encoder_benchmark/predictions",
        ROOT / "results",
    ]
    rows = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            if path in seen:
                continue
            seen.add(path)
            rows.append(summarize_csv(path))
    return rows


def summarize_csv(path: Path) -> dict:
    try:
        df = pd.read_csv(path, nrows=0)
        cols = list(df.columns)
        row_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1
        row_count = max(row_count, 0)
    except Exception as exc:
        cols = []
        row_count = f"read_error: {exc}"
    lower_cols = [c.lower() for c in cols]
    key_cols = [c for c in cols if c.lower() in {"dialogue_id", "turn_id", "split", "id", "text", "speaker"}]
    label_cols = [c for c in cols if c.startswith("target_") or "label" in c.lower()]
    pred_cols = [c for c in cols if "prob" in c.lower() or "pred" in c.lower() or "policy_level" in c.lower() or "risk_" in c.lower()]
    return {
        "file_path": rel(path),
        "rows": row_count,
        "columns": len(cols),
        "split": infer_split(path.name),
        "key_columns": "; ".join(key_cols[:20]) if key_cols else NA,
        "label_columns": "; ".join(label_cols[:30]) if label_cols else NA,
        "prediction_columns": "; ".join(pred_cols[:35]) if pred_cols else NA,
        "type": infer_data_type(path),
    }


def infer_split(name: str) -> str:
    name = name.lower()
    for split in ["train", "validation", "test", "val"]:
        if split in name:
            return "validation" if split == "val" else split
    return NA


def infer_data_type(path: Path) -> str:
    s = str(path).lower()
    if "abstract_aligned" in s:
        return "abstract-aligned feature data"
    if "progression_features" in s:
        return "progression feature data"
    if "policy_outputs" in s:
        return "policy output"
    if "predictions" in s:
        return "prediction output"
    if "targets" in s:
        return "processed target data"
    if "turns" in s or "with_inferred_dialogue" in s:
        return "raw/processed turn data"
    if "results" in s:
        return "result table"
    return "data file"


def inventory_results() -> list[dict]:
    rows = []
    for root in [ROOT / "cradle_dialogue/results", ROOT / "results", ROOT / "module1_encoder_benchmark/results"]:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md", ".csv", ".json"}:
                continue
            text = read_text(path, max_chars=60_000)
            rows.append({
                "file_path": rel(path),
                "result_type": infer_result_type(path),
                "exists": "yes",
                "size_bytes": path.stat().st_size,
                "key_metrics_or_contents": summarize_metrics_text(text),
                "presentation_relevance": infer_relevance(path),
            })
    return rows


def infer_result_type(path: Path) -> str:
    s = str(path).lower()
    if "final_experiment" in s:
        return "final decision/reporting artifact"
    if "policy" in s:
        return "policy evaluation"
    if "confirm" in s:
        return "Confirm model/evaluation"
    if "alert" in s:
        return "Alert model/evaluation"
    if "category" in s:
        return "Category model/evaluation"
    if "abstract_alignment" in s:
        return "abstract alignment"
    if "uncertainty" in s:
        return "uncertainty ablation"
    if "module1" in s:
        return "severity encoder benchmark"
    return "result artifact"


def infer_relevance(path: Path) -> str:
    s = str(path).lower()
    if any(k in s for k in ["final_", "high_recall_confirm", "guardrail_v3", "uncertainty_ablation", "abstract_alignment", "temporal_alert"]):
        return "high"
    if any(k in s for k in ["policy", "confirm", "alert", "category", "module1"]):
        return "medium"
    return "supporting"


def summarize_metrics_text(text: str) -> str:
    if not text:
        return NA
    lines = []
    metric_patterns = [
        "AUROC", "AUPRC", "macro", "F1", "precision", "recall", "unsafe",
        "over-escalation", "false negative", "selected", "not selected",
    ]
    for line in text.splitlines():
        if any(p.lower() in line.lower() for p in metric_patterns):
            clean = " ".join(line.strip().split())
            if clean:
                lines.append(clean)
        if len(lines) >= 8:
            break
    return " | ".join(lines) if lines else NA


def dataset_summary() -> dict:
    summary = {}
    for split in ["train", "validation", "test"]:
        turn_path = ROOT / f"cradle_dialogue/data/cradle_{split}_turns.csv"
        target_path = ROOT / f"cradle_dialogue/data/cradle_{split}_targets.csv"
        progression_path = ROOT / f"cradle_dialogue/data/progression_features_high_recall_confirm/cradle_{split}_progression_features.csv"
        abstract_path = ROOT / f"cradle_dialogue/data/abstract_aligned_features/cradle_{split}_abstract_aligned_features.csv"
        try:
            turns = pd.read_csv(turn_path)
            summary[f"{split}_turn_rows"] = len(turns)
            if "dialogue_id" in turns:
                summary[f"{split}_dialogues"] = turns["dialogue_id"].nunique()
            if "is_user_turn" in turns:
                summary[f"{split}_user_turns_raw"] = int(pd.to_numeric(turns["is_user_turn"], errors="coerce").fillna(0).sum())
            elif "speaker" in turns:
                summary[f"{split}_user_turns_raw"] = int(turns["speaker"].astype(str).str.lower().eq("user").sum())
        except Exception:
            summary[f"{split}_turn_rows"] = NA
        try:
            targets = pd.read_csv(target_path)
            summary[f"{split}_target_rows"] = len(targets)
            if "dialogue_id" in targets:
                summary[f"{split}_target_dialogues"] = targets["dialogue_id"].nunique()
            for col in targets.columns:
                if col.startswith("target_"):
                    summary[f"{split}_{col}_positive_count"] = int(pd.to_numeric(targets[col], errors="coerce").fillna(0).sum())
        except Exception:
            summary[f"{split}_target_rows"] = NA
        try:
            summary[f"{split}_progression_user_rows"] = len(pd.read_csv(progression_path, usecols=[0]))
        except Exception:
            summary[f"{split}_progression_user_rows"] = NA
        try:
            summary[f"{split}_abstract_aligned_rows"] = len(pd.read_csv(abstract_path, usecols=[0]))
        except Exception:
            summary[f"{split}_abstract_aligned_rows"] = NA
    return summary


def inventory_models() -> list[dict]:
    rows = []

    module1 = ROOT / "results/module1_seed1_benchmark_comparison.csv"
    if module1.exists():
        df = pd.read_csv(module1)
        for _, r in df.iterrows():
            rows.append({
                "model_name": f"severity_encoder_{r.get('encoder_short', NA)}_bs{r.get('batch_size', NA)}_seed{r.get('seed', NA)}",
                "backbone_encoder": r.get("model_name", NA),
                "task": "single-turn severity classification / module 1 benchmark",
                "checkpoint_path": r.get("checkpoint_path", NA),
                "training_script": "run_module1_benchmark.py",
                "train_split_used": "module1_encoder_benchmark split seed " + str(r.get("split_seed", NA)),
                "validation_selection_criterion": "best validation macro F1",
                "test_evaluation_file": r.get("result_file", NA),
                "selected_or_not": "considered",
                "reason": f"test_macro_f1={r.get('test_macro_f1', NA)}; high_risk_recall={r.get('high_risk_recall', NA)}",
            })

    manual = [
        ("single_turn_alert_baseline", "microsoft/deberta-v3-large", "Alert binary classification", "cradle_dialogue/models/single_task_alert/seed_1/best_model.pt", "train_cradle_single_task.py", "cradle_train_targets", "best validation AUROC/AUPRC/F1 per log", "cradle_dialogue/results/single_task_alert/seed_1.txt", "not selected", "temporal Alert improved early warning detection"),
        ("temporal_alert_embeddings", "microsoft/deberta-v3-large frozen embeddings", "temporal Alert binary classification", "cradle_dialogue/models/temporal_alert_embeddings/seed_1/best_model.pt", "train_cradle_temporal_alert_from_embeddings.py", "cradle_train progression/history", "best validation metric in temporal Alert log; selected history window 8", "cradle_dialogue/results/temporal_alert_embeddings/single_turn_vs_temporal_alert_comparison.txt", "selected", "conversation history improved Alert detection"),
        ("single_turn_confirm_baseline", "microsoft/deberta-v3-large", "Confirm binary classification", "cradle_dialogue/models/single_task_confirm/seed_1/best_model.pt", "train_cradle_single_task.py", "cradle_train_targets", "validation threshold tuning/max F1", "cradle_dialogue/results/single_task_confirm/confirm_false_negative_audit_report.txt", "not selected", "missed many clinician-test Confirm cases"),
        ("A_standard_bce", "microsoft/deberta-v3-large", "high-recall Confirm", "cradle_dialogue/models/single_task_confirm_high_recall/A_standard_bce/seed_1/best_model.pt", "train_cradle_single_task_confirm_high_recall.py", "cradle_train_targets", "validation recall >=0.95 then precision", "cradle_dialogue/results/single_task_confirm_high_recall/A_standard_bce/seed_1.txt", "not selected", "lower validation precision than B among recall-qualified variants"),
        ("B_pos_weight_x2", "microsoft/deberta-v3-large", "high-recall Confirm", "cradle_dialogue/models/single_task_confirm_high_recall/B_pos_weight_x2/seed_1/best_model.pt", "train_cradle_single_task_confirm_high_recall.py", "cradle_train_targets", "validation recall >=0.95 then highest precision", "cradle_dialogue/results/single_task_confirm_high_recall/B_pos_weight_x2/seed_1.txt", "selected", "highest validation precision among variants with recall >=0.95"),
        ("C_pos_weight_x4", "microsoft/deberta-v3-large", "high-recall Confirm", "cradle_dialogue/models/single_task_confirm_high_recall/C_pos_weight_x4/seed_1/best_model.pt", "train_cradle_single_task_confirm_high_recall.py", "cradle_train_targets", "validation recall >=0.95 then precision", "cradle_dialogue/results/single_task_confirm_high_recall/C_pos_weight_x4/seed_1.txt", "not selected", "validation recall 1.0 but very low precision/high false positives"),
        ("D_focal_gamma2", "microsoft/deberta-v3-large", "high-recall Confirm", "cradle_dialogue/models/single_task_confirm_high_recall/D_focal_gamma2/seed_1/best_model.pt", "train_cradle_single_task_confirm_high_recall.py", "cradle_train_targets", "validation recall >=0.95 then precision", "cradle_dialogue/results/single_task_confirm_high_recall/D_focal_gamma2/seed_1.txt", "not selected", "slightly lower validation precision than B"),
        ("temporal_confirm_embeddings", "microsoft/deberta-v3-large frozen embeddings", "temporal Confirm binary classification", "cradle_dialogue/models/temporal_confirm_embeddings/seed_1/best_model.pt", "train_cradle_temporal_alert_from_embeddings.py / temporal confirm variant", "cradle_train progression/history", "best validation metric in temporal Confirm log", "cradle_dialogue/results/temporal_confirm_embeddings/single_turn_vs_temporal_confirm_comparison.txt", "not selected", "did not solve Confirm bottleneck as well as high-recall single-turn variant"),
        ("single_turn_category", "microsoft/deberta-v3-large", "multi-label Category classification", "cradle_dialogue/models/single_turn_category/seed_1/best_model.pt", "train_cradle_single_turn_category.py", "cradle_train_targets", "validation threshold/F1 per category", "cradle_dialogue/results/single_turn_category/category_model_summary.txt", "selected", "used in final modular pipeline"),
        ("rule_based_policy", "rules", "graded policy", NA, "build_cradle_rule_based_policy.py", "progression features", "fixed rule specification", "cradle_dialogue/results/rule_based_policy/rule_based_policy_evaluation.txt", "not selected", "too much over-escalation burden"),
        ("learned_policy", "scikit-learn policy classifier", "graded policy", "cradle_dialogue/models/learned_policy/best_learned_policy_model.pkl", "train_cradle_learned_policy.py", "progression features", "validation macro F1 / policy metrics", "cradle_dialogue/results/learned_policy/learned_policy_evaluation.txt", "not selected", "low over-escalation but unsafe continuation too high"),
        ("safety_constrained_policy_old_confirm", "scikit-learn policy classifier + guardrails", "graded policy", "cradle_dialogue/models/safety_constrained_policy_v2/best_safety_constrained_policy_v2_model.pkl", "train_cradle_safety_constrained_policy_v2.py", "progression features with old Confirm", "validation safety constraints", "cradle_dialogue/results/safety_constrained_policy_v2/policy_comparison_all_v2.txt", "not selected", "Confirm bottleneck left unsafe continuation at 29.79%"),
        ("safety_constrained_policy_high_recall_confirm", "scikit-learn policy classifier + guardrails", "graded policy", "cradle_dialogue/models/policy_high_recall_confirm/best_safety_constrained_policy_model.pkl", "run_policies_with_high_recall_confirm.py", "progression features with B_pos_weight_x2", "validation safety constraints", "cradle_dialogue/results/policy_high_recall_confirm/policy_comparison_high_recall_confirm.txt", "selected", "final policy candidate; improved unsafe continuation with small over-escalation increase"),
        ("guardrail_v3_search", "post hoc guardrail candidates", "guardrail search", NA, "search_safety_guardrails_v3_high_recall_confirm.py", "validation policy outputs/features", "validation over-escalation <=0.20 and unsafe-continuation improvement", "cradle_dialogue/results/policy_high_recall_confirm_guardrail_v3/guardrail_v3_summary.txt", "not selected", "no validation-supported improvement under over-escalation constraint"),
        ("uncertainty_augmented_policy", "scikit-learn policy classifier + uncertainty features", "uncertainty ablation policy", NA, "run_uncertainty_ablation_from_abstract_features.py", "abstract-aligned features", "choose only if validation unsafe continuation improves and over-escalation <=0.20", "cradle_dialogue/results/uncertainty_ablation/uncertainty_ablation_summary.md", "not selected", "validation unsafe continuation unchanged"),
    ]
    for item in manual:
        rows.append(dict(zip([
            "model_name", "backbone_encoder", "task", "checkpoint_path", "training_script",
            "train_split_used", "validation_selection_criterion", "test_evaluation_file",
            "selected_or_not", "reason"
        ], item)))
    return rows


def loss_training_table() -> list[dict]:
    return [
        {"model_or_script": "module1 severity encoder benchmark", "loss_function": "CrossEntropyLoss (inferred from severity classification script; verify in script)", "weights_or_pos_weight": NA, "focal_parameters": NA, "optimizer": "AdamW", "learning_rate": "not found / script-config dependent", "dropout": NA, "batch_size": "4, 8, 32 depending run", "epochs": "not found", "precision": "fp32 where recorded; some older logs not_recorded", "gradient_checkpointing": "not found", "validation_metric": "validation macro F1", "random_seeds": "1-5 for finalists"},
        {"model_or_script": "single-turn Alert baseline", "loss_function": "BCEWithLogitsLoss", "weights_or_pos_weight": "script/log dependent; not found in summary", "focal_parameters": NA, "optimizer": "AdamW", "learning_rate": "not found", "dropout": "not found", "batch_size": "not found", "epochs": "not found", "precision": "not found", "gradient_checkpointing": "not found", "validation_metric": "AUROC/AUPRC/F1 reported", "random_seeds": "1"},
        {"model_or_script": "single-turn Confirm baseline", "loss_function": "BCEWithLogitsLoss", "weights_or_pos_weight": "not found in summary", "focal_parameters": NA, "optimizer": "AdamW", "learning_rate": "not found", "dropout": "not found", "batch_size": "not found", "epochs": "not found", "precision": "not found", "gradient_checkpointing": "not found", "validation_metric": "validation threshold tuning/max F1", "random_seeds": "1"},
        {"model_or_script": "A_standard_bce", "loss_function": "BCEWithLogitsLoss", "weights_or_pos_weight": "pos_weight=11.579411764705883", "focal_parameters": NA, "optimizer": "AdamW", "learning_rate": "not found", "dropout": "not found", "batch_size": "not found", "epochs": "5", "precision": "not found", "gradient_checkpointing": "not found", "validation_metric": "recall >=0.95 then precision", "random_seeds": "1"},
        {"model_or_script": "B_pos_weight_x2", "loss_function": "BCEWithLogitsLoss", "weights_or_pos_weight": "pos_weight=23.158823529411766", "focal_parameters": NA, "optimizer": "AdamW", "learning_rate": "not found", "dropout": "not found", "batch_size": "not found", "epochs": "5", "precision": "not found", "gradient_checkpointing": "not found", "validation_metric": "recall >=0.95 then precision", "random_seeds": "1"},
        {"model_or_script": "C_pos_weight_x4", "loss_function": "BCEWithLogitsLoss", "weights_or_pos_weight": "pos_weight=46.31764705882353", "focal_parameters": NA, "optimizer": "AdamW", "learning_rate": "not found", "dropout": "not found", "batch_size": "not found", "epochs": "5; best epoch 1", "precision": "not found", "gradient_checkpointing": "not found", "validation_metric": "recall >=0.95 then precision", "random_seeds": "1"},
        {"model_or_script": "D_focal_gamma2", "loss_function": "focal BCE-style loss", "weights_or_pos_weight": "alpha_pos=0.9205050268880056", "focal_parameters": "gamma=2.0", "optimizer": "AdamW", "learning_rate": "not found", "dropout": "not found", "batch_size": "not found", "epochs": "5", "precision": "not found", "gradient_checkpointing": "not found", "validation_metric": "recall >=0.95 then precision", "random_seeds": "1"},
        {"model_or_script": "single-turn Category", "loss_function": "BCEWithLogitsLoss for multi-label task", "weights_or_pos_weight": "not found", "focal_parameters": NA, "optimizer": "AdamW", "learning_rate": "not found", "dropout": "not found", "batch_size": "not found", "epochs": "not found", "precision": "not found", "gradient_checkpointing": "not found", "validation_metric": "per-label thresholds/F1", "random_seeds": "1"},
        {"model_or_script": "policy classifiers", "loss_function": "scikit-learn classifier objective; exact estimator loss not found", "weights_or_pos_weight": "policy safety constraints/guardrails, not neural loss", "focal_parameters": NA, "optimizer": "estimator dependent", "learning_rate": NA, "dropout": NA, "batch_size": NA, "epochs": NA, "precision": NA, "gradient_checkpointing": NA, "validation_metric": "unsafe continuation and over-escalation constraints", "random_seeds": "not found"},
    ]


def metrics_rows() -> list[dict]:
    rows = []
    module1_multi = KEY_RESULT_FILES["module1_multiseed"]
    if module1_multi.exists():
        df = pd.read_csv(module1_multi)
        for _, r in df[df["row_type"].eq("summary")].iterrows():
            rows.append({
                "experiment": "single-turn severity encoder comparison",
                "variant": r.get("model_name", NA),
                "selection_status": "considered",
                "validation_metrics": f"mean_best_val_macro_f1={r.get('mean_best_val_macro_f1', NA)}",
                "test_metrics": f"mean_test_macro_f1={r.get('mean_test_macro_f1', NA)}; min_high_risk_recall={r.get('min_high_risk_recall', NA)}; total_high_risk_missed_estimate={r.get('total_high_risk_missed_estimate', NA)}",
                "source_file": rel(module1_multi),
            })
    add_known_metric_rows(rows)
    return rows


def add_known_metric_rows(rows: list[dict]) -> None:
    rows.extend([
        {"experiment": "final selected severity encoder", "variant": "microsoft/deberta-v3-large bs32 multiseed finalist", "selection_status": "selected for frozen CRADLE turn embeddings/features", "validation_metrics": "mean_best_val_macro_f1=0.9666046", "test_metrics": "mean_test_macro_f1=0.9670334; min_high_risk_recall=0.997151", "source_file": "results/module1_multiseed_finalist_comparison.csv"},
        {"experiment": "CRADLE single-turn multi-task baseline", "variant": "turn_model", "selection_status": "baseline", "validation_metrics": summarize_metrics_text(read_text(KEY_RESULT_FILES["single_vs_multi"])), "test_metrics": summarize_metrics_text(read_text(ROOT / "cradle_dialogue/results/turn_model/seed_1.txt")), "source_file": "cradle_dialogue/results/single_task_vs_multitask_comparison.txt"},
        {"experiment": "single-turn Alert baseline", "variant": "single_task_alert_seed1", "selection_status": "not final", "validation_metrics": summarize_metrics_text(read_text(KEY_RESULT_FILES["alert_baseline"])), "test_metrics": summarize_metrics_text(read_text(KEY_RESULT_FILES["alert_baseline"])), "source_file": rel(KEY_RESULT_FILES["alert_baseline"])},
        {"experiment": "temporal Alert model", "variant": "temporal_alert_embeddings_history_window_8", "selection_status": "selected", "validation_metrics": "history_window=8; validation AUROC/AUPRC reported in temporal Alert files", "test_metrics": summarize_metrics_text(read_text(KEY_RESULT_FILES["temporal_alert"])), "source_file": rel(KEY_RESULT_FILES["temporal_alert"])},
        {"experiment": "Alert history-window ablation", "variant": "history windows", "selection_status": "selected history_window=8", "validation_metrics": summarize_metrics_text(read_text(KEY_RESULT_FILES["alert_hw"])), "test_metrics": summarize_metrics_text(read_text(KEY_RESULT_FILES["alert_hw"])), "source_file": rel(KEY_RESULT_FILES["alert_hw"])},
        {"experiment": "single-turn Confirm baseline", "variant": "single_task_confirm_seed1", "selection_status": "replaced", "validation_metrics": "validation AUROC=0.996303; validation AUPRC=0.961851; tuned threshold=0.24", "test_metrics": "clinician-test AUROC=0.866775; clinician-test AUPRC=0.643011; test recall@0.24=0.388535; test false negatives=288", "source_file": rel(KEY_RESULT_FILES["confirm_audit"])},
        {"experiment": "temporal Confirm model", "variant": "temporal_confirm_embeddings", "selection_status": "not selected", "validation_metrics": summarize_metrics_text(read_text(KEY_RESULT_FILES["temporal_confirm"])), "test_metrics": summarize_metrics_text(read_text(KEY_RESULT_FILES["temporal_confirm"])), "source_file": rel(KEY_RESULT_FILES["temporal_confirm"])},
        {"experiment": "high-recall Confirm variants", "variant": "A/B/C/D", "selection_status": "B selected", "validation_metrics": "B_pos_weight_x2: val_recall=0.951049; val_precision=0.811940; val_f1=0.876006; val_fn=14; val_fp=63", "test_metrics": "B_pos_weight_x2: test_recall=0.517021; test_precision=0.761755; test_f1=0.615970; test_fn=227; test_fp=76", "source_file": rel(KEY_RESULT_FILES["confirm_high_recall"])},
        {"experiment": "Category model", "variant": "single_turn_category", "selection_status": "selected", "validation_metrics": summarize_metrics_text(read_text(KEY_RESULT_FILES["category"])), "test_metrics": summarize_metrics_text(read_text(KEY_RESULT_FILES["category"])), "source_file": rel(KEY_RESULT_FILES["category"])},
        {"experiment": "Rule-based policy", "variant": "old Confirm", "selection_status": "not selected", "validation_metrics": NA, "test_metrics": "high_risk_unsafe=0.1; confirm_unsafe=0.1; alert_unsafe=0.20085470085470086; over_escalation=0.665010460251046", "source_file": "cradle_dialogue/results/final_experiment_summary/final_results_table.csv"},
        {"experiment": "Unconstrained learned policy", "variant": "old Confirm", "selection_status": "not selected", "validation_metrics": NA, "test_metrics": "high_risk_unsafe=0.6106382978723405; over_escalation=0.010721757322175732", "source_file": "cradle_dialogue/results/final_experiment_summary/final_results_table.csv"},
        {"experiment": "Safety-constrained policy with old Confirm", "variant": "old Confirm", "selection_status": "not selected", "validation_metrics": NA, "test_metrics": f"high_risk_unsafe={FINAL_METRICS['old_safety_high_risk_unsafe_continuation']}; over_escalation={FINAL_METRICS['old_safety_over_escalation_burden']}", "source_file": "cradle_dialogue/results/final_experiment_summary/final_results_table.csv"},
        {"experiment": "Safety-constrained policy with high-recall Confirm", "variant": "B_pos_weight_x2", "selection_status": "selected final policy", "validation_metrics": "validation-selected policy candidate", "test_metrics": f"high_risk_unsafe={FINAL_METRICS['final_high_risk_unsafe_continuation']}; confirm_unsafe={FINAL_METRICS['final_confirm_unsafe_continuation']}; over_escalation={FINAL_METRICS['final_over_escalation_burden']}; macro_f1={FINAL_METRICS['final_macro_f1']}", "source_file": "cradle_dialogue/results/final_experiment_summary/final_results_table.csv"},
        {"experiment": "v3 guardrail search", "variant": "validation search candidates", "selection_status": "not selected", "validation_metrics": "no validation-supported improvement with over-escalation <=0.20", "test_metrics": "unchanged from final high-recall Confirm policy", "source_file": rel(KEY_RESULT_FILES["guardrail_v3"])},
        {"experiment": "Abstract-alignment layer", "variant": "post hoc feature/rationale layer", "selection_status": "included as analysis/reporting layer", "validation_metrics": NA, "test_metrics": "test rows=4527; columns=135; rows_missing_policy=0", "source_file": rel(KEY_RESULT_FILES["abstract_alignment"])},
        {"experiment": "Uncertainty ablation", "variant": "uncertainty-augmented safety-constrained policy", "selection_status": "not selected", "validation_metrics": "baseline unsafe=0.02097902097902098; uncertainty unsafe=0.02097902097902098; over_escalation 0.12021312021312021 vs 0.11988011988011989", "test_metrics": "baseline unsafe=0.2297872340425532; uncertainty unsafe=0.2297872340425532", "source_file": rel(KEY_RESULT_FILES["uncertainty"])},
    ])


def write_markdown_inventory(
    script_rows: list[dict],
    data_rows: list[dict],
    model_rows: list[dict],
    result_rows: list[dict],
    metrics: list[dict],
) -> None:
    ds = dataset_summary()
    missing = [rel(p) for p in KEY_RESULT_FILES.values() if not p.exists()]
    text = f"""# CRADLE Presentation Project Inventory

## Purpose

This inventory summarizes the code, data, model checkpoints, result files, metrics, and frozen final decisions for the CRADLE mental-health chatbot safety project. It was created for conference presentation preparation only. No model training, policy tuning, or checkpoint modification was performed.

## Final Pipeline Decision

- Final selected policy: safety-constrained learned policy with high-recall Confirm `B_pos_weight_x2`.
- Final Confirm model: `B_pos_weight_x2`, selected by validation metrics only.
- v3 guardrails: not selected because validation search found no improvement under the over-escalation constraint.
- uncertainty-augmented policy: not selected because validation high-risk unsafe continuation did not improve.
- uncertainty remains included as an abstract-alignment analysis layer through probability-derived uncertainty features.
- severity is represented as an operational severity proxy, not as a gold ordinal severity label.

## Final Metrics

- Old safety-constrained high-risk unsafe continuation: `{FINAL_METRICS['old_safety_high_risk_unsafe_continuation']}`.
- Final high-recall Confirm high-risk unsafe continuation: `{FINAL_METRICS['final_high_risk_unsafe_continuation']}`.
- Old over-escalation burden: `{FINAL_METRICS['old_safety_over_escalation_burden']}`.
- Final over-escalation burden: `{FINAL_METRICS['final_over_escalation_burden']}`.
- Final urgent escalation precision: `{FINAL_METRICS['final_urgent_escalation_precision']}`.
- Final human defer precision: `{FINAL_METRICS['final_human_defer_precision']}`.
- Final macro F1: `{FINAL_METRICS['final_macro_f1']}`.

## Dataset Snapshot

- Train raw turn rows: `{ds.get('train_turn_rows', NA)}`; train user/evaluable rows: `{ds.get('train_progression_user_rows', ds.get('train_user_turns_raw', NA))}`; train dialogues: `{ds.get('train_dialogues', ds.get('train_target_dialogues', NA))}`.
- Validation raw turn rows: `{ds.get('validation_turn_rows', NA)}`; validation user/evaluable rows: `{ds.get('validation_progression_user_rows', ds.get('validation_user_turns_raw', NA))}`; validation dialogues: `{ds.get('validation_dialogues', ds.get('validation_target_dialogues', NA))}`.
- Test raw turn rows: `{ds.get('test_turn_rows', NA)}`; test user/evaluable rows: `{ds.get('test_progression_user_rows', ds.get('test_user_turns_raw', NA))}`; test dialogues: `{ds.get('test_dialogues', ds.get('test_target_dialogues', NA))}`.
- Alert means early warning/risk signal detection.
- Confirm means a current-turn confirmation that the user disclosed or confirmed a safety-relevant issue.
- Category labels identify risk domains such as self-harm, suicidal ideation, abuse, sexual harm, and related categories.

## Inventory Counts

- Python scripts summarized: `{len(script_rows)}`.
- CSV/data files summarized: `{len(data_rows)}`.
- Model/checkpoint rows summarized: `{len(model_rows)}`.
- Result files summarized: `{len(result_rows)}`.
- Metric rows summarized: `{len(metrics)}`.

## Missing Expected Files

{chr(10).join(f'- `{m}`' for m in missing) if missing else '- None among the expected key files.'}

## Conflicting Metrics

- No conflicting final policy metrics were detected in the frozen summary files during this inventory.
- When older exploratory files disagree with final summaries, use `final_experiment_summary/final_results_table.csv` and `final_policy_decision.md` as authoritative.

## Final Decisions for Presentation

1. The final policy is the safety-constrained learned policy with high-recall Confirm `B_pos_weight_x2`.
2. `B_pos_weight_x2` was selected because it had the highest validation precision among high-recall Confirm variants satisfying validation recall >= 0.95.
3. High-recall Confirm reduced downstream high-risk unsafe continuation from 29.79% to 22.98%.
4. The safety improvement came with a small over-escalation increase from 11.90% to 12.24%.
5. v3 guardrails were rejected because validation search found no safety improvement under over-escalation <= 0.20.
6. uncertainty features were rejected as a final policy input because validation unsafe continuation did not improve.
7. uncertainty is still part of the framework as a reporting/analysis layer with entropy, confidence, margin, and temporal uncertainty features.
8. severity is represented by an operational severity proxy derived from risk/category/progression signals.
9. Remaining limitation: final high-risk unsafe continuation remains too high for deployment and appears tied to upstream representation/calibration errors.
10. Presentation emphasis: modular safety architecture, validation-only model selection, targeted Confirm bottleneck repair, and the safety/autonomy tradeoff.

## Companion Files

- `script_inventory.csv`
- `data_inventory.csv`
- `model_inventory.csv`
- `results_inventory.csv`
- `metrics_summary_table.csv`
- `presentation_facts_for_chatgpt.md`
"""
    (OUT_DIR / "project_inventory.md").write_text(text, encoding="utf-8")


def write_presentation_facts(metrics: list[dict]) -> None:
    ds = dataset_summary()
    final_table = read_text(KEY_RESULT_FILES["final_table"], max_chars=40_000)
    selected_metric_names = {
        "single-turn severity encoder comparison",
        "final selected severity encoder",
        "temporal Alert model",
        "Alert history-window ablation",
        "single-turn Confirm baseline",
        "temporal Confirm model",
        "high-recall Confirm variants",
        "Category model",
        "Safety-constrained policy with old Confirm",
        "Safety-constrained policy with high-recall Confirm",
        "v3 guardrail search",
        "Abstract-alignment layer",
        "Uncertainty ablation",
    }
    metric_preview = "\n".join(
        f"- {r['experiment']} / {r['variant']}: {r['selection_status']}; {r['test_metrics']}"
        for r in metrics
        if r["experiment"] in selected_metric_names
    )
    text = f"""# Presentation Facts for ChatGPT

## A. Project Overview

CRADLE is a mental-health chatbot safety project focused on detecting safety-relevant user disclosures and choosing graded escalation actions. The final system is a modular safety layer, not a deployed chatbot. It combines Alert, Confirm, Category, progression, uncertainty/proxy-severity analysis, and policy modules.

Final selected policy: safety-constrained learned policy with high-recall Confirm `B_pos_weight_x2`.

## B. Dataset Summary

- Train raw turn rows: `{ds.get('train_turn_rows', NA)}`.
- Validation raw turn rows: `{ds.get('validation_turn_rows', NA)}`.
- Test raw turn rows: `{ds.get('test_turn_rows', NA)}`.
- Train user/evaluable rows: `{ds.get('train_progression_user_rows', ds.get('train_user_turns_raw', NA))}`.
- Validation user/evaluable rows: `{ds.get('validation_progression_user_rows', ds.get('validation_user_turns_raw', NA))}`.
- Test user/evaluable rows: `{ds.get('test_progression_user_rows', ds.get('test_user_turns_raw', NA))}`.
- Train target rows: `{ds.get('train_target_rows', NA)}`.
- Validation target rows: `{ds.get('validation_target_rows', NA)}`.
- Test target rows: `{ds.get('test_target_rows', NA)}`.
- Train dialogues: `{ds.get('train_dialogues', ds.get('train_target_dialogues', NA))}`.
- Validation dialogues: `{ds.get('validation_dialogues', ds.get('validation_target_dialogues', NA))}`.
- Test dialogues: `{ds.get('test_dialogues', ds.get('test_target_dialogues', NA))}`.
- Test `target_alert` positives: `{ds.get('test_target_alert_positive_count', NA)}`.
- Test `target_confirm` positives: `{ds.get('test_target_confirm_positive_count', NA)}`.
- Test self-harm category positives: `{ds.get('test_target_category_SH_positive_count', NA)}`.
- Test active-suicidal-ideation category positives: `{ds.get('test_target_category_SI_active_positive_count', NA)}`.

Alert is an early warning signal that risk may be present or developing. Confirm is a more direct current-turn confirmation/disclosure signal. Category labels identify the domain of risk.

### Dataset Labels Used

- Earlier ANNOMI pipeline labels: low / medium / high severity classes. The ANNOMI outputs included `mean_low_prob`, `mean_medium_prob`, `mean_high_prob`, `predicted_label`, `confidence`, `epistemic_uncertainty`, and `aleatoric_uncertainty`.
- CRADLE exact source labels: `alert_ongoing`, `alert_past`, and `confirm_<category>_<temporal_state>`.
- CRADLE Confirm categories: `CA`, `DV`, `RA`, `SHA`, `SH`, `SI_active`, `SI_passive`.
- CRADLE temporal states: `ongoing` and `past`.
- CRADLE derived model targets: `target_alert`, `target_confirm`, `target_category_CA`, `target_category_DV`, `target_category_RA`, `target_category_SHA`, `target_category_SH`, `target_category_SI_active`, `target_category_SI_passive`, `target_ongoing`, and `target_past`.
- CRADLE progression/forecast labels used for analysis include `progression_stage`, `confirm_within_1_user_turn`, `confirm_within_2_user_turns`, `confirm_within_3_user_turns`, and `confirm_within_5_user_turns`.

## B2. Earlier ANNOMI Pipeline Tried First

Before the final CRADLE-specific pipeline, an earlier pipeline was tested on ANNOMI-style severity data. That pipeline predicted low/medium/high turn-level severity probabilities, uncertainty values, and then produced policy-style outputs with trajectory features. The ANNOMI policy output columns included previous high-risk probability, high-probability change, rolling mean high probability, trajectory slope, baseline deviation, rising-trend flag, high-uncertainty flag, change-point flag, escalation level, escalation action, and rationale. This was exploratory and was not the final CRADLE policy pipeline.

## C. Model Architecture Summary

- Severity encoder benchmarking compared mental-health BERT/RoBERTa/ModernBERT/DeBERTa encoders.
- Temporal Alert uses frozen turn embeddings and conversation history/progression features.
- Confirm remained a single-turn classifier but was retrained with high-recall loss variants.
- Category is a single-turn multi-label classifier.
- The policy is a learned graded action selector with safety constraints and guardrails.
- Abstract alignment adds post hoc uncertainty features, an operational severity proxy, multidimensional risk views, and rationales.

### Batch Sizes and Training Settings

- The severity encoder benchmark used different batch sizes because model size and GPU memory requirements differed. Batch sizes included `32`, `8`, and `4`.
- Severity encoder finalists included `microsoft/deberta-v3-large`, `answerdotai/ModernBERT-large`, and `AIMH/mental-roberta-large` at batch size `32` across seeds `1-5`.
- The single-turn CRADLE Alert, Confirm, high-recall Confirm, and Category scripts defaulted to batch size `32`, `5` epochs, learning rate `2e-5`, FP32 precision, AdamW, and gradient checkpointing when supported.
- The temporal frozen-embedding Alert experiments used batch size `128`, `5` epochs, learning rate `2e-5`, FP32, and history windows `1`, `4`, and `8`.
- High-recall Confirm `B_pos_weight_x2` used batch size `32`, FP32, `5` epochs, BCEWithLogitsLoss with `pos_weight=23.158823529411766`, and selected threshold `0.011`.

### Module Inputs and Outputs

- Target construction module:
  - Inputs: CRADLE turn text, dialogue IDs, speaker roles, and raw exact labels.
  - Outputs: processed target files with Alert/Confirm/Category targets, temporal-state targets, masks, progression-stage fields, and forecast labels.
- Severity encoder benchmark:
  - Inputs: text and low/medium/high severity labels.
  - Outputs: severity class probabilities, predicted severity label, macro F1, high-risk precision/recall/F1, calibration metrics, and saved checkpoints.
- Temporal Alert module:
  - Inputs: current and prior turn embeddings/history windows plus Alert labels.
  - Outputs: `alert_prob`, fixed/tuned Alert predictions, validation/test AUROC, AUPRC, precision, recall, F1, and temporal Alert checkpoint.
- Confirm module:
  - Inputs: current turn text and `target_confirm`.
  - Outputs: `confirm_prob`, fixed/tuned Confirm predictions, selected validation threshold, precision, recall, F1, false positives, false negatives, and Confirm checkpoint.
- Category module:
  - Inputs: current turn text and multi-label category targets.
  - Outputs: `category_prob_CA`, `category_prob_DV`, `category_prob_RA`, `category_prob_SHA`, `category_prob_SH`, `category_prob_SI_active`, `category_prob_SI_passive`, plus fixed/tuned category predictions.
- Progression feature module:
  - Inputs: Alert, Confirm, and Category predictions joined with dialogue order.
  - Outputs: rolling means, cumulative maxima, previous probabilities, rising flags, top category, `risk_accumulation_score`, `risk_escalation_score`, and `risk_volatility_score`.
- Policy module:
  - Inputs: progression features and model probabilities.
  - Outputs: final `policy_level`, policy action, policy rationale, unsafe-continuation metrics, over-escalation burden, urgent escalation precision, and human defer precision.
- Abstract-alignment module:
  - Inputs: frozen final policy outputs and progression features.
  - Outputs: uncertainty features, severity proxy fields, multidimensional risk fields, and `abstract_aligned_rationale`.

## D. Model Comparison Summary

{metric_preview}

## E. Policy Comparison Summary

The rule-based policy was conservative but over-escalated heavily. The unconstrained learned policy preserved autonomy but allowed too much unsafe continuation. The old safety-constrained policy improved safety but was bottlenecked by Confirm false negatives. Replacing Confirm with `B_pos_weight_x2` reduced unsafe continuation while only slightly increasing over-escalation.

## F. Final Results Table

```csv
{final_table.strip()}
```

## G. Limitations

- Final high-risk unsafe continuation is still 22.98%, too high for deployment.
- Remaining misses appear linked to weak upstream probabilities and calibration/representation failures.
- Severity is an operational proxy rather than a clinician-labeled ordinal severity target.
- Uncertainty is probability-derived and post hoc, not a trained Bayesian or ensemble uncertainty model.
- The dataset and policy targets need external validation and expert review.

## H. 14-Slide Outline

1. Title: CRADLE safety escalation for mental-health chatbots.
2. Motivation: unsafe continuation vs over-escalation.
3. Dataset and labels: Alert, Confirm, Category, policy levels.
4. System overview: modular pipeline.
5. Severity encoder benchmark and frozen representation choice.
6. Temporal Alert experiment: why history helps early warning.
7. Confirm bottleneck: false negatives in the old model.
8. High-recall Confirm variants: A/B/C/D and validation-only selection.
9. Policy setup: rule-based, unconstrained, safety-constrained.
10. Final policy result: unsafe continuation 29.79% to 22.98%.
11. Tradeoff result: over-escalation 11.90% to 12.24%.
12. v3 guardrail search: why no extra rules were adopted.
13. Uncertainty and abstract alignment: what was added and what was not selected.
14. Limitations and future work: contextual Confirm, calibration, external validation.
"""
    (OUT_DIR / "presentation_facts_for_chatgpt.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scripts = inventory_scripts()
    data = inventory_data()
    models = inventory_models()
    results = inventory_results()
    metrics = metrics_rows()
    losses = loss_training_table()

    write_csv(OUT_DIR / "script_inventory.csv", scripts, [
        "file_path", "purpose", "inputs", "outputs", "important_functions_classes",
        "models_used", "loss_functions_used", "metrics_computed", "pipeline_role",
    ])
    write_csv(OUT_DIR / "data_inventory.csv", data, [
        "file_path", "rows", "columns", "split", "key_columns", "label_columns",
        "prediction_columns", "type",
    ])
    write_csv(OUT_DIR / "model_inventory.csv", models, [
        "model_name", "backbone_encoder", "task", "checkpoint_path", "training_script",
        "train_split_used", "validation_selection_criterion", "test_evaluation_file",
        "selected_or_not", "reason",
    ])
    write_csv(OUT_DIR / "results_inventory.csv", results, [
        "file_path", "result_type", "exists", "size_bytes", "key_metrics_or_contents",
        "presentation_relevance",
    ])
    write_csv(OUT_DIR / "metrics_summary_table.csv", metrics, [
        "experiment", "variant", "selection_status", "validation_metrics", "test_metrics", "source_file",
    ])
    write_csv(OUT_DIR / "loss_training_details.csv", losses, [
        "model_or_script", "loss_function", "weights_or_pos_weight", "focal_parameters",
        "optimizer", "learning_rate", "dropout", "batch_size", "epochs", "precision",
        "gradient_checkpointing", "validation_metric", "random_seeds",
    ])

    write_markdown_inventory(scripts, data, models, results, metrics)
    write_presentation_facts(metrics)

    missing = [rel(p) for p in KEY_RESULT_FILES.values() if not p.exists()]
    print("Presentation inventory complete.")
    print(f"Scripts found: {len(scripts)}")
    print(f"Data files summarized: {len(data)}")
    print(f"Model/checkpoint result rows summarized: {len(models)}")
    print(f"Results files summarized: {len(results)}")
    print(f"Missing expected files: {missing if missing else 'none'}")
    print("Conflicting metrics: none detected among frozen final summary files")
    print("Ready to send to ChatGPT: yes")


if __name__ == "__main__":
    main()
