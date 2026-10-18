"""
External Test Dataset Evaluation Script
========================================
Evaluates the trained TF-IDF + Logistic Regression model on an external test dataset (e.g. test.csv).
Computes detailed classification metrics, security decision distribution, confusion matrix,
and exports comprehensive reports and misclassification logs.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from preprocessing import preprocess_text
from data_validation import LABEL_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def map_labels(series: pd.Series, label_mapping: Dict[int, str]) -> pd.Series:
    """Map string or int labels to numeric IDs."""
    name_to_id = {v.lower().strip(): k for k, v in label_mapping.items()}
    # Also handle common variants
    name_to_id["jailbreak"] = 2
    name_to_id["benign"] = 0
    name_to_id["promptinjection"] = 1
    name_to_id["injection"] = 1
    name_to_id["prompt_injection"] = 1

    def convert_label(val):
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, str):
            val_clean = val.lower().strip()
            if val_clean.isdigit():
                return int(val_clean)
            if val_clean in name_to_id:
                return name_to_id[val_clean]
        raise ValueError(f"Unrecognized label value: {val}")

    return series.apply(convert_label)


def compute_security_decision(
    proba: np.ndarray,
    classes: np.ndarray,
    benign_allow_min: float = 0.70,
    attack_block_min: float = 0.60,
) -> str:
    """Apply security threshold logic to determine allow/block/review."""
    classes_list = list(classes)
    benign_idx = classes_list.index(0) if 0 in classes_list else None
    attack_indices = [j for j, c in enumerate(classes_list) if c != 0]

    p_benign = float(proba[benign_idx]) if benign_idx is not None else 0.0
    p_attack = float(max([proba[j] for j in attack_indices])) if attack_indices else 0.0

    if p_benign >= benign_allow_min and p_attack < (1.0 - benign_allow_min):
        return "allow"
    elif p_attack >= attack_block_min:
        return "block"
    else:
        return "review"


def run_evaluation(
    data_path: str,
    model_path: str,
    label_mapping_path: str,
    output_dir: str,
    text_col: str = "text",
    label_col: str = "label",
    benign_threshold: float = 0.70,
    attack_threshold: float = 0.60,
) -> Dict[str, Any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Model and Artifacts
    logger.info(f"Loading model pipeline from: {model_path}")
    pipeline = joblib.load(model_path)
    model_classes = pipeline.classes_
    logger.info(f"Model classes: {model_classes}")

    logger.info(f"Loading label mapping from: {label_mapping_path}")
    with open(label_mapping_path, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)
    label_names = {int(k): v for k, v in raw_mapping.items()}

    # 2. Load Dataset
    logger.info(f"Loading test dataset from: {data_path}")
    ext = Path(data_path).suffix.lower()
    if ext == ".jsonl":
        records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    item = json.loads(line_str)
                    raw_lbl = str(item.get("label", "")).strip().lower()
                    fam = str(item.get("attack_family", "")).strip().upper() if item.get("attack_family") else ""
                    if raw_lbl in ("attack", "1") and fam == "PI":
                        mapped_lbl = "promptinjection"
                    elif raw_lbl in ("attack", "1") and fam == "JB":
                        mapped_lbl = "jailbreak"
                    elif raw_lbl in ("benign", "0"):
                        mapped_lbl = "benign"
                    else:
                        mapped_lbl = raw_lbl
                    records.append({
                        "id": item.get("id"),
                        "text": item.get("text", ""),
                        "label": mapped_lbl,
                        "attack_family": fam,
                    })
                except Exception:
                    continue
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(data_path)
    total_raw = len(df)
    logger.info(f"Loaded {total_raw} rows. Columns: {df.columns.tolist()}")

    # Find text and label columns if named differently
    if text_col not in df.columns:
        for candidate in ["prompt", "sentence", "content", "input"]:
            if candidate in df.columns:
                text_col = candidate
                break
    if label_col not in df.columns:
        for candidate in ["labels", "target", "class"]:
            if candidate in df.columns:
                label_col = candidate
                break

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Could not find text column '{text_col}' or label column '{label_col}' in {df.columns.tolist()}")

    # Clean missing values if any
    null_texts = df[text_col].isnull().sum()
    null_labels = df[label_col].isnull().sum()
    if null_texts > 0 or null_labels > 0:
        logger.warning(f"Dropping {null_texts} null texts and {null_labels} null labels")
        df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)

    # Map labels to numeric IDs
    y_true = map_labels(df[label_col], label_names).to_numpy()
    present_labels = sorted(list(set(y_true)))
    logger.info(f"Test dataset label distribution: {pd.Series(y_true).value_counts().to_dict()}")
    for lbl in present_labels:
        logger.info(f"  Class {lbl} ({label_names.get(lbl, 'unknown')}): {(y_true == lbl).sum()} samples")

    # 3. Preprocess Text
    logger.info("Applying text preprocessing (Unicode NFC & whitespace normalization)...")
    t_prep_start = time.perf_counter()
    X_processed = df[text_col].astype(str).apply(preprocess_text)
    prep_time = time.perf_counter() - t_prep_start
    logger.info(f"Preprocessing completed in {prep_time:.2f}s ({prep_time / len(df) * 1000:.3f} ms/sample)")

    # 4. Model Inference
    logger.info("Running model inference...")
    t_infer_start = time.perf_counter()
    y_pred = pipeline.predict(X_processed)
    y_proba = pipeline.predict_proba(X_processed)
    infer_time = time.perf_counter() - t_infer_start
    ms_per_sample = (infer_time / len(df)) * 1000
    throughput = len(df) / infer_time
    logger.info(f"Inference completed in {infer_time:.2f}s ({ms_per_sample:.3f} ms/sample, {throughput:.1f} samples/sec)")

    # 5. Security Decision Layer
    logger.info("Applying security threshold decision layer...")
    security_decisions = [
        compute_security_decision(
            proba,
            model_classes,
            benign_allow_min=benign_threshold,
            attack_block_min=attack_threshold,
        )
        for proba in y_proba
    ]
    df["security_decision"] = security_decisions
    decision_counts = df["security_decision"].value_counts().to_dict()

    # 6. Compute Standard Evaluation Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_p = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Classification report
    eval_labels = sorted(list(set(present_labels) | set(model_classes)))
    eval_target_names = [label_names.get(c, str(c)) for c in eval_labels]
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=eval_labels,
        target_names=eval_target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=eval_labels,
        target_names=eval_target_names,
        zero_division=0,
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=eval_labels)

    # Binary metrics if 2 classes (0: benign, 2: jailbreak)
    binary_metrics = {}
    if set(present_labels).issubset({0, 2}):
        # Map 0 -> 0 (negative), 2 -> 1 (positive)
        y_true_bin = (y_true == 2).astype(int)
        y_pred_bin = (y_pred == 2).astype(int)
        
        # Attack class probability (index of class 2)
        idx_2 = list(model_classes).index(2) if 2 in model_classes else None
        if idx_2 is not None:
            p_attack = y_proba[:, idx_2]
            try:
                roc_auc = roc_auc_score(y_true_bin, p_attack)
            except Exception:
                roc_auc = None
            
            p_vals, r_vals, _ = precision_recall_curve(y_true_bin, p_attack)
            pr_auc = auc(r_vals, p_vals)
        else:
            roc_auc = None
            pr_auc = None

        # TN, FP, FN, TP
        tn = int(cm[eval_labels.index(0)][eval_labels.index(0)])
        fp = int(cm[eval_labels.index(0)][eval_labels.index(2)])
        fn = int(cm[eval_labels.index(2)][eval_labels.index(0)])
        tp = int(cm[eval_labels.index(2)][eval_labels.index(2)])

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        binary_metrics = {
            "true_negatives_benign_correct": tn,
            "false_positives_false_alarms": fp,
            "false_negatives_missed_attacks": fn,
            "true_positives_attacks_caught": tp,
            "false_positive_rate_fpr": round(fpr, 5),
            "false_negative_rate_fnr": round(fnr, 5),
            "specificity": round(specificity, 5),
            "roc_auc": round(roc_auc, 5) if roc_auc is not None else None,
            "pr_auc": round(pr_auc, 5) if pr_auc is not None else None,
        }

    # Security policy matrix breakdown
    # Cross-tabulate true label vs security decision
    sec_crosstab = pd.crosstab(
        pd.Series(y_true).map(lambda x: label_names.get(x, str(x))),
        df["security_decision"],
    )
    sec_crosstab_dict = sec_crosstab.to_dict()

    # Calculate policy outcomes
    benign_mask = (y_true == 0)
    attack_mask = (y_true == 2)

    benign_allowed = int((benign_mask & (df["security_decision"] == "allow")).sum())
    benign_reviewed = int((benign_mask & (df["security_decision"] == "review")).sum())
    benign_blocked = int((benign_mask & (df["security_decision"] == "block")).sum())

    attack_blocked = int((attack_mask & (df["security_decision"] == "block")).sum())
    attack_reviewed = int((attack_mask & (df["security_decision"] == "review")).sum())
    attack_allowed = int((attack_mask & (df["security_decision"] == "allow")).sum())

    security_policy_summary = {
        "thresholds": {
            "benign_allow_min": benign_threshold,
            "attack_block_min": attack_threshold,
        },
        "decision_counts": decision_counts,
        "crosstab": sec_crosstab_dict,
        "benign_outcomes": {
            "total": int(benign_mask.sum()),
            "allowed_safe_pass": benign_allowed,
            "allowed_rate": round(benign_allowed / max(1, benign_mask.sum()), 5),
            "sent_to_review": benign_reviewed,
            "review_rate": round(benign_reviewed / max(1, benign_mask.sum()), 5),
            "blocked_false_alarm": benign_blocked,
            "false_alarm_rate": round(benign_blocked / max(1, benign_mask.sum()), 5),
        },
        "attack_outcomes": {
            "total": int(attack_mask.sum()),
            "blocked_threat_intercepted": attack_blocked,
            "block_rate": round(attack_blocked / max(1, attack_mask.sum()), 5),
            "sent_to_review_mitigated": attack_reviewed,
            "review_rate": round(attack_reviewed / max(1, attack_mask.sum()), 5),
            "allowed_critical_leak": attack_allowed,
            "leak_rate": round(attack_allowed / max(1, attack_mask.sum()), 5),
        },
        "overall_threat_interception_rate": round((attack_blocked + attack_reviewed) / max(1, attack_mask.sum()), 5),
    }

    # 7. Save Predictions & Misclassifications
    df["true_label_id"] = y_true
    df["true_label_name"] = [label_names.get(y, str(y)) for y in y_true]
    df["predicted_label_id"] = y_pred
    df["predicted_label_name"] = [label_names.get(p, str(p)) for p in y_pred]
    df["is_correct"] = (y_true == y_pred)

    for i, cls in enumerate(model_classes):
        cls_name = label_names.get(int(cls), str(cls))
        df[f"proba_{cls_name}"] = y_proba[:, i].round(6)

    predictions_path = out_path / "test_csv_predictions.csv"
    df.to_csv(predictions_path, index=False)
    logger.info(f"Saved predictions to: {predictions_path}")

    # Misclassifications
    df_errors = df[~df["is_correct"]].copy()
    errors_path = out_path / "test_csv_misclassifications.csv"
    df_errors.to_csv(errors_path, index=False)
    logger.info(f"Saved {len(df_errors)} misclassified cases to: {errors_path}")

    # 8. Confusion Matrix Export (CSV & PNG)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in eval_target_names],
        columns=[f"pred_{name}" for name in eval_target_names],
    )
    cm_csv_path = out_path / "test_csv_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    # Plot Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=eval_target_names,
        yticklabels=eval_target_names,
        ax=axes[0],
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[0].set_title("Test Dataset Confusion Matrix (Raw Counts)", fontsize=14, pad=12)
    axes[0].set_ylabel("Ground Truth Label", fontsize=12)
    axes[0].set_xlabel("Predicted Label", fontsize=12)

    cm_sum = cm.sum(axis=1, keepdims=True)
    # Avoid zero division if class has zero support
    cm_norm = np.divide(cm.astype(float), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".3f",
        cmap="Greens",
        xticklabels=eval_target_names,
        yticklabels=eval_target_names,
        ax=axes[1],
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[1].set_title("Test Dataset Confusion Matrix (Normalized)", fontsize=14, pad=12)
    axes[1].set_ylabel("Ground Truth Label", fontsize=12)
    axes[1].set_xlabel("Predicted Label", fontsize=12)

    plt.suptitle("Model Evaluation on External Dataset (test.csv)", fontsize=16, y=1.02)
    plt.tight_layout()
    cm_png_path = out_path / "test_csv_confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix plot to: {cm_png_path}")

    # Plot ROC and Precision-Recall Curves if binary
    if "roc_auc" in binary_metrics and binary_metrics["roc_auc"] is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fpr_arr, tpr_arr, _ = roc_curve(y_true_bin, p_attack)
        axes[0].plot(fpr_arr, tpr_arr, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {binary_metrics['roc_auc']:.4f})")
        axes[0].plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
        axes[0].set_ylabel("True Positive Rate (Recall)")
        axes[0].set_title("ROC Curve - Jailbreak Detection")
        axes[0].legend(loc="lower right")
        axes[0].grid(alpha=0.3)

        axes[1].plot(r_vals, p_vals, color="#2ca02c", lw=2.5, label=f"PR Curve (AUC = {binary_metrics['pr_auc']:.4f})")
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall Curve - Jailbreak Detection")
        axes[1].legend(loc="lower left")
        axes[1].grid(alpha=0.3)

        roc_png_path = out_path / "test_csv_roc_pr_curve.png"
        plt.tight_layout()
        plt.savefig(roc_png_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved ROC & PR curves to: {roc_png_path}")

    # 9. Compile Full Metrics Summary
    results = {
        "dataset_path": str(data_path),
        "total_samples": len(df),
        "execution_summary": {
            "preprocessing_seconds": round(prep_time, 4),
            "inference_seconds": round(infer_time, 4),
            "latency_ms_per_sample": round(ms_per_sample, 3),
            "throughput_samples_per_sec": round(throughput, 1),
        },
        "overall_metrics": {
            "accuracy": round(acc, 5),
            "macro_precision": round(macro_p, 5),
            "macro_recall": round(macro_r, 5),
            "macro_f1": round(macro_f1, 5),
            "weighted_precision": round(weighted_p, 5),
            "weighted_recall": round(weighted_r, 5),
            "weighted_f1": round(weighted_f1, 5),
        },
        "binary_security_metrics": binary_metrics,
        "security_policy_evaluation": security_policy_summary,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": eval_target_names,
    }

    metrics_json_path = out_path / "test_csv_evaluation_report.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved full JSON evaluation report to: {metrics_json_path}")

    # Print Full Console Summary
    print("\n" + "=" * 70)
    print("      MODEL TEST REPORT: C:\\Users\\FPT\\Desktop\\TF_IDF\\test.csv")
    print("=" * 70)
    print(f"Total Samples Evaluated: {len(df):,}")
    print(f"Inference Latency:       {ms_per_sample:.3f} ms/sample ({throughput:.1f} samples/sec)")
    print(f"Overall Accuracy:        {acc * 100:.2f}%")
    print(f"Macro F1-Score:          {macro_f1:.4f}")
    print(f"Weighted F1-Score:       {weighted_f1:.4f}")
    if binary_metrics:
        print(f"ROC AUC:                 {binary_metrics.get('roc_auc', 'N/A')}")
        print(f"PR AUC:                  {binary_metrics.get('pr_auc', 'N/A')}")
        print(f"False Positive Rate:     {binary_metrics.get('false_positive_rate_fpr') * 100:.2f}% ({binary_metrics.get('false_positives_false_alarms')} benign false alarms)")
        print(f"False Negative Rate:     {binary_metrics.get('false_negative_rate_fnr') * 100:.2f}% ({binary_metrics.get('false_negatives_missed_attacks')} missed jailbreaks)")
    print("-" * 70)
    print("CLASSIFICATION REPORT:")
    print(report_text)
    print("-" * 70)
    print("CONFUSION MATRIX:")
    print(cm_df.to_string())
    print("-" * 70)
    print("SECURITY DECISION LAYER OUTCOMES:")
    print(f"  Thresholds: Benign Allow >= {benign_threshold:.2f} | Attack Block >= {attack_threshold:.2f}")
    print(f"  Decisions:  Allow: {decision_counts.get('allow', 0)} | Block: {decision_counts.get('block', 0)} | Review: {decision_counts.get('review', 0)}")
    print(f"  Benign:     {benign_allowed}/{benign_mask.sum()} allowed ({benign_allowed / max(1, benign_mask.sum()) * 100:.1f}%), {benign_reviewed} review, {benign_blocked} blocked")
    print(f"  Jailbreak:  {attack_blocked}/{attack_mask.sum()} blocked ({attack_blocked / max(1, attack_mask.sum()) * 100:.1f}%), {attack_reviewed} review, {attack_allowed} allowed (critical)")
    print(f"  Overall Threat Interception Rate (Blocked + Review): {(attack_blocked + attack_reviewed) / max(1, attack_mask.sum()) * 100:.2f}%")
    print("=" * 70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test dataset")
    parser.add_argument(
        "--data",
        type=str,
        default=r"C:\Users\FPT\Desktop\TF_IDF\test.csv",
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=r"C:\Users\FPT\Desktop\TF_IDF\tf-idf\artifacts\model.joblib",
        help="Path to trained model pipeline (.joblib)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=r"C:\Users\FPT\Desktop\TF_IDF\tf-idf\artifacts\label_mapping.json",
        help="Path to label mapping JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\Users\FPT\Desktop\TF_IDF\tf-idf\reports",
        help="Directory to save evaluation reports and plots",
    )
    parser.add_argument(
        "--benign-threshold",
        type=float,
        default=0.70,
        help="Benign allow threshold (default 0.70)",
    )
    parser.add_argument(
        "--attack-threshold",
        type=float,
        default=0.60,
        help="Attack block threshold (default 0.60)",
    )

    args = parser.parse_args()

    run_evaluation(
        data_path=args.data,
        model_path=args.model,
        label_mapping_path=args.labels,
        output_dir=args.output_dir,
        benign_threshold=args.benign_threshold,
        attack_threshold=args.attack_threshold,
    )


if __name__ == "__main__":
    main()
