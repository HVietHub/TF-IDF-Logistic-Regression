"""
Evaluation Module
=================
Model evaluation, metrics computation, confusion matrix export, and experiment comparison.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    pipeline: Pipeline,
    X: pd.Series,
    y: pd.Series,
    label_names: Dict[int, str],
    dataset_name: str = "Validation",
) -> Dict[str, Any]:
    """Evaluate a trained pipeline and return comprehensive metrics."""
    classes = sorted(label_names.keys())
    target_names = [label_names[c] for c in classes]

    # Inference timing
    t0 = time.time()
    y_pred = pipeline.predict(X)
    inference_time = time.time() - t0

    y_proba = pipeline.predict_proba(X)

    # Compute metrics
    acc = accuracy_score(y, y_pred)
    macro_p = precision_score(y, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

    # Per-class report
    report = classification_report(
        y, y_pred, target_names=target_names, labels=classes,
        output_dict=True, zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=classes)

    metrics = {
        "dataset": dataset_name,
        "num_samples": len(y),
        "accuracy": round(acc, 5),
        "macro_precision": round(macro_p, 5),
        "macro_recall": round(macro_r, 5),
        "macro_f1": round(macro_f1, 5),
        "weighted_f1": round(weighted_f1, 5),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "inference_time_seconds": round(inference_time, 4),
        "inference_per_sample_ms": round(inference_time / len(y) * 1000, 4),
    }

    # Per-class metrics
    for cls_id, cls_name in label_names.items():
        if cls_name in report:
            cls_report = report[cls_name]
            metrics["per_class"][cls_name] = {
                "precision": round(cls_report["precision"], 5),
                "recall": round(cls_report["recall"], 5),
                "f1-score": round(cls_report["f1-score"], 5),
                "support": int(cls_report["support"]),
            }

    # Print summary
    print(f"\n--- {dataset_name} Evaluation ---")
    print(f"Samples: {len(y)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro    P: {macro_p:.4f} | R: {macro_r:.4f} | F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Inference: {inference_time:.3f}s ({inference_time/len(y)*1000:.3f} ms/sample)")
    print()
    print(classification_report(y, y_pred, target_names=target_names, labels=classes, zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    return metrics


def generate_classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Dict[int, str],
) -> Dict:
    """Generate a classification report as a dictionary."""
    classes = sorted(label_names.keys())
    target_names = [label_names[c] for c in classes]
    return classification_report(
        y_true, y_pred, target_names=target_names, labels=classes,
        output_dict=True, zero_division=0,
    )


def export_confusion_matrix_csv(
    pipeline: Pipeline,
    X: pd.Series,
    y: pd.Series,
    label_names: Dict[int, str],
    output_path: str,
) -> None:
    """Export confusion matrix as CSV."""
    classes = sorted(label_names.keys())
    target_names = [label_names[c] for c in classes]

    y_pred = pipeline.predict(X)
    cm = confusion_matrix(y, y_pred, labels=classes)

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in target_names],
        columns=[f"pred_{name}" for name in target_names],
    )
    cm_df.to_csv(output_path)
    print(f"Confusion matrix saved: {output_path}")


def plot_confusion_matrix(
    pipeline: Pipeline,
    X: pd.Series,
    y: pd.Series,
    label_names: Dict[int, str],
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix as an image."""
    classes = sorted(label_names.keys())
    target_names = [label_names[c] for c in classes]

    y_pred = pipeline.predict(X)
    cm = confusion_matrix(y, y_pred, labels=classes)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names,
        ax=axes[0],
    )
    axes[0].set_title(f"{title} (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized (row-wise)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".3f", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names,
        ax=axes[1],
    )
    axes[1].set_title(f"{title} (Normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix plot saved: {output_path}")


def compare_experiments(results: List[Dict]) -> pd.DataFrame:
    """Compare multiple experiment results into a summary table."""
    rows = []
    for r in results:
        row = {
            "experiment": r.get("experiment_name", "unknown"),
            "accuracy": r.get("accuracy", 0),
            "macro_precision": r.get("macro_precision", 0),
            "macro_recall": r.get("macro_recall", 0),
            "macro_f1": r.get("macro_f1", 0),
            "weighted_f1": r.get("weighted_f1", 0),
            "train_time_s": r.get("train_time_seconds", 0),
            "inference_time_s": r.get("inference_time_seconds", 0),
        }
        # Add per-class recall (critical for security)
        for cls_name in ["benign", "promptinjection", "jailbreak"]:
            cls_data = r.get("per_class", {}).get(cls_name, {})
            row[f"recall_{cls_name}"] = cls_data.get("recall", 0)
            row[f"f1_{cls_name}"] = cls_data.get("f1-score", 0)
        rows.append(row)

    return pd.DataFrame(rows)
