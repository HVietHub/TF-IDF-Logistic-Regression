"""
Data Validation Module
======================
Validates the dataset for the prompt injection/jailbreak detection pipeline.
Checks schema, types, label validity, missing values, duplicates, and conflicts.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALID_LABELS = {0, 1, 2}
LABEL_NAMES = {0: "benign", 1: "promptinjection", 2: "jailbreak"}


def load_dataset(path: str) -> pd.DataFrame:
    """Load a dataset from CSV or Parquet based on file extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use CSV or Parquet.")

    logger.info(f"Loaded dataset from {path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_schema(
    df: pd.DataFrame,
    prompt_col: str = "prompt",
    label_col: str = "label",
) -> Dict:
    """Validate dataset schema: required columns, types, and basic integrity."""
    report: Dict = {
        "total_records": len(df),
        "columns_found": df.columns.tolist(),
        "errors": [],
        "warnings": [],
    }

    # --- Check required columns ---
    for col in [prompt_col, label_col]:
        if col not in df.columns:
            report["errors"].append(f"Missing required column: '{col}'")

    if report["errors"]:
        return report

    # --- Prompt validation ---
    null_prompts = df[prompt_col].isnull().sum()
    if null_prompts > 0:
        report["warnings"].append(f"{null_prompts} null prompt(s) found")

    # Convert prompts to string and check for empties
    str_prompts = df[prompt_col].astype(str).str.strip()
    empty_prompts = (str_prompts == "").sum() + (str_prompts == "nan").sum()
    if empty_prompts > 0:
        report["warnings"].append(f"{empty_prompts} empty/nan prompt(s) found")

    # --- Label validation ---
    null_labels = df[label_col].isnull().sum()
    if null_labels > 0:
        report["errors"].append(f"{null_labels} null label(s) found")

    try:
        labels = df[label_col].dropna().astype(int)
    except (ValueError, TypeError):
        report["errors"].append(
            f"Labels cannot be converted to int. Unique values: {df[label_col].unique()[:10]}"
        )
        return report

    invalid_labels = set(labels.unique()) - VALID_LABELS
    if invalid_labels:
        report["errors"].append(f"Invalid labels found: {invalid_labels}")

    # --- Class distribution ---
    label_counts = df[label_col].value_counts().sort_index()
    report["class_distribution"] = {
        int(k): {"count": int(v), "name": LABEL_NAMES.get(int(k), "unknown")}
        for k, v in label_counts.items()
    }

    # --- Duplicates ---
    exact_dups = df[prompt_col].duplicated().sum()
    report["duplicate_prompts"] = int(exact_dups)
    if exact_dups > 0:
        report["warnings"].append(f"{exact_dups} duplicate prompt(s) found")

    # --- Conflicting labels ---
    if exact_dups > 0:
        conflict_groups = df.groupby(prompt_col)[label_col].nunique()
        conflicting = conflict_groups[conflict_groups > 1]
        report["conflicting_label_prompts"] = int(len(conflicting))
        if len(conflicting) > 0:
            report["warnings"].append(
                f"{len(conflicting)} prompt(s) have conflicting labels"
            )
    else:
        report["conflicting_label_prompts"] = 0

    # --- Prompt length stats ---
    lengths = df[prompt_col].astype(str).str.len()
    report["prompt_length_stats"] = {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(round(lengths.mean(), 1)),
        "median": float(lengths.median()),
        "std": float(round(lengths.std(), 1)),
    }

    # --- Missing value summary ---
    report["missing_values"] = {
        col: int(df[col].isnull().sum()) for col in df.columns
    }

    return report


def clean_dataset(
    df: pd.DataFrame,
    prompt_col: str = "prompt",
    label_col: str = "label",
) -> Tuple[pd.DataFrame, Dict]:
    """Clean dataset: handle missing values, convert types, remove invalid rows.
    
    Returns the cleaned DataFrame and a cleaning report.
    """
    cleaning_report: Dict = {
        "original_count": len(df),
        "removed_null_prompts": 0,
        "removed_empty_prompts": 0,
        "removed_invalid_labels": 0,
        "removed_null_labels": 0,
    }

    df = df.copy()

    # Remove null prompts
    null_mask = df[prompt_col].isnull()
    cleaning_report["removed_null_prompts"] = int(null_mask.sum())
    df = df[~null_mask]

    # Convert prompts to strings
    df[prompt_col] = df[prompt_col].astype(str)

    # Remove empty prompts (after stripping)
    empty_mask = df[prompt_col].str.strip().isin(["", "nan", "None"])
    cleaning_report["removed_empty_prompts"] = int(empty_mask.sum())
    df = df[~empty_mask]

    # Remove null labels
    null_label_mask = df[label_col].isnull()
    cleaning_report["removed_null_labels"] = int(null_label_mask.sum())
    df = df[~null_label_mask]

    # Convert labels to int
    df[label_col] = df[label_col].astype(int)

    # Remove invalid labels
    invalid_mask = ~df[label_col].isin(VALID_LABELS)
    cleaning_report["removed_invalid_labels"] = int(invalid_mask.sum())
    df = df[~invalid_mask]

    cleaning_report["final_count"] = len(df)
    cleaning_report["removed_total"] = (
        cleaning_report["original_count"] - cleaning_report["final_count"]
    )

    if cleaning_report["removed_total"] > 0:
        logger.warning(
            f"Removed {cleaning_report['removed_total']} rows during cleaning"
        )

    df = df.reset_index(drop=True)
    return df, cleaning_report


def print_validation_report(report: Dict) -> None:
    """Pretty-print the validation report."""
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Total records: {report['total_records']}")
    print(f"Columns: {report['columns_found']}")
    print()

    if report.get("class_distribution"):
        print("Class Distribution:")
        total = sum(v["count"] for v in report["class_distribution"].values())
        for label, info in sorted(report["class_distribution"].items()):
            pct = info["count"] / total * 100
            print(f"  {label} ({info['name']}): {info['count']:,} ({pct:.1f}%)")
        print()

    if report.get("prompt_length_stats"):
        stats = report["prompt_length_stats"]
        print("Prompt Length Statistics:")
        print(f"  Min: {stats['min']}, Max: {stats['max']}")
        print(f"  Mean: {stats['mean']}, Median: {stats['median']}")
        print(f"  Std: {stats['std']}")
        print()

    print(f"Duplicate prompts: {report.get('duplicate_prompts', 'N/A')}")
    print(f"Conflicting labels: {report.get('conflicting_label_prompts', 'N/A')}")
    print(f"Missing values: {report.get('missing_values', 'N/A')}")
    print()

    if report.get("errors"):
        print("ERRORS:")
        for e in report["errors"]:
            print(f"  [ERROR] {e}")
    else:
        print("[OK] No validation errors.")

    if report.get("warnings"):
        print("WARNINGS:")
        for w in report["warnings"]:
            print(f"  [WARN] {w}")
    else:
        print("[OK] No warnings.")

    print("=" * 60)
