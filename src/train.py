"""
Training Module
===============
TF-IDF + Logistic Regression training pipeline for prompt injection detection.
Supports word-level, character-level, and combined feature experiments.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from numpy import float32
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from data_validation import (
    LABEL_NAMES,
    clean_dataset,
    load_dataset,
    print_validation_report,
    validate_schema,
)
from evaluate import (
    evaluate_model,
    export_confusion_matrix_csv,
    generate_classification_report_dict,
    plot_confusion_matrix,
    compare_experiments,
)
from preprocessing import preprocess_dataframe

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def create_stratified_splits(
    df: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Create reproducible stratified train/val/test splits.
    
    Prevents data leakage by ensuring no duplicate prompts cross splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    split_info: Dict[str, Any] = {
        "random_seed": random_seed,
        "requested_ratios": {
            "train": train_ratio, "val": val_ratio, "test": test_ratio,
        },
    }

    n = len(df)

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    df_train, df_valtest = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=random_seed,
        stratify=df[label_col],
    )

    # Second split: val vs test (from the val+test portion)
    relative_test_ratio = test_ratio / val_test_ratio
    df_val, df_test = train_test_split(
        df_valtest,
        test_size=relative_test_ratio,
        random_state=random_seed,
        stratify=df_valtest[label_col],
    )

    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Record actual split sizes
    split_info["actual_sizes"] = {
        "train": len(df_train),
        "val": len(df_val),
        "test": len(df_test),
        "total": n,
    }
    split_info["actual_ratios"] = {
        "train": round(len(df_train) / n, 4),
        "val": round(len(df_val) / n, 4),
        "test": round(len(df_test) / n, 4),
    }

    # Record per-split class distributions
    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        dist = split_df[label_col].value_counts().sort_index().to_dict()
        split_info[f"{name}_distribution"] = {int(k): int(v) for k, v in dist.items()}

    logger.info(
        f"Split sizes — Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}"
    )
    return df_train, df_val, df_test, split_info


def build_word_tfidf(config: Dict) -> TfidfVectorizer:
    """Build word-level TF-IDF vectorizer from config."""
    tfidf_cfg = config.get("tfidf_word", {})
    ngram = tuple(tfidf_cfg.get("ngram_range", [1, 2]))
    return TfidfVectorizer(
        analyzer=tfidf_cfg.get("analyzer", "word"),
        ngram_range=ngram,
        max_features=tfidf_cfg.get("max_features", 100000),
        sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        min_df=tfidf_cfg.get("min_df", 1),
        max_df=tfidf_cfg.get("max_df", 1.0),
        strip_accents=tfidf_cfg.get("strip_accents", None),
        lowercase=tfidf_cfg.get("lowercase", False),
        token_pattern=tfidf_cfg.get("token_pattern", r"(?u)\S+"),
        dtype=float32,
    )


def build_char_tfidf(config: Dict) -> TfidfVectorizer:
    """Build character-level TF-IDF vectorizer from config."""
    tfidf_cfg = config.get("tfidf_char", {})
    ngram = tuple(tfidf_cfg.get("ngram_range", [2, 5]))
    return TfidfVectorizer(
        analyzer=tfidf_cfg.get("analyzer", "char_wb"),
        ngram_range=ngram,
        max_features=tfidf_cfg.get("max_features", 100000),
        sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        min_df=tfidf_cfg.get("min_df", 2),
        max_df=tfidf_cfg.get("max_df", 1.0),
        strip_accents=tfidf_cfg.get("strip_accents", None),
        lowercase=tfidf_cfg.get("lowercase", False),
        dtype=float32,
    )


def build_logistic_regression(config: Dict) -> LogisticRegression:
    """Build logistic regression classifier from config."""
    lr_cfg = config.get("logistic_regression", {})
    return LogisticRegression(
        max_iter=lr_cfg.get("max_iter", 2000),
        class_weight=lr_cfg.get("class_weight", "balanced"),
        random_state=lr_cfg.get("random_state", 42),
        solver=lr_cfg.get("solver", "lbfgs"),
        C=lr_cfg.get("C", 1.0),
    )


def build_pipeline_a(config: Dict) -> Pipeline:
    """Experiment A: Word-level TF-IDF + Logistic Regression."""
    return Pipeline([
        ("tfidf", build_word_tfidf(config)),
        ("clf", build_logistic_regression(config)),
    ])


def build_pipeline_b(config: Dict) -> Pipeline:
    """Experiment B: Character-level TF-IDF + Logistic Regression."""
    return Pipeline([
        ("tfidf", build_char_tfidf(config)),
        ("clf", build_logistic_regression(config)),
    ])


def build_pipeline_c(config: Dict) -> Pipeline:
    """Experiment C: Combined word + char TF-IDF + Logistic Regression."""
    feature_union = FeatureUnion([
        ("word_tfidf", build_word_tfidf(config)),
        ("char_tfidf", build_char_tfidf(config)),
    ])
    return Pipeline([
        ("features", feature_union),
        ("clf", build_logistic_regression(config)),
    ])


def train_and_evaluate_experiment(
    name: str,
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    X_val: pd.Series,
    y_val: pd.Series,
    label_names: Dict[int, str],
) -> Dict:
    """Train a pipeline and evaluate on validation set. Return metrics dict."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    # Train
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.2f}s")

    # Evaluate on validation set
    metrics = evaluate_model(
        pipeline, X_val, y_val, label_names, dataset_name="Validation"
    )
    metrics["experiment_name"] = name
    metrics["train_time_seconds"] = round(train_time, 3)

    return metrics


def save_artifacts(
    pipeline: Pipeline,
    config: Dict,
    split_info: Dict,
    validation_report: Dict,
    label_names: Dict[int, str],
    experiment_name: str,
    artifacts_dir: str,
    cleaning_report: Dict,
    dataset_stats: Dict,
) -> None:
    """Save all training artifacts to disk."""
    art_dir = Path(artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = art_dir / "model.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    model_size = model_path.stat().st_size
    print(f"Model saved: {model_path} ({model_size / 1e6:.2f} MB)")

    # Save label mapping
    label_path = art_dir / "label_mapping.json"
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in label_names.items()}, f, indent=2)

    # Save config snapshot
    config_path = art_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    # Save split info
    split_path = art_dir / "split_info.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # Save dataset stats
    stats_path = art_dir / "dataset_stats.json"
    full_stats = {
        "validation_report": dataset_stats,
        "cleaning_report": cleaning_report,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(full_stats, f, indent=2, default=str)

    # Save dependency versions
    import sklearn
    versions = {
        "python": sys.version,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }
    versions_path = art_dir / "versions.json"
    with open(versions_path, "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2)

    print(f"All artifacts saved to {art_dir}")


def main(config_path: str = None) -> None:
    """Main training pipeline entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Determine config path
    if config_path is None:
        config_path = str(
            Path(__file__).parent.parent / "configs" / "baseline.yaml"
        )
    config = load_config(config_path)

    project_root = Path(__file__).parent.parent
    artifacts_dir = str(project_root / config.get("artifacts_dir", "artifacts"))
    reports_dir = str(project_root / config.get("reports_dir", "reports"))
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    label_names = {int(k): v for k, v in config["labels"].items()}

    # ===== STEP 1: Load and validate dataset =====
    print("\n" + "=" * 60)
    print("STEP 1: Data Loading & Validation")
    print("=" * 60)

    df = load_dataset(config["dataset"]["path"])
    prompt_col = config["dataset"]["prompt_column"]
    label_col = config["dataset"]["label_column"]

    validation_report = validate_schema(df, prompt_col, label_col)
    print_validation_report(validation_report)

    if validation_report.get("errors"):
        print("FATAL: Validation errors found. Aborting.")
        sys.exit(1)

    # Clean
    df, cleaning_report = clean_dataset(df, prompt_col, label_col)
    print(f"After cleaning: {len(df)} records")
    if cleaning_report["removed_total"] > 0:
        print(f"  Removed: {cleaning_report['removed_total']} rows")

    # ===== STEP 2: Preprocessing =====
    print("\n" + "=" * 60)
    print("STEP 2: Text Preprocessing")
    print("=" * 60)

    prep_cfg = config.get("preprocessing", {})
    df = preprocess_dataframe(
        df,
        prompt_col=prompt_col,
        processed_col="prompt_processed",
        do_normalize_unicode=prep_cfg.get("normalize_unicode", True),
        do_normalize_whitespace=prep_cfg.get("normalize_whitespace", True),
        do_lowercase=prep_cfg.get("lowercase", False),
    )
    print(f"Preprocessing complete. Using 'prompt_processed' for training.")

    # ===== STEP 3: Train/Val/Test Split =====
    print("\n" + "=" * 60)
    print("STEP 3: Stratified Split")
    print("=" * 60)

    split_cfg = config.get("split", {})
    df_train, df_val, df_test, split_info = create_stratified_splits(
        df,
        label_col=label_col,
        train_ratio=split_cfg.get("train_ratio", 0.80),
        val_ratio=split_cfg.get("val_ratio", 0.10),
        test_ratio=split_cfg.get("test_ratio", 0.10),
        random_seed=split_cfg.get("random_seed", 42),
    )

    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    print(f"Train distribution: {df_train[label_col].value_counts().sort_index().to_dict()}")
    print(f"Val distribution:   {df_val[label_col].value_counts().sort_index().to_dict()}")
    print(f"Test distribution:  {df_test[label_col].value_counts().sort_index().to_dict()}")

    X_train = df_train["prompt_processed"]
    y_train = df_train[label_col]
    X_val = df_val["prompt_processed"]
    y_val = df_val[label_col]
    X_test = df_test["prompt_processed"]
    y_test = df_test[label_col]

    # ===== STEP 4: Run Experiments =====
    print("\n" + "=" * 60)
    print("STEP 4: Model Experiments")
    print("=" * 60)

    experiments: List[Tuple[str, Pipeline]] = [
        ("A_word_tfidf", build_pipeline_a(config)),
        ("B_char_tfidf", build_pipeline_b(config)),
        ("C_combined_tfidf", build_pipeline_c(config)),
    ]

    all_results: List[Dict] = []
    trained_pipelines: Dict[str, Pipeline] = {}

    for exp_name, pipeline in experiments:
        try:
            result = train_and_evaluate_experiment(
                exp_name, pipeline,
                X_train, y_train,
                X_val, y_val,
                label_names,
            )
            all_results.append(result)
            trained_pipelines[exp_name] = pipeline
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            print(f"  [FAIL] Experiment {exp_name} failed: {e}")

    # ===== STEP 5: Compare experiments & select best =====
    print("\n" + "=" * 60)
    print("STEP 5: Experiment Comparison & Model Selection")
    print("=" * 60)

    if not all_results:
        print("FATAL: No experiments completed successfully.")
        sys.exit(1)

    comparison_df = compare_experiments(all_results)
    print("\nExperiment Comparison:")
    print(comparison_df.to_string(index=False))

    # Save comparison
    comparison_path = Path(reports_dir) / "experiment_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved: {comparison_path}")

    # Select best model by weighted F1 (balances precision/recall across classes)
    best_idx = comparison_df["weighted_f1"].astype(float).idxmax()
    best_name = comparison_df.loc[best_idx, "experiment"]
    best_pipeline = trained_pipelines[best_name]
    print(f"\n>>> Best model: {best_name} (weighted F1: {comparison_df.loc[best_idx, 'weighted_f1']})")

    # ===== STEP 6: Final Test Evaluation =====
    print("\n" + "=" * 60)
    print("STEP 6: Final Test Set Evaluation")
    print("=" * 60)

    test_metrics = evaluate_model(
        best_pipeline, X_test, y_test, label_names, dataset_name="Test"
    )

    # Save test classification report
    test_report_path = Path(reports_dir) / "test_classification_report.json"
    with open(test_report_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, default=str)
    print(f"Saved: {test_report_path}")

    # Save validation report for best model
    best_val_result = [r for r in all_results if r["experiment_name"] == best_name][0]
    val_report_path = Path(reports_dir) / "val_classification_report.json"
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(best_val_result, f, indent=2, default=str)
    print(f"Saved: {val_report_path}")

    # Save confusion matrices
    export_confusion_matrix_csv(
        best_pipeline, X_val, y_val, label_names,
        str(Path(reports_dir) / "confusion_matrix_val.csv"),
    )
    export_confusion_matrix_csv(
        best_pipeline, X_test, y_test, label_names,
        str(Path(reports_dir) / "confusion_matrix_test.csv"),
    )

    # Plot confusion matrices
    plot_confusion_matrix(
        best_pipeline, X_val, y_val, label_names,
        str(Path(reports_dir) / "confusion_matrix_val.png"),
        title=f"Validation Confusion Matrix ({best_name})",
    )
    plot_confusion_matrix(
        best_pipeline, X_test, y_test, label_names,
        str(Path(reports_dir) / "confusion_matrix_test.png"),
        title=f"Test Confusion Matrix ({best_name})",
    )

    # Save all metrics
    all_metrics = {
        "best_experiment": best_name,
        "validation_metrics": best_val_result,
        "test_metrics": test_metrics,
    }
    metrics_path = Path(reports_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"Saved: {metrics_path}")

    # ===== STEP 7: Save Artifacts =====
    print("\n" + "=" * 60)
    print("STEP 7: Saving Artifacts")
    print("=" * 60)

    save_artifacts(
        pipeline=best_pipeline,
        config=config,
        split_info=split_info,
        validation_report=best_val_result,
        label_names=label_names,
        experiment_name=best_name,
        artifacts_dir=artifacts_dir,
        cleaning_report=cleaning_report,
        dataset_stats=validation_report,
    )

    # ===== STEP 8: Threshold Calibration =====
    print("\n" + "=" * 60)
    print("STEP 8: Security Threshold Calibration")
    print("=" * 60)

    thresh_cfg = config.get("security_thresholds", {})
    if thresh_cfg.get("enabled", False):
        calibrate_thresholds(
            best_pipeline, X_val, y_val, label_names, reports_dir
        )

    print("\n" + "=" * 60)
    print("[OK] TRAINING PIPELINE COMPLETE")
    print("=" * 60)


def calibrate_thresholds(
    pipeline: Pipeline,
    X_val: pd.Series,
    y_val: pd.Series,
    label_names: Dict[int, str],
    reports_dir: str,
) -> Dict:
    """Calibrate security decision thresholds on validation set."""
    probas = pipeline.predict_proba(X_val)
    classes_list = list(pipeline.classes_)

    # Robust index mapping: class 0 is benign, any non-zero class is attack
    benign_idx = classes_list.index(0) if 0 in classes_list else None
    attack_indices = [j for j, c in enumerate(classes_list) if c != 0]

    if benign_idx is not None:
        p_benign_all = probas[:, benign_idx]
    else:
        p_benign_all = np.zeros(len(probas))

    if attack_indices:
        p_attack_all = np.max(probas[:, attack_indices], axis=1)
    else:
        p_attack_all = np.zeros(len(probas))

    results = []

    # Evaluate a grid of thresholds
    benign_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    attack_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]

    print(f"{'benign_th':>10} {'attack_th':>10} {'allow%':>8} {'block%':>8} {'review%':>8} {'FN_rate':>8} {'FP_rate':>8}")
    print("-" * 70)

    for benign_th in benign_thresholds:
        for attack_th in attack_thresholds:
            allow_mask = (p_benign_all >= benign_th) & (p_attack_all < (1.0 - benign_th))
            block_mask = (~allow_mask) & (p_attack_all >= attack_th)
            review_mask = (~allow_mask) & (~block_mask)

            decisions = np.empty(len(probas), dtype=object)
            decisions[allow_mask] = "allow"
            decisions[block_mask] = "block"
            decisions[review_mask] = "review"
            true_labels = y_val.values

            n = len(decisions)
            allow_pct = (decisions == "allow").sum() / n * 100
            block_pct = (decisions == "block").sum() / n * 100
            review_pct = (decisions == "review").sum() / n * 100

            # False negatives: attacks that were allowed
            attack_mask = true_labels != 0
            fn_rate = 0.0
            if attack_mask.sum() > 0:
                fn_rate = ((decisions[attack_mask] == "allow").sum()
                           / attack_mask.sum() * 100)

            # False positives: benign that were blocked
            benign_mask = true_labels == 0
            fp_rate = 0.0
            if benign_mask.sum() > 0:
                fp_rate = ((decisions[benign_mask] == "block").sum()
                           / benign_mask.sum() * 100)

            print(
                f"{benign_th:>10.2f} {attack_th:>10.2f} "
                f"{allow_pct:>7.1f}% {block_pct:>7.1f}% {review_pct:>7.1f}% "
                f"{fn_rate:>7.2f}% {fp_rate:>7.2f}%"
            )

            results.append({
                "benign_threshold": benign_th,
                "attack_threshold": attack_th,
                "allow_pct": round(allow_pct, 2),
                "block_pct": round(block_pct, 2),
                "review_pct": round(review_pct, 2),
                "false_negative_rate": round(fn_rate, 3),
                "false_positive_rate": round(fp_rate, 3),
            })

    # Save calibration results
    cal_path = Path(reports_dir) / "threshold_calibration.json"
    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {cal_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train prompt injection classifier")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: configs/baseline.yaml)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
