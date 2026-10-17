"""
Prediction / Inference Module
=============================
Reusable prediction interface for the trained prompt injection classifier.
Supports single prompt, batch inference, and security decision layer.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import preprocess_text


class PromptClassifier:
    """Inference wrapper for the trained TF-IDF + LR prompt classifier."""

    def __init__(
        self,
        model_path: str,
        label_mapping_path: Optional[str] = None,
        security_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            model_path: Path to the serialized model pipeline (.joblib).
            label_mapping_path: Path to label_mapping.json.
            security_thresholds: Optional dict with 'benign_allow_min' and 'attack_block_min'.
        """
        self.pipeline: Pipeline = joblib.load(model_path)
        self.classes = self.pipeline.classes_

        # Load label mapping
        if label_mapping_path and Path(label_mapping_path).exists():
            with open(label_mapping_path, "r", encoding="utf-8") as f:
                self.label_names = {int(k): v for k, v in json.load(f).items()}
        else:
            self.label_names = {0: "benign", 1: "promptinjection", 2: "jailbreak"}

        self.security_thresholds = security_thresholds

    def predict_single(
        self,
        prompt: str,
        apply_security: bool = True,
    ) -> Dict[str, Any]:
        """Classify a single prompt.
        
        Returns:
            Dict with predicted label, probabilities, security decision, and latency.
        """
        # Preprocess
        processed = preprocess_text(prompt)

        # Inference
        t0 = time.perf_counter()
        pred = self.pipeline.predict([processed])[0]
        proba = self.pipeline.predict_proba([processed])[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        # Build probability dict
        prob_dict = {}
        for i, cls in enumerate(self.classes):
            prob_dict[self.label_names.get(int(cls), str(cls))] = round(float(proba[i]), 6)

        result: Dict[str, Any] = {
            "predicted_label_id": int(pred),
            "predicted_label_name": self.label_names.get(int(pred), str(pred)),
            "probabilities": prob_dict,
            "inference_latency_ms": round(latency_ms, 3),
        }

        # Security decision
        if apply_security and self.security_thresholds:
            result["security_decision"] = self._security_decision(proba)

        return result

    def predict_batch(
        self,
        prompts: List[str],
        apply_security: bool = True,
    ) -> List[Dict[str, Any]]:
        """Classify a batch of prompts.
        
        Returns:
            List of result dicts.
        """
        # Preprocess all
        processed = [preprocess_text(p) for p in prompts]

        # Batch inference
        t0 = time.perf_counter()
        preds = self.pipeline.predict(processed)
        probas = self.pipeline.predict_proba(processed)
        latency_ms = (time.perf_counter() - t0) * 1000

        results = []
        for i, (prompt, pred, proba) in enumerate(zip(prompts, preds, probas)):
            prob_dict = {}
            for j, cls in enumerate(self.classes):
                prob_dict[self.label_names.get(int(cls), str(cls))] = round(float(proba[j]), 6)

            result: Dict[str, Any] = {
                "prompt": prompt[:200],  # Truncate for display
                "predicted_label_id": int(pred),
                "predicted_label_name": self.label_names.get(int(pred), str(pred)),
                "probabilities": prob_dict,
                "inference_latency_ms": round(latency_ms / len(prompts), 3),
            }

            if apply_security and self.security_thresholds:
                result["security_decision"] = self._security_decision(proba)

            results.append(result)

        return results

    def _security_decision(self, proba: np.ndarray) -> str:
        """Apply security threshold logic to determine allow/block/review."""
        benign_th = self.security_thresholds.get("benign_allow_min", 0.70)
        attack_th = self.security_thresholds.get("attack_block_min", 0.60)

        classes_list = list(self.classes)
        benign_idx = classes_list.index(0) if 0 in classes_list else None
        attack_indices = [j for j, c in enumerate(classes_list) if c != 0]

        p_benign = float(proba[benign_idx]) if benign_idx is not None else 0.0
        p_attack = float(max([proba[j] for j in attack_indices])) if attack_indices else 0.0

        if p_benign >= benign_th and p_attack < (1 - benign_th):
            return "allow"
        elif p_attack >= attack_th:
            return "block"
        else:
            return "review"


def main():
    """CLI entry point for inference."""
    parser = argparse.ArgumentParser(
        description="Classify prompts for injection/jailbreak detection"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model.joblib",
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Path to label_mapping.json",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Single prompt to classify",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="File with prompts (one per line) for batch classification",
    )
    parser.add_argument(
        "--security", action="store_true",
        help="Enable security decision layer",
    )
    parser.add_argument(
        "--benign-threshold", type=float, default=0.70,
        help="Minimum benign probability to allow (default: 0.70)",
    )
    parser.add_argument(
        "--attack-threshold", type=float, default=0.60,
        help="Minimum attack probability to block (default: 0.60)",
    )

    args = parser.parse_args()

    # Resolve default paths
    project_root = Path(__file__).parent.parent
    model_path = args.model or str(project_root / "artifacts" / "model.joblib")
    labels_path = args.labels or str(project_root / "artifacts" / "label_mapping.json")

    security_thresholds = None
    if args.security:
        security_thresholds = {
            "benign_allow_min": args.benign_threshold,
            "attack_block_min": args.attack_threshold,
        }

    classifier = PromptClassifier(
        model_path=model_path,
        label_mapping_path=labels_path,
        security_thresholds=security_thresholds,
    )

    if args.prompt:
        result = classifier.predict_single(args.prompt, apply_security=args.security)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        results = classifier.predict_batch(prompts, apply_security=args.security)
        for r in results:
            print(json.dumps(r, ensure_ascii=False))

    else:
        # Interactive mode
        print("Prompt Injection Classifier — Interactive Mode")
        print("Type a prompt and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("Prompt> ").strip()
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt:
                    continue
                result = classifier.predict_single(prompt, apply_security=args.security)
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print()
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    main()
