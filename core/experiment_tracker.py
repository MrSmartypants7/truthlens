"""
Experiment Tracker — logs benchmark results for comparing model performance.

WHY TRACK EXPERIMENTS: When you change the embedding model, chunk size, or
verification prompt, you need to know if things got better or worse. This
module logs every benchmark run with all parameters, so you can:
  - Compare "did prompt v2 improve accuracy?"
  - Track cross-model consistency (GPT-4 vs Claude vs Llama)
  - Detect regressions over time

This is the "prompt benchmarking pipeline" mentioned in the resume bullet.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import settings
from app.logging_config import get_logger
from models.schemas import BenchmarkResult, ClaimVerification, VerificationStatus

logger = get_logger("experiment_tracker")


class ExperimentTracker:
    """Tracks and persists benchmark experiment results."""

    def __init__(self, experiment_dir: str = None):
        self.experiment_dir = Path(experiment_dir or settings.experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_verifications(
        self,
        verifications: list[ClaimVerification],
        ground_truth: list[dict],
        model_name: str,
        experiment_id: str = None,
    ) -> BenchmarkResult:
        """
        Compare verification results against ground truth labels.

        Args:
            verifications: List of ClaimVerification from the pipeline
            ground_truth: List of {"claim": str, "expected_status": str}
            model_name: Which LLM was being tested
            experiment_id: Optional ID (auto-generated if not provided)

        Returns:
            BenchmarkResult with accuracy, precision, recall metrics

        METRICS EXPLAINED:
        - Accuracy: % of claims where our verdict matches ground truth
        - Precision: Of claims we said were "supported," what % actually are?
        - Recall: Of claims that ARE supported, what % did we correctly identify?
        """
        experiment_id = experiment_id or f"exp-{int(time.time())}"

        if len(verifications) != len(ground_truth):
            logger.warning(
                "length_mismatch",
                verifications=len(verifications),
                ground_truth=len(ground_truth),
            )

        # Match predictions to ground truth
        correct = 0
        true_positives = 0  # Correctly identified as supported
        false_positives = 0  # Said supported, but actually not
        false_negatives = 0  # Said not supported, but actually is

        total = min(len(verifications), len(ground_truth))
        details_list = []

        for i in range(total):
            predicted = verifications[i].status.value
            expected = ground_truth[i]["expected_status"]

            is_correct = predicted == expected
            if is_correct:
                correct += 1

            # For precision/recall, treat "supported" as the positive class
            if predicted == "supported" and expected == "supported":
                true_positives += 1
            elif predicted == "supported" and expected != "supported":
                false_positives += 1
            elif predicted != "supported" and expected == "supported":
                false_negatives += 1

            details_list.append({
                "claim": verifications[i].claim.text,
                "predicted": predicted,
                "expected": expected,
                "correct": is_correct,
                "confidence": verifications[i].confidence,
            })

        # Calculate metrics (with safety for division by zero)
        accuracy = correct / total if total > 0 else 0.0
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

        # Average verification time
        avg_time = 0.0  # Would be populated from actual timing data

        result = BenchmarkResult(
            experiment_id=experiment_id,
            model_name=model_name,
            total_claims=total,
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            avg_verification_time_ms=avg_time,
            details={
                "correct": correct,
                "total": total,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "per_claim": details_list,
            },
        )

        # Persist result
        self._save_result(result)

        logger.info(
            "benchmark_complete",
            experiment_id=experiment_id,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
        )

        return result

    def _save_result(self, result: BenchmarkResult):
        """Save experiment result to JSON file."""
        filepath = self.experiment_dir / f"{result.experiment_id}.json"
        with open(filepath, "w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

    def list_experiments(self) -> list[dict]:
        """List all saved experiment results."""
        results = []
        for filepath in sorted(self.experiment_dir.glob("*.json")):
            with open(filepath) as f:
                results.append(json.load(f))
        return results

    def compare_models(self, experiment_ids: list[str] = None) -> dict:
        """
        Compare benchmark results across models/experiments.

        Returns a summary table useful for deciding which model/config is best.
        """
        experiments = self.list_experiments()

        if experiment_ids:
            experiments = [
                e for e in experiments if e["experiment_id"] in experiment_ids
            ]

        if not experiments:
            return {"message": "No experiments found"}

        summary = {
            "experiments": len(experiments),
            "comparison": [],
        }

        for exp in experiments:
            summary["comparison"].append({
                "experiment_id": exp["experiment_id"],
                "model": exp["model_name"],
                "accuracy": exp["accuracy"],
                "precision": exp["precision"],
                "recall": exp["recall"],
                "total_claims": exp["total_claims"],
            })

        # Sort by accuracy descending
        summary["comparison"].sort(key=lambda x: x["accuracy"], reverse=True)

        return summary
