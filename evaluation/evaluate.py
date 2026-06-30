"""
Evaluation module for comprehensive model assessment.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from evaluation.metrics import MetricsCalculator
from utils.logger import get_logger
from utils.helpers import save_json

logger = get_logger(__name__)


class Evaluator:
    """
    Comprehensive evaluator for text generation models.
    """
    
    def __init__(
        self,
        output_dir: str = "outputs/evaluation",
        metrics: List[str] = ["bleu", "rouge", "exact_match", "partial_match"]
    ):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
            metrics: List of metrics to compute
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics
    
    def evaluate_batch(
        self,
        references: List[str],
        hypotheses: List[str],
        save_predictions: bool = True,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions against references.
        
        Args:
            references: List of reference texts
            hypotheses: List of hypothesis texts
            save_predictions: Whether to save predictions
            output_file: Output file path
            
        Returns:
            Dictionary with aggregated metrics
        """
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses must have same length")
        
        logger.info(f"Evaluating {len(references)} predictions...")
        
        individual_scores = []
        aggregated_scores = {metric: [] for metric in self.metrics}
        
        for ref, hyp in tqdm(zip(references, hypotheses), total=len(references)):
            scores = MetricsCalculator.calculate_all_metrics(
                ref, hyp, metrics=self.metrics
            )
            individual_scores.append(scores)
            
            for metric in self.metrics:
                if metric in scores:
                    aggregated_scores[metric].append(scores[metric])
        
        # Calculate means
        final_scores = {}
        for metric, values in aggregated_scores.items():
            if values:
                final_scores[f"{metric}_mean"] = np.mean(values)
                final_scores[f"{metric}_std"] = np.std(values)
                final_scores[f"{metric}_min"] = np.min(values)
                final_scores[f"{metric}_max"] = np.max(values)
        
        # Save if requested
        if save_predictions:
            if output_file is None:
                output_file = self.output_dir / "predictions.jsonl"
            
            with open(output_file, 'w') as f:
                for ref, hyp, scores in zip(references, hypotheses, individual_scores):
                    record = {
                        "reference": ref,
                        "hypothesis": hyp,
                        "scores": scores
                    }
                    f.write(json.dumps(record) + '\n')
        
        logger.info("Evaluation completed")
        return final_scores
    
    def generate_report(
        self,
        final_scores: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate evaluation report.
        
        Args:
            final_scores: Dictionary of final scores
            output_file: Output file path
            
        Returns:
            Report string
        """
        report = []
        report.append("\n" + "="*60)
        report.append("EVALUATION REPORT")
        report.append("="*60)
        
        for metric, score in sorted(final_scores.items()):
            if isinstance(score, float):
                report.append(f"{metric:.<40} {score:.4f}")
            else:
                report.append(f"{metric:.<40} {score}")
        
        report.append("="*60 + "\n")
        
        report_str = "\n".join(report)
        
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report_str)
            logger.info(f"Report saved to {output_file}")
        
        return report_str
