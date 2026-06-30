"""
Metrics calculation module for evaluating text generation quality.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
import math
from utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """
    Calculate various metrics for text generation evaluation.
    Supports BLEU, METEOR, ROUGE, and custom metrics.
    """
    
    @staticmethod
    def bleu_score(
        reference: str,
        hypothesis: str,
        max_n: int = 4,
        smooth: bool = False
    ) -> Dict[str, float]:
        """
        Calculate BLEU score.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            max_n: Maximum n-gram size
            smooth: Whether to apply smoothing
            
        Returns:
            Dictionary with individual n-gram scores and overall BLEU
        """
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        scores = {}
        weights = [1.0 / max_n] * max_n
        
        for n in range(1, max_n + 1):
            # Get n-grams
            ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
            hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])
            
            # Calculate precision
            matches = sum((hyp_ngrams & ref_ngrams).values())
            total = sum(hyp_ngrams.values())
            
            if smooth:
                precision = (matches + 1) / (total + 1) if total > 0 else 0
            else:
                precision = matches / total if total > 0 else 0
            
            scores[f"bleu_{n}"] = precision
        
        # Calculate geometric mean
        log_scores = [weights[i] * math.log(scores[f"bleu_{i+1}"] + 1e-16) for i in range(max_n)]
        bleu = math.exp(sum(log_scores))
        
        scores["bleu"] = bleu
        return scores
    
    @staticmethod
    def rouge_score(
        reference: str,
        hypothesis: str,
        rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores (simplified implementation).
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            rouge_types: Types of ROUGE to calculate
            
        Returns:
            Dictionary with ROUGE scores
        """
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        scores = {}
        
        for rouge_type in rouge_types:
            if rouge_type == "rouge1":
                n = 1
            elif rouge_type == "rouge2":
                n = 2
            elif rouge_type == "rougeL":
                # LCS-based
                lcs_len = MetricsCalculator._lcs_length(ref_tokens, hyp_tokens)
                recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0
                precision = lcs_len / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall + 1e-16)
                scores[rouge_type] = f1
                continue
            else:
                continue
            
            ref_ngrams = set([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
            hyp_ngrams = set([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])
            
            overlap = len(ref_ngrams & hyp_ngrams)
            recall = overlap / len(ref_ngrams) if len(ref_ngrams) > 0 else 0
            precision = overlap / len(hyp_ngrams) if len(hyp_ngrams) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall + 1e-16)
            
            scores[rouge_type] = f1
        
        return scores
    
    @staticmethod
    def _lcs_length(ref: List[str], hyp: List[str]) -> int:
        """
        Calculate Longest Common Subsequence length.
        
        Args:
            ref: Reference tokens
            hyp: Hypothesis tokens
            
        Returns:
            LCS length
        """
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    @staticmethod
    def exact_match(reference: str, hypothesis: str) -> float:
        """
        Calculate exact match score (0 or 1).
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return 1.0 if reference.lower().strip() == hypothesis.lower().strip() else 0.0
    
    @staticmethod
    def partial_match(reference: str, hypothesis: str) -> float:
        """
        Calculate partial match score based on token overlap.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            
        Returns:
            Score between 0 and 1
        """
        ref_tokens = set(reference.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        
        if not ref_tokens and not hyp_tokens:
            return 1.0
        if not ref_tokens or not hyp_tokens:
            return 0.0
        
        overlap = len(ref_tokens & hyp_tokens)
        union = len(ref_tokens | hyp_tokens)
        
        return overlap / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_all_metrics(
        reference: str,
        hypothesis: str,
        metrics: List[str] = ["bleu", "rouge", "exact_match", "partial_match"]
    ) -> Dict[str, Any]:
        """
        Calculate all specified metrics.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary with all metric scores
        """
        results = {}
        
        try:
            if "bleu" in metrics:
                results.update(MetricsCalculator.bleu_score(reference, hypothesis))
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
        
        try:
            if "rouge" in metrics:
                results.update(MetricsCalculator.rouge_score(reference, hypothesis))
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
        
        try:
            if "exact_match" in metrics:
                results["exact_match"] = MetricsCalculator.exact_match(reference, hypothesis)
        except Exception as e:
            logger.warning(f"Exact match calculation failed: {e}")
        
        try:
            if "partial_match" in metrics:
                results["partial_match"] = MetricsCalculator.partial_match(reference, hypothesis)
        except Exception as e:
            logger.warning(f"Partial match calculation failed: {e}")
        
        return results
