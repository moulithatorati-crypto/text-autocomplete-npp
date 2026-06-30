"""
Beam search decoder for efficient sequence generation.
"""

import torch
from typing import List, Dict, Optional, Tuple
import heapq
from utils.logger import get_logger

logger = get_logger(__name__)


class BeamSearchDecoder:
    """
    Beam search decoder with various decoding strategies.
    """
    
    def __init__(
        self,
        beam_size: int = 4,
        max_length: int = 128,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        no_repeat_ngram_size: int = 2
    ):
        """
        Initialize beam search decoder.
        
        Args:
            beam_size: Number of beams
            max_length: Maximum sequence length
            length_penalty: Length normalization penalty
            temperature: Sampling temperature
            no_repeat_ngram_size: Size of n-grams to avoid repetition
        """
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.no_repeat_ngram_size = no_repeat_ngram_size
    
    @staticmethod
    def beam_search(
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        beam_size: int = 4,
        max_length: int = 128,
        length_penalty: float = 1.0,
        num_return_sequences: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform beam search decoding.
        
        Args:
            model: Language model with generate method
            input_ids: Input token IDs
            attention_mask: Attention mask
            beam_size: Number of beams
            max_length: Maximum length
            length_penalty: Length penalty coefficient
            num_return_sequences: Number of sequences to return
            **kwargs: Additional model arguments
            
        Returns:
            Generated token sequences
        """
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=beam_size,
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty,
            early_stopping=True,
            **kwargs
        )
        
        return outputs
    
    @staticmethod
    def top_k_sampling(
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        top_k: int = 50,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform top-k sampling.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum length
            top_k: Top-k parameter
            temperature: Temperature for sampling
            num_return_sequences: Number of sequences to return
            **kwargs: Additional arguments
            
        Returns:
            Generated token sequences
        """
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            **kwargs
        )
        
        return outputs
    
    @staticmethod
    def top_p_sampling(
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        top_p: float = 0.95,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Perform top-p (nucleus) sampling.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum length
            top_p: Top-p parameter (nucleus probability)
            temperature: Temperature for sampling
            num_return_sequences: Number of sequences to return
            **kwargs: Additional arguments
            
        Returns:
            Generated token sequences
        """
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            **kwargs
        )
        
        return outputs
