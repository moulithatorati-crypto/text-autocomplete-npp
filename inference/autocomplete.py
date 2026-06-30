"""
Autocomplete engine for interactive text generation.
"""

import torch
from typing import List, Dict, Optional, Any
from models.t5_wrapper import T5Wrapper
from inference.beam_search import BeamSearchDecoder
from utils.logger import get_logger

logger = get_logger(__name__)


class AutocompleteEngine:
    """
    Interactive text autocomplete engine using trained T5 model.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        decoding_strategy: str = "beam_search"
    ):
        """
        Initialize autocomplete engine.
        
        Args:
            model_path: Path to trained model
            device: Device to run on
            decoding_strategy: Decoding strategy ("beam_search", "top_k", "top_p")
        """
        self.model_path = model_path
        self.device = device
        self.decoding_strategy = decoding_strategy
        
        # Load model
        self.model = T5Wrapper(model_name=model_path, device=device)
        self.decoder = BeamSearchDecoder()
        
        logger.info(f"✓ Autocomplete engine initialized with {decoding_strategy}")
    
    def complete(
        self,
        partial_text: str,
        num_return_sequences: int = 5,
        max_length: int = 128,
        beam_size: int = 5,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """
        Generate completions for partial text.
        
        Args:
            partial_text: Partial text to complete
            num_return_sequences: Number of completions to generate
            max_length: Maximum length of completions
            beam_size: Beam size for beam search
            temperature: Temperature for sampling
            top_k: Top-k for sampling
            top_p: Top-p for sampling
            **kwargs: Additional arguments
            
        Returns:
            List of completion strings
        """
        # Prepare input
        input_text = f"generate next phrase: {partial_text}"
        
        inputs = self.model.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        if self.decoding_strategy == "beam_search":
            output_ids = BeamSearchDecoder.beam_search(
                self.model.model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                beam_size=beam_size,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                **kwargs
            )
        
        elif self.decoding_strategy == "top_k":
            output_ids = BeamSearchDecoder.top_k_sampling(
                self.model.model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                **kwargs
            )
        
        elif self.decoding_strategy == "top_p":
            output_ids = BeamSearchDecoder.top_p_sampling(
                self.model.model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown decoding strategy: {self.decoding_strategy}")
        
        # Decode
        completions = self.model.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )
        
        return completions
    
    def complete_interactive(
        self,
        partial_text: str,
        num_candidates: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Interactive completion with metadata.
        
        Args:
            partial_text: Partial text to complete
            num_candidates: Number of candidates to return
            **kwargs: Additional generation arguments
            
        Returns:
            Dictionary with completions and metadata
        """
        completions = self.complete(
            partial_text,
            num_return_sequences=num_candidates,
            **kwargs
        )
        
        return {
            "input": partial_text,
            "completions": completions,
            "num_candidates": len(completions),
            "model": self.model_path,
            "strategy": self.decoding_strategy,
        }
