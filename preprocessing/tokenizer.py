"""
Tokenizer utilities for NPP dataset.
"""

from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)


class NPPTokenizer:
    """
    Wrapper around HuggingFace tokenizers for NPP dataset.
    """
    
    def __init__(
        self,
        model_name: str = "t5-base",
        max_source_length: int = 512,
        max_target_length: int = 128
    ):
        """
        Initialize tokenizer.
        
        Args:
            model_name: Model name for tokenizer
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
        """
        self.model_name = model_name
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"✓ Tokenizer loaded: {model_name}")
    
    def tokenize(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, Any]:
        """
        Tokenize a single text.
        
        Args:
            text: Text to tokenize
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Return tensor type ("pt" for PyTorch)
            
        Returns:
            Tokenized output
        """
        if max_length is None:
            max_length = self.max_source_length
        
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
    
    def encode_source(
        self,
        text: str,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Encode source text (query for NPP).
        
        Args:
            text: Source text
            max_length: Maximum length
            **kwargs: Additional tokenization arguments
            
        Returns:
            Encoded tokens
        """
        if max_length is None:
            max_length = self.max_source_length
        
        return self.tokenize(text, max_length=max_length, **kwargs)
    
    def encode_target(
        self,
        text: str,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Encode target text (predicted phrase for NPP).
        
        Args:
            text: Target text
            max_length: Maximum length
            **kwargs: Additional tokenization arguments
            
        Returns:
            Encoded tokens
        """
        if max_length is None:
            max_length = self.max_target_length
        
        return self.tokenize(text, max_length=max_length, **kwargs)
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: Token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def batch_encode_source(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Batch encode source texts.
        
        Args:
            texts: List of source texts
            max_length: Maximum length
            **kwargs: Additional tokenization arguments
            
        Returns:
            Batch encoded tokens
        """
        if max_length is None:
            max_length = self.max_source_length
        
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            **kwargs
        )
    
    def batch_encode_target(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Batch encode target texts.
        
        Args:
            texts: List of target texts
            max_length: Maximum length
            **kwargs: Additional tokenization arguments
            
        Returns:
            Batch encoded tokens
        """
        if max_length is None:
            max_length = self.max_target_length
        
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            **kwargs
        )
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
