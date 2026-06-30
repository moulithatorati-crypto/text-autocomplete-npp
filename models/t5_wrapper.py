"""
T5 Model wrapper for NPP training and inference.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
from utils.logger import get_logger
from utils.helpers import count_parameters, format_parameters, get_model_size

logger = get_logger(__name__)


class T5Wrapper:
    """
    Wrapper around T5 model for conditional text generation.
    Supports training and inference for NPP tasks.
    """
    
    def __init__(
        self,
        model_name: str = "t5-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize T5 wrapper.
        
        Args:
            model_name: Model name (e.g., "t5-base", "t5-large")
            device: Device to load model on
            cache_dir: Directory to cache models
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load T5 model and tokenizer.
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
            self.model.to(self.device)
            
            num_params = count_parameters(self.model)
            model_size = get_model_size(self.model)
            
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  - Parameters: {format_parameters(num_params)}")
            logger.info(f"  - Size: {model_size:.2f}MB")
            logger.info(f"  - Device: {self.device}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def save_model(self, output_dir: str) -> None:
        """
        Save model and tokenizer to directory.
        
        Args:
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.model.save_pretrained(str(output_dir))
            self.tokenizer.save_pretrained(str(output_dir))
            logger.info(f"✓ Model saved to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_pretrained(self, model_path: str) -> None:
        """
        Load pretrained model from directory.
        
        Args:
            model_path: Path to pretrained model
        """
        try:
            logger.info(f"Loading pretrained model from {model_path}")
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            self.model.to(self.device)
            
            logger.info(f"✓ Pretrained model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        early_stopping: bool = True,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = False,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate sequences using the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            early_stopping: Whether to use early stopping
            num_return_sequences: Number of sequences to return
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty
            length_penalty: Length penalty for beam search
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token sequences
        """
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
        
        return outputs
    
    def predict_batch(
        self,
        texts: List[str],
        max_length: int = 128,
        num_beams: int = 4,
        num_return_sequences: int = 1,
        batch_size: int = 32,
        **kwargs
    ) -> List[str]:
        """
        Generate predictions for batch of texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum generation length
            num_beams: Number of beams
            num_return_sequences: Number of sequences per input
            batch_size: Batch size for processing
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated texts
        """
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            output_ids = self.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                **kwargs
            )
            
            # Decode
            batch_predictions = self.tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True
            )
            predictions.extend(batch_predictions)
        
        return predictions
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for encoder
            decoder_input_ids: Decoder input token IDs
            decoder_attention_mask: Attention mask for decoder
            labels: Target labels for loss computation
            **kwargs: Additional model arguments
            
        Returns:
            Model outputs
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def get_encoder(self) -> nn.Module:
        """
        Get encoder module.
        
        Returns:
            Encoder module
        """
        return self.model.encoder
    
    def get_decoder(self) -> nn.Module:
        """
        Get decoder module.
        
        Returns:
            Decoder module
        """
        return self.model.decoder
    
    def freeze_encoder(self) -> None:
        """
        Freeze encoder parameters.
        """
        for param in self.get_encoder().parameters():
            param.requires_grad = False
        logger.info("✓ Encoder frozen")
    
    def unfreeze_encoder(self) -> None:
        """
        Unfreeze encoder parameters.
        """
        for param in self.get_encoder().parameters():
            param.requires_grad = True
        logger.info("✓ Encoder unfrozen")
    
    def freeze_decoder(self) -> None:
        """
        Freeze decoder parameters.
        """
        for param in self.get_decoder().parameters():
            param.requires_grad = False
        logger.info("✓ Decoder frozen")
    
    def unfreeze_decoder(self) -> None:
        """
        Unfreeze decoder parameters.
        """
        for param in self.get_decoder().parameters():
            param.requires_grad = True
        logger.info("✓ Decoder unfrozen")
    
    def get_trainable_parameters(self) -> int:
        """
        Get number of trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return count_parameters(self.model)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "trainable_parameters": self.get_trainable_parameters(),
            "model_size_mb": get_model_size(self.model),
            "vocab_size": len(self.tokenizer),
        }
