"""
Trainer for NPP and fine-tuning stages.
Handles mixed precision, gradient accumulation, checkpointing, and more.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LinearLR, CosineAnnealingLR
from accelerate import Accelerator
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path
import json
from datetime import datetime

from models.t5_wrapper import T5Wrapper
from utils.logger import get_logger
from utils.helpers import save_json, load_json

logger = get_logger(__name__)


class NPPTrainer:
    """
    Trainer for NPP and autocomplete fine-tuning tasks.
    """
    
    def __init__(
        self,
        model: T5Wrapper,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: T5Wrapper model instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration dictionary
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}
        
        # Extract config
        self.batch_size = self.config.get("batch_size", 32)
        self.epochs = self.config.get("epochs", 10)
        self.learning_rate = self.config.get("learning_rate", 1e-4)
        self.weight_decay = self.config.get("weight_decay", 0.01)
        self.warmup_steps = self.config.get("warmup_steps", 500)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self.mixed_precision = self.config.get("mixed_precision", "fp16")
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        self.num_workers = self.config.get("num_workers", 4)
        self.output_dir = self.config.get("output_dir", "outputs/checkpoints")
        self.logging_steps = self.config.get("logging_steps", 100)
        self.eval_steps = self.config.get("eval_steps", None)
        self.save_strategy = self.config.get("save_strategy", "epoch")
        self.early_stopping_patience = self.config.get("early_stopping_patience", 3)
        
        # Initialize
        self.accelerator = Accelerator(
            mixed_precision=self.mixed_precision if self.mixed_precision != "no" else "no"
        )
        
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        
        self._setup()
    
    def _setup(self) -> None:
        """
        Setup training components.
        """
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        if self.val_dataset:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False
            )
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Calculate total training steps
        total_steps = len(self.train_dataloader) * self.epochs // self.gradient_accumulation_steps
        
        # Setup scheduler
        self.scheduler = self._get_scheduler(total_steps)
        
        # Prepare with accelerator
        self.model.model, self.optimizer, self.train_dataloader = \
            self.accelerator.prepare(self.model.model, self.optimizer, self.train_dataloader)
        
        if self.val_dataloader:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)
        
        logger.info(f"✓ Training setup complete")
        logger.info(f"  - Batch size: {self.batch_size}")
        logger.info(f"  - Epochs: {self.epochs}")
        logger.info(f"  - Total steps: {total_steps}")
        logger.info(f"  - Learning rate: {self.learning_rate}")
    
    def _get_scheduler(self, total_steps: int):
        """
        Create learning rate scheduler.
        
        Args:
            total_steps: Total training steps
            
        Returns:
            Scheduler instance
        """
        scheduler_type = self.config.get("scheduler", "linear")
        
        if scheduler_type == "linear":
            def lr_lambda(current_step):
                if current_step < self.warmup_steps:
                    return float(current_step) / float(max(1, self.warmup_steps))
                return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - self.warmup_steps)))
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        elif scheduler_type == "cosine":
            def lr_lambda(current_step):
                if current_step < self.warmup_steps:
                    return float(current_step) / float(max(1, self.warmup_steps))
                progress = float(current_step - self.warmup_steps) / float(max(1, total_steps - self.warmup_steps))
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress))))
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        else:
            return LinearLR(self.optimizer, start_factor=1.0, total_iters=total_steps)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Loss value
        """
        self.model.model.train()
        
        outputs = self.model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        loss = loss / self.gradient_accumulation_steps
        
        self.accelerator.backward(loss)
        
        return loss.item() * self.gradient_accumulation_steps
    
    def eval_step(self) -> float:
        """
        Evaluate on validation set.
        
        Returns:
            Validation loss
        """
        if not self.val_dataloader:
            return float('inf')
        
        self.model.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", disable=not self.accelerator.is_main_process):
                outputs = self.model.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        return avg_loss
    
    def train(
        self,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting training...")
        
        start_epoch = 0
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint = torch.load(os.path.join(resume_from_checkpoint, "pytorch_model.bin"))
            self.model.model.load_state_dict(checkpoint)
            
            # Load training state
            state_path = os.path.join(resume_from_checkpoint, "training_state.json")
            if os.path.exists(state_path):
                state = load_json(state_path)
                start_epoch = state.get("epoch", 0)
                self.global_step = state.get("global_step", 0)
        
        for epoch in range(start_epoch, self.epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            logger.info(f"{'='*60}")
            
            # Training
            epoch_loss = 0.0
            num_steps = 0
            
            pbar = tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                disable=not self.accelerator.is_main_process,
                desc="Training"
            )
            
            for step, batch in pbar:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_steps += 1
                
                # Gradient accumulation and optimization step
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.max_grad_norm
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                
                # Logging
                if (step + 1) % self.logging_steps == 0:
                    avg_loss = epoch_loss / num_steps
                    lr = self.scheduler.get_last_lr()[0]
                    
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}"
                    })
                    
                    logger.info(
                        f"Step {self.global_step}: "
                        f"loss={avg_loss:.4f}, "
                        f"lr={lr:.2e}"
                    )
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_steps
            logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_epoch_loss:.4f}")
            
            # Validation
            if self.val_dataloader:
                val_loss = self.eval_step()
                logger.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
                
                # Early stopping and checkpoint saving
                if val_loss < self.best_eval_loss:
                    self.best_eval_loss = val_loss
                    self.patience_counter = 0
                    
                    if self.save_strategy == "epoch":
                        self._save_checkpoint(epoch)
                else:
                    self.patience_counter += 1
                    logger.info(f"Patience: {self.patience_counter}/{self.early_stopping_patience}")
                    
                    if self.patience_counter >= self.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break
            else:
                if self.save_strategy == "epoch":
                    self._save_checkpoint(epoch)
            
            # Track history
            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "val_loss": val_loss if self.val_dataloader else None,
                "global_step": self.global_step,
            })
        
        logger.info("\n" + "="*60)
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_eval_loss:.4f}")
        logger.info("="*60)
        
        return {"history": self.training_history, "best_loss": self.best_eval_loss}
    
    def _save_checkpoint(self, epoch: int) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
        """
        checkpoint_dir = os.path.join(self.output_dir, f"epoch_{epoch + 1}")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model.model)
        unwrapped_model.save_pretrained(checkpoint_dir)
        self.model.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config
        }
        save_json(state, os.path.join(checkpoint_dir, "training_state.json"))
        
        logger.info(f"✓ Checkpoint saved: {checkpoint_dir}")
