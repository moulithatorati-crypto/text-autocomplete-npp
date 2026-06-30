"""
Main orchestration script for NPP Text Auto-Completion system.
Handles data preparation, training, fine-tuning, and inference.
"""

import os
import torch
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from utils.logger import get_logger
from utils.helpers import load_config, create_directories
from utils.seed import setup_reproducibility

logger = get_logger(__name__)


class NPPPipeline:
    """
    Complete pipeline for NPP training and inference.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.setup_directories()
        self.setup_reproducibility()
    
    def setup_directories(self) -> None:
        """
        Create necessary directories.
        """
        dirs = [
            self.config["dataset"]["raw_path"],
            self.config["dataset"]["processed_path"],
            self.config["output"]["checkpoint_dir"],
            self.config["output"]["results_dir"],
            self.config["output"]["predictions_dir"],
            self.config["logging"]["tensorboard_dir"],
        ]
        create_directories(dirs)
        logger.info("✓ Directories created")
    
    def setup_reproducibility(self) -> None:
        """
        Setup reproducibility settings.
        """
        setup_reproducibility(
            seed=self.config["system"]["seed"],
            deterministic=self.config["system"]["deterministic"],
            disable_cudnn_benchmark=self.config["system"]["cudnn_benchmark"]
        )
        logger.info("✓ Reproducibility setup complete")
    
    def prepare_data(
        self,
        data_path: Optional[str] = None,
        generate_npp: bool = True
    ) -> None:
        """
        Prepare dataset for training.
        
        Args:
            data_path: Path to raw data
            generate_npp: Whether to generate NPP dataset
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*60)
        
        if not data_path:
            logger.warning("No data path provided. Skipping data preparation.")
            logger.info("To prepare data:")
            logger.info(f"  1. Place raw data in: {self.config['dataset']['raw_path']}")
            logger.info("  2. Run: python main.py --prepare-data --data-path <path>")
            return
        
        logger.info(f"Loading data from: {data_path}")
        
        # TODO: Implement actual data loading and preprocessing
        logger.info("✓ Data preparation would run here")
        logger.info(f"  - Input: {data_path}")
        logger.info(f"  - Output: {self.config['dataset']['processed_path']}")
    
    def train_npp(
        self,
        resume_from: Optional[str] = None
    ) -> None:
        """
        Train model on NPP task.
        
        Args:
            resume_from: Checkpoint to resume from
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 2: NPP INTERMEDIATE TRAINING")
        logger.info("="*60)
        
        logger.info("Training configuration:")
        logger.info(f"  - Epochs: {self.config['training']['epochs']}")
        logger.info(f"  - Batch size: {self.config['training']['batch_size']}")
        logger.info(f"  - Learning rate: {self.config['training']['learning_rate']}")
        logger.info(f"  - Output: {self.config['training']['output_dir']}")
        
        logger.info("✓ NPP training would run here")
    
    def finetune(
        self,
        resume_from: Optional[str] = None
    ) -> None:
        """
        Fine-tune model on autocomplete task.
        
        Args:
            resume_from: Checkpoint to resume from
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 3: FINE-TUNING FOR AUTO-COMPLETION")
        logger.info("="*60)
        
        logger.info("Fine-tuning configuration:")
        logger.info(f"  - Epochs: {self.config['finetuning']['epochs']}")
        logger.info(f"  - Batch size: {self.config['finetuning']['batch_size']}")
        logger.info(f"  - Learning rate: {self.config['finetuning']['learning_rate']}")
        logger.info(f"  - Output: {self.config['finetuning']['output_dir']}")
        
        logger.info("✓ Fine-tuning would run here")
    
    def evaluate(
        self,
        test_data_path: Optional[str] = None
    ) -> None:
        """
        Evaluate model on test set.
        
        Args:
            test_data_path: Path to test data
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 4: EVALUATION")
        logger.info("="*60)
        
        logger.info("Evaluation configuration:")
        logger.info(f"  - Metrics: {self.config['evaluation']['metrics']}")
        logger.info(f"  - Output: {self.config['evaluation']['output_dir']}")
        
        logger.info("✓ Evaluation would run here")
    
    def infer(
        self,
        text: str,
        num_candidates: int = 5
    ) -> None:
        """
        Run inference on text.
        
        Args:
            text: Input text
            num_candidates: Number of candidates
        """
        logger.info("\n" + "="*60)
        logger.info("STEP 5: INFERENCE")
        logger.info("="*60)
        
        logger.info(f"Input text: {text}")
        logger.info(f"Number of candidates: {num_candidates}")
        
        try:
            from inference.autocomplete import AutocompleteEngine
            
            model_path = self.config["inference"]["model_path"]
            logger.info(f"Loading model from: {model_path}")
            
            if not Path(model_path).exists():
                logger.error(f"Model not found at: {model_path}")
                logger.info("Please train the model first using: python main.py --train-npp")
                return
            
            engine = AutocompleteEngine(
                model_path=model_path,
                device=self.config["inference"]["device"]
            )
            
            completions = engine.complete(
                partial_text=text,
                num_return_sequences=num_candidates,
                beam_size=self.config["inference"]["beam_size"],
                max_length=self.config["inference"]["max_length"]
            )
            
            logger.info("\nGenerated Completions:")
            for i, completion in enumerate(completions, 1):
                logger.info(f"  {i}. {completion}")
        
        except Exception as e:
            logger.error(f"Inference failed: {e}")
    
    def demo(self) -> None:
        """
        Run interactive demo.
        """
        logger.info("\n" + "="*60)
        logger.info("INTERACTIVE DEMO")
        logger.info("="*60)
        
        examples = [
            "The quick brown fox",
            "Machine learning is",
            "Natural language processing",
        ]
        
        for example in examples:
            logger.info(f"\nExample: {example}")
            self.infer(example, num_candidates=3)
    
    def print_config(self) -> None:
        """
        Print current configuration.
        """
        logger.info("\n" + "="*60)
        logger.info("CONFIGURATION")
        logger.info("="*60)
        
        def print_dict(d: Dict[str, Any], indent: int = 0) -> None:
            for key, value in d.items():
                if isinstance(value, dict):
                    logger.info(" " * indent + f"{key}:")
                    print_dict(value, indent + 2)
                else:
                    logger.info(" " * indent + f"{key}: {value}")
        
        print_dict(self.config)


def main():
    """
    Main entry point for CLI.
    """
    parser = argparse.ArgumentParser(
        description="NPP Text Auto-Completion System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data
  python main.py --prepare-data --data-path data/raw/
  
  # Train NPP model
  python main.py --train-npp
  
  # Fine-tune for auto-completion
  python main.py --finetune
  
  # Run inference
  python main.py --infer "The quick brown"
  
  # Launch Gradio UI
  python main.py --gradio
  
  # Run demo
  python main.py --demo
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )
    
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Prepare dataset"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to raw data"
    )
    
    parser.add_argument(
        "--train-npp",
        action="store_true",
        help="Train NPP model"
    )
    
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune for auto-completion"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model"
    )
    
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test data"
    )
    
    parser.add_argument(
        "--infer",
        type=str,
        default=None,
        help="Run inference on text"
    )
    
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=5,
        help="Number of candidates for inference"
    )
    
    parser.add_argument(
        "--gradio",
        action="store_true",
        help="Launch Gradio UI"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive demo"
    )
    
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show current configuration"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = NPPPipeline(config_path=args.config)
    
    # Show configuration
    if args.show_config:
        pipeline.print_config()
        return
    
    # Run selected pipeline
    if args.prepare_data:
        pipeline.prepare_data(data_path=args.data_path)
    
    if args.train_npp:
        pipeline.train_npp()
    
    if args.finetune:
        pipeline.finetune()
    
    if args.evaluate:
        pipeline.evaluate(test_data_path=args.test_data)
    
    if args.infer:
        pipeline.infer(text=args.infer, num_candidates=args.num_candidates)
    
    if args.gradio:
        from webapp.gradio_app import GradioApp
        app = GradioApp(
            model_path=pipeline.config["inference"]["model_path"],
            config_path=args.config
        )
        app.launch()
    
    if args.demo:
        pipeline.demo()
    
    # If no action specified, show help
    if not any([args.prepare_data, args.train_npp, args.finetune, args.evaluate, args.infer, args.gradio, args.demo, args.show_config]):
        parser.print_help()


if __name__ == "__main__":
    main()
