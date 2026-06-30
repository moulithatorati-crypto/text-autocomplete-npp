"""
Gradio web UI for NPP Text Auto-Completion System.
"""

import gradio as gr
import torch
from typing import List, Dict, Any
from inference.autocomplete import AutocompleteEngine
from utils.logger import get_logger
from utils.helpers import load_config

logger = get_logger(__name__)


class GradioApp:
    """
    Gradio web interface for the autocomplete system.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "configs/config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize Gradio app.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            device: Device to run on
        """
        self.model_path = model_path
        self.config = load_config(config_path)
        self.device = device
        
        # Initialize engine
        self.engine = AutocompleteEngine(
            model_path=model_path,
            device=device,
            decoding_strategy="beam_search"
        )
        
        logger.info("✓ Gradio app initialized")
    
    def autocomplete(
        self,
        partial_text: str,
        num_candidates: int = 5,
        temperature: float = 0.8,
        top_p: float = 0.95,
        beam_size: int = 5,
        max_length: int = 128
    ) -> str:
        """
        Generate completions for input text.
        
        Args:
            partial_text: Partial text to complete
            num_candidates: Number of candidates to generate
            temperature: Temperature for sampling
            top_p: Top-p parameter
            beam_size: Beam size
            max_length: Maximum length
            
        Returns:
            Formatted output string
        """
        if not partial_text.strip():
            return "Please enter some text to complete."
        
        try:
            result = self.engine.complete_interactive(
                partial_text=partial_text,
                num_candidates=num_candidates,
                temperature=temperature,
                top_p=top_p,
                beam_size=beam_size,
                max_length=max_length
            )
            
            # Format output
            output = f"🔍 Input: **{partial_text}**\n\n"
            output += "📝 **Completions:**\n\n"
            
            for i, completion in enumerate(result["completions"], 1):
                output += f"{i}. {completion}\n"
            
            return output
        
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"❌ Error: {str(e)}"
    
    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
        debug: bool = False
    ) -> None:
        """
        Launch Gradio interface.
        
        Args:
            server_name: Server name
            server_port: Server port
            share: Whether to create public share link
            debug: Debug mode
        """
        with gr.Blocks(
            title="🚀 Next Phrase Prediction - Text Auto-Completion",
            theme=gr.themes.Soft()
        ) as demo:
            # Header
            gr.Markdown(
                """
                # 🚀 Next Phrase Prediction - Text Auto-Completion
                
                An advanced text auto-completion system using Next Phrase Prediction (NPP)
                with T5 transformer model.
                
                **How to use:**
                1. Enter partial text in the input field
                2. Adjust generation parameters if desired
                3. Click "Generate Completions" button
                4. View top predicted completions
                """
            )
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=3):
                    input_text = gr.Textbox(
                        label="📝 Partial Text",
                        placeholder="Enter your partial text here...",
                        lines=3,
                        interactive=True
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Parameters")
                    
                    num_candidates = gr.Slider(
                        label="Number of Candidates",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1
                    )
                    
                    beam_size = gr.Slider(
                        label="Beam Size",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1
                    )
                    
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1
                    )
                    
                    top_p = gr.Slider(
                        label="Top-p (Nucleus)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05
                    )
                    
                    max_length = gr.Slider(
                        label="Max Length",
                        minimum=10,
                        maximum=256,
                        value=128,
                        step=10
                    )
            
            gr.Markdown("---")
            
            with gr.Row():
                submit_btn = gr.Button(
                    "🎯 Generate Completions",
                    variant="primary",
                    scale=1
                )
                clear_btn = gr.Button("🗑️ Clear", scale=0)
            
            output_text = gr.Markdown(label="📊 Results")
            
            # Event handlers
            submit_btn.click(
                fn=self.autocomplete,
                inputs=[
                    input_text,
                    num_candidates,
                    temperature,
                    top_p,
                    beam_size,
                    max_length
                ],
                outputs=output_text
            )
            
            clear_btn.click(
                fn=lambda: ("", 5, 5, 0.8, 0.95, 128, ""),
                outputs=[input_text, num_candidates, beam_size, temperature, top_p, max_length, output_text]
            )
            
            # Examples
            gr.Markdown("---")
            gr.Markdown("### 📚 Example Prompts")
            
            examples = [
                "The quick brown fox",
                "Machine learning is a subset of",
                "Natural language processing helps computers",
                "The weather today is",
                "According to recent studies"
            ]
            
            gr.Examples(
                examples=[[ex] for ex in examples],
                inputs=[input_text]
            )
        
        logger.info(f"🚀 Launching Gradio app on {server_name}:{server_port}")
        
        demo.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug
        )


def main():
    """
    Main entry point for Gradio app.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio UI for NPP system")
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/checkpoints/finetune",
        help="Path to trained model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Server name"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    
    args = parser.parse_args()
    
    app = GradioApp(
        model_path=args.model_path,
        config_path=args.config,
        device=args.device
    )
    
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )


if __name__ == "__main__":
    main()
