"""
Constituency parsing module for extracting phrases from sentences.
Supports both Benepar and AllenNLP parsers.
"""

import torch
from typing import List, Dict, Optional, Tuple, Set
from nltk.tree import Tree
import nltk
from utils.logger import get_logger

logger = get_logger(__name__)


class ConstituencyParser:
    """
    Wrapper for constituency parsing to extract noun phrases, verb phrases, etc.
    Supports multiple parsing backends (Benepar, AllenNLP).
    """
    
    def __init__(
        self,
        parser_type: str = "benepar",
        model_name: str = "benepar_en3",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the constituency parser.
        
        Args:
            parser_type: Type of parser ("benepar" or "allennlp")
            model_name: Name of the model to load
            device: Device to run parser on ("cuda" or "cpu")
        """
        self.parser_type = parser_type
        self.model_name = model_name
        self.device = device
        self.parser = None
        
        self._init_parser()
    
    def _init_parser(self) -> None:
        """
        Initialize the parser based on type.
        """
        try:
            if self.parser_type == "benepar":
                self._init_benepar()
            elif self.parser_type == "allennlp":
                self._init_allennlp()
            else:
                raise ValueError(f"Unknown parser type: {self.parser_type}")
            
            logger.info(f"✓ {self.parser_type} parser initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize parser: {e}")
            raise
    
    def _init_benepar(self) -> None:
        """
        Initialize Benepar parser.
        """
        try:
            import benepar
            
            # Download model if not present
            try:
                benepar.download(self.model_name)
            except:
                pass  # Model might already be downloaded
            
            self.parser = benepar.Parser(self.model_name, device=self.device)
        except ImportError:
            logger.error("benepar not installed. Install with: pip install benepar")
            raise
    
    def _init_allennlp(self) -> None:
        """
        Initialize AllenNLP parser.
        """
        try:
            from allennlp.predictors.predictor import Predictor
            self.parser = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
            )
        except ImportError:
            logger.error("allennlp not installed. Install with: pip install allennlp")
            raise
    
    def parse(self, sentence: str) -> Optional[Tree]:
        """
        Parse a single sentence and return parse tree.
        
        Args:
            sentence: Input sentence to parse
            
        Returns:
            NLTK Tree object or None if parsing fails
        """
        try:
            if self.parser_type == "benepar":
                tree = self.parser.parse(sentence)
            else:  # allennlp
                result = self.parser.predict(sentence=sentence)
                tree_str = result["hierplane_tree"]["root"]
                tree = Tree.fromstring(tree_str)
            
            return tree
        except Exception as e:
            logger.warning(f"Failed to parse: {sentence[:50]}... Error: {e}")
            return None
    
    def extract_phrases(
        self,
        tree: Tree,
        phrase_types: List[str] = ["NP", "VP", "PP"]
    ) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        Extract phrases of specified types from parse tree.
        Returns only the lowest-level phrases (no nested phrases of same type).
        
        Args:
            tree: Parse tree from parser
            phrase_types: List of phrase types to extract (e.g., ["NP", "VP", "PP"])
            
        Returns:
            Dictionary mapping phrase type to list of (start_pos, end_pos, phrase_text)
        """
        phrases_dict = {ptype: [] for ptype in phrase_types}
        
        def get_position_in_sentence(subtree: Tree) -> Tuple[int, int]:
            """
            Get character positions of a subtree in the sentence.
            
            Args:
                subtree: Subtree to get position for
                
            Returns:
                Tuple of (start_position, end_position)
            """
            leaves = list(subtree.leaves())
            if not leaves:
                return 0, 0
            
            start = 0
            end = 0
            for leaf in leaves:
                if leaf:
                    end += len(str(leaf)) + 1
            
            return start, end
        
        def traverse(subtree: Tree, phrase_types: List[str]) -> None:
            """
            Recursively traverse tree and extract phrases.
            
            Args:
                subtree: Current subtree
                phrase_types: Phrase types to extract
            """
            if isinstance(subtree, Tree):
                label = subtree.label()
                
                # Check if this is a phrase we want to extract
                if label in phrase_types:
                    # Get the phrase text
                    phrase_text = " ".join(subtree.leaves())
                    
                    # Check if children contain same phrase type (nested)
                    has_nested_same_type = any(
                        isinstance(child, Tree) and child.label() == label
                        for child in subtree
                    )
                    
                    # Only add if no nested phrases of same type
                    if not has_nested_same_type:
                        phrases_dict[label].append(phrase_text)
                    else:
                        # Recursively process children
                        for child in subtree:
                            if isinstance(child, Tree):
                                traverse(child, phrase_types)
                else:
                    # Continue traversing
                    for child in subtree:
                        if isinstance(child, Tree):
                            traverse(child, phrase_types)
        
        traverse(tree, phrase_types)
        return phrases_dict
    
    def parse_and_extract(
        self,
        sentence: str,
        phrase_types: List[str] = ["NP", "VP", "PP"]
    ) -> Optional[Dict[str, List[str]]]:
        """
        Parse sentence and extract phrases in one step.
        
        Args:
            sentence: Input sentence
            phrase_types: Phrase types to extract
            
        Returns:
            Dictionary of extracted phrases or None if parsing fails
        """
        tree = self.parse(sentence)
        if tree is None:
            return None
        
        return self.extract_phrases(tree, phrase_types)
    
    def batch_parse_and_extract(
        self,
        sentences: List[str],
        phrase_types: List[str] = ["NP", "VP", "PP"],
        batch_size: int = 32
    ) -> List[Optional[Dict[str, List[str]]]]:
        """
        Parse and extract phrases from batch of sentences.
        
        Args:
            sentences: List of sentences to parse
            phrase_types: Phrase types to extract
            batch_size: Batch size for processing
            
        Returns:
            List of extracted phrase dictionaries
        """
        results = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            for sentence in batch:
                result = self.parse_and_extract(sentence, phrase_types)
                results.append(result)
            
            if (i + batch_size) % (batch_size * 5) == 0:
                logger.info(f"Processed {i + batch_size}/{len(sentences)} sentences")
        
        return results
