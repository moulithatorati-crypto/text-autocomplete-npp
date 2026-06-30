"""
NPP Dataset builder for generating training data.
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from parsers.constituency_parser import ConstituencyParser
from preprocessing.phrase_extractor import PhraseExtractor
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NPPExample:
    """Single NPP training example."""
    input_text: str
    target_phrase: str
    phrase_type: str
    full_sentence: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "input_text": self.input_text,
            "target_phrase": self.target_phrase,
            "phrase_type": self.phrase_type,
            "full_sentence": self.full_sentence,
        }


class NPPDatasetBuilder:
    """
    Build Next Phrase Prediction dataset from sentences and extracted phrases.
    """
    
    def __init__(
        self,
        parser: Optional[ConstituencyParser] = None,
        extractor: Optional[PhraseExtractor] = None,
        seed: int = 42
    ):
        """
        Initialize dataset builder.
        
        Args:
            parser: Constituency parser instance
            extractor: Phrase extractor instance
            seed: Random seed
        """
        self.parser = parser
        self.extractor = extractor or PhraseExtractor()
        self.seed = seed
        random.seed(seed)
    
    def create_npp_example(
        self,
        sentence: str,
        phrase_groups: Dict[str, List[str]],
        phrase_type: Optional[str] = None,
        target_phrase: Optional[str] = None
    ) -> Optional[NPPExample]:
        """
        Create a single NPP example from a sentence and phrase.
        
        Args:
            sentence: Full sentence
            phrase_groups: Dictionary of phrase types to phrase lists
            phrase_type: Specific phrase type to use (if None, random)
            target_phrase: Specific phrase to use (if None, random)
            
        Returns:
            NPPExample or None if creation fails
        """
        # Select phrase type
        if phrase_type is None:
            available_types = [pt for pt, phrases in phrase_groups.items() if phrases]
            if not available_types:
                return None
            phrase_type = random.choice(available_types)
        
        if phrase_type not in phrase_groups or not phrase_groups[phrase_type]:
            return None
        
        # Select target phrase
        if target_phrase is None:
            target_phrase = random.choice(phrase_groups[phrase_type])
        
        # Get context before phrase
        context = PhraseExtractor.get_context_before_phrase(sentence, target_phrase)
        if context is None:
            return None
        
        # Create input text with NPP prompt
        input_text = f"generate next phrase: {context}"
        
        return NPPExample(
            input_text=input_text,
            target_phrase=target_phrase,
            phrase_type=phrase_type,
            full_sentence=sentence
        )
    
    def create_examples_from_sentence(
        self,
        sentence: str,
        phrase_groups: Dict[str, List[str]],
        num_examples: int = 5,
        strategy: str = "random"
    ) -> List[NPPExample]:
        """
        Create multiple NPP examples from a single sentence.
        
        Args:
            sentence: Full sentence
            phrase_groups: Dictionary of phrase types to phrase lists
            num_examples: Number of examples to create
            strategy: Strategy for creating examples ("random", "systematic", "stratified")
            
        Returns:
            List of NPPExample objects
        """
        examples = []
        
        if strategy == "random":
            for _ in range(num_examples):
                example = self.create_npp_example(sentence, phrase_groups)
                if example and example not in examples:
                    examples.append(example)
        
        elif strategy == "systematic":
            # Create examples for each phrase type
            phrase_types = [pt for pt, phrases in phrase_groups.items() if phrases]
            for phrase_type in phrase_types:
                phrases = phrase_groups[phrase_type]
                per_type = max(1, num_examples // len(phrase_types))
                
                for phrase in phrases[:per_type]:
                    example = self.create_npp_example(
                        sentence,
                        phrase_groups,
                        phrase_type=phrase_type,
                        target_phrase=phrase
                    )
                    if example and example not in examples:
                        examples.append(example)
        
        elif strategy == "stratified":
            # Ensure balanced distribution across phrase types
            phrase_types = [pt for pt, phrases in phrase_groups.items() if phrases]
            if not phrase_types:
                return examples
            
            per_type = num_examples // len(phrase_types)
            remainder = num_examples % len(phrase_types)
            
            for i, phrase_type in enumerate(phrase_types):
                count = per_type + (1 if i < remainder else 0)
                phrases = phrase_groups[phrase_type]
                
                for j, phrase in enumerate(phrases):
                    if j >= count:
                        break
                    
                    example = self.create_npp_example(
                        sentence,
                        phrase_groups,
                        phrase_type=phrase_type,
                        target_phrase=phrase
                    )
                    if example and example not in examples:
                        examples.append(example)
        
        return examples[:num_examples]
    
    def build_dataset(
        self,
        sentences: List[str],
        phrase_groups_list: List[Dict[str, List[str]]],
        num_examples_per_sentence: int = 5,
        strategy: str = "random",
        remove_duplicates: bool = True
    ) -> List[NPPExample]:
        """
        Build complete NPP dataset from sentences and phrases.
        
        Args:
            sentences: List of input sentences
            phrase_groups_list: List of phrase group dictionaries
            num_examples_per_sentence: Examples to create per sentence
            strategy: Strategy for example creation
            remove_duplicates: Whether to remove duplicate examples
            
        Returns:
            List of NPPExample objects
        """
        all_examples = []
        
        for i, (sentence, phrase_groups) in enumerate(zip(sentences, phrase_groups_list)):
            examples = self.create_examples_from_sentence(
                sentence,
                phrase_groups,
                num_examples=num_examples_per_sentence,
                strategy=strategy
            )
            all_examples.extend(examples)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Built {i + 1}/{len(sentences)} sentence examples")
        
        if remove_duplicates:
            # Remove duplicate examples
            seen = set()
            unique_examples = []
            
            for ex in all_examples:
                key = (ex.input_text, ex.target_phrase)
                if key not in seen:
                    seen.add(key)
                    unique_examples.append(ex)
            
            all_examples = unique_examples
            logger.info(f"Removed duplicates: {len(all_examples)} unique examples")
        
        return all_examples
    
    def save_examples(
        self,
        examples: List[NPPExample],
        output_path: str,
        format: str = "jsonl"
    ) -> None:
        """
        Save examples to file.
        
        Args:
            examples: List of examples
            output_path: Output file path
            format: Output format ("jsonl" or "csv")
        """
        import json
        import csv
        from pathlib import Path
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for ex in examples:
                    f.write(json.dumps(ex.to_dict()) + '\n')
        
        elif format == "csv":
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["input_text", "target_phrase", "phrase_type", "full_sentence"])
                writer.writeheader()
                for ex in examples:
                    writer.writerow(ex.to_dict())
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")
