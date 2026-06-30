"""
Phrase extraction utilities for preprocessing sentences.
"""

import re
from typing import List, Dict, Tuple, Set, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from utils.logger import get_logger

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class PhraseExtractor:
    """
    Extract and process phrases from sentences.
    """
    
    def __init__(self, language: str = "english"):
        """
        Initialize phrase extractor.
        
        Args:
            language: Language for stopword filtering
        """
        self.language = language
        try:
            self.stopwords = set(stopwords.words(language))
        except:
            logger.warning(f"Stopwords not available for {language}")
            self.stopwords = set()
    
    @staticmethod
    def normalize_text(text: str, lowercase: bool = True, remove_urls: bool = True) -> str:
        """
        Normalize text.
        
        Args:
            text: Input text
            lowercase: Whether to convert to lowercase
            remove_urls: Whether to remove URLs
            
        Returns:
            Normalized text
        """
        if remove_urls:
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        if lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def tokenize_sentences(text: str) -> List[str]:
        """
        Tokenize text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback to simple splitting
            sentences = text.split('. ')
        
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def tokenize_words(sentence: str) -> List[str]:
        """
        Tokenize sentence into words.
        
        Args:
            sentence: Input sentence
            
        Returns:
            List of words
        """
        try:
            tokens = word_tokenize(sentence)
        except:
            tokens = sentence.split()
        
        return tokens
    
    def filter_phrase(self, phrase: str, min_length: int = 2, max_length: int = 20) -> bool:
        """
        Check if phrase meets quality criteria.
        
        Args:
            phrase: Phrase to filter
            min_length: Minimum number of words
            max_length: Maximum number of words
            
        Returns:
            True if phrase passes filter
        """
        words = self.tokenize_words(phrase)
        
        if len(words) < min_length or len(words) > max_length:
            return False
        
        # At least one non-stopword
        non_stopwords = [w for w in words if w.lower() not in self.stopwords]
        if not non_stopwords:
            return False
        
        # No special characters (except hyphens)
        if not re.match(r'^[a-zA-Z0-9\s\-]+$', phrase):
            return False
        
        return True
    
    def extract_ngrams(
        self,
        sentence: str,
        n_min: int = 1,
        n_max: int = 5,
        min_length: int = 2,
        max_length: int = 20
    ) -> List[str]:
        """
        Extract n-grams from sentence.
        
        Args:
            sentence: Input sentence
            n_min: Minimum n for n-grams
            n_max: Maximum n for n-grams
            min_length: Minimum phrase length
            max_length: Maximum phrase length
            
        Returns:
            List of extracted phrases
        """
        words = self.tokenize_words(sentence)
        phrases = set()
        
        for n in range(n_min, min(n_max + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                if self.filter_phrase(phrase, min_length, max_length):
                    phrases.add(phrase)
        
        return list(phrases)
    
    @staticmethod
    def get_phrase_position(sentence: str, phrase: str) -> Optional[Tuple[int, int]]:
        """
        Get start and end position of phrase in sentence.
        
        Args:
            sentence: Full sentence
            phrase: Phrase to locate
            
        Returns:
            Tuple of (start, end) or None if not found
        """
        phrase_lower = phrase.lower()
        sentence_lower = sentence.lower()
        
        start = sentence_lower.find(phrase_lower)
        if start == -1:
            return None
        
        end = start + len(phrase)
        return (start, end)
    
    @staticmethod
    def get_context_before_phrase(
        sentence: str,
        phrase: str,
        context_type: str = "prefix"
    ) -> Optional[str]:
        """
        Get context before a phrase.
        
        Args:
            sentence: Full sentence
            phrase: Target phrase
            context_type: "prefix" (full prefix) or "last_word" (last word before phrase)
            
        Returns:
            Context string or None
        """
        pos = PhraseExtractor.get_phrase_position(sentence, phrase)
        if pos is None:
            return None
        
        start, end = pos
        
        if context_type == "prefix":
            return sentence[:start].strip()
        elif context_type == "last_word":
            prefix = sentence[:start].strip()
            words = prefix.split()
            return words[-1] if words else ""
        
        return None
    
    @staticmethod
    def get_context_after_phrase(
        sentence: str,
        phrase: str,
        context_type: str = "suffix"
    ) -> Optional[str]:
        """
        Get context after a phrase.
        
        Args:
            sentence: Full sentence
            phrase: Target phrase
            context_type: "suffix" (remaining) or "first_word" (first word after)
            
        Returns:
            Context string or None
        """
        pos = PhraseExtractor.get_phrase_position(sentence, phrase)
        if pos is None:
            return None
        
        start, end = pos
        
        if context_type == "suffix":
            return sentence[end:].strip()
        elif context_type == "first_word":
            suffix = sentence[end:].strip()
            words = suffix.split()
            return words[0] if words else ""
        
        return None
    
    def remove_duplicate_phrases(self, phrases: List[str]) -> List[str]:
        """
        Remove duplicate phrases while preserving order.
        
        Args:
            phrases: List of phrases
            
        Returns:
            List of unique phrases
        """
        seen = set()
        unique = []
        
        for phrase in phrases:
            phrase_normalized = phrase.lower().strip()
            if phrase_normalized not in seen:
                seen.add(phrase_normalized)
                unique.append(phrase)
        
        return unique
