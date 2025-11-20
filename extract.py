"""
Claim extraction module for identifying verifiable statements from text.
"""

import re
import spacy
from typing import List, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from config import SPACY_MODEL, MAX_CLAIM_LENGTH
from utils import get_logger

logger = get_logger(__name__)


class ClaimExtractor:
    """Extract verifiable claims from text using multiple strategies."""
    
    def __init__(self, use_spacy: bool = True, use_transformers: bool = False):
        """
        Initialize the claim extractor.
        
        Args:
            use_spacy: Whether to use spaCy for NER-based extraction
            use_transformers: Whether to use HuggingFace transformers
        """
        self.use_spacy = use_spacy
        self.use_transformers = use_transformers
        
        self.nlp = None
        self.claim_classifier = None
        
        if use_spacy:
            try:
                self.nlp = spacy.load(SPACY_MODEL)
                logger.info(f"Loaded spaCy model: {SPACY_MODEL}")
            except OSError:
                logger.warning(f"spaCy model {SPACY_MODEL} not found. Install with: python -m spacy download {SPACY_MODEL}")
                self.use_spacy = False
        
        if use_transformers:
            try:
                # Use a zero-shot classification model for claim detection
                self.claim_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                logger.info("Loaded HuggingFace zero-shot classifier")
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace model: {e}")
                self.use_transformers = False
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract verifiable claims from input text.
        
        Args:
            text: Input text to extract claims from
            
        Returns:
            List of extracted claims
        """
        logger.info(f"Extracting claims from text: {text[:100]}...")
        
        claims = []
        
        # Strategy 1: Sentence-based extraction with filtering
        sentence_claims = self._extract_sentence_claims(text)
        claims.extend(sentence_claims)
        
        # Strategy 2: spaCy-based extraction (if available)
        if self.use_spacy and self.nlp:
            spacy_claims = self._extract_spacy_claims(text)
            claims.extend(spacy_claims)
        
        # Strategy 3: Transformer-based extraction (if available)
        if self.use_transformers and self.claim_classifier:
            transformer_claims = self._extract_transformer_claims(text)
            claims.extend(transformer_claims)
        
        # Remove duplicates and filter
        claims = self._deduplicate_and_filter(claims)
        
        logger.info(f"Extracted {len(claims)} claims")
        return claims
    
    def _extract_sentence_claims(self, text: str) -> List[str]:
        """Extract claims by splitting into sentences and applying heuristics."""
        sentences = self._split_sentences(text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if self._is_potential_claim(sentence):
                claims.append(sentence)
        
        return claims
    
    def _extract_spacy_claims(self, text: str) -> List[str]:
        """Extract claims using spaCy NER and dependency parsing."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        claims = []
        
        # Look for sentences with specific patterns
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Check for factual patterns
            if self._has_factual_patterns(doc, sent):
                claims.append(sent_text)
        
        return claims
    
    def _extract_transformer_claims(self, text: str) -> List[str]:
        """Extract claims using transformer-based classification."""
        if not self.claim_classifier:
            return []
        
        sentences = self._split_sentences(text)
        claims = []
        
        # Classify each sentence as claim or not
        labels = ["factual claim", "opinion", "question", "other"]
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
                
            try:
                result = self.claim_classifier(sentence, labels)
                
                # If classified as factual claim with high confidence
                if (result['labels'][0] == "factual claim" and 
                    result['scores'][0] > 0.5):
                    claims.append(sentence.strip())
            
            except Exception as e:
                logger.warning(f"Error classifying sentence: {e}")
                continue
        
        return claims
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_potential_claim(self, sentence: str) -> bool:
        """Check if a sentence could be a verifiable claim."""
        sentence = sentence.lower().strip()
        
        # Skip if too short or too long
        if len(sentence) < 10 or len(sentence) > MAX_CLAIM_LENGTH:
            return False
        
        # Skip questions
        if sentence.endswith('?'):
            return False
        
        # Skip first-person statements (likely opinions)
        first_person_patterns = [
            r'\bi think\b', r'\bi believe\b', r'\bi feel\b', 
            r'\bin my opinion\b', r'\bi guess\b'
        ]
        for pattern in first_person_patterns:
            if re.search(pattern, sentence):
                return False
        
        # Look for factual indicators
        factual_patterns = [
            r'\b(announced|declared|confirmed|reported|stated)\b',
            r'\b(will|has|have|is|are|was|were)\b',
            r'\b(starting|beginning|since|from|until)\b',
            r'\b(government|company|organization|study|research)\b',
            r'\b\d{4}\b',  # Years
            r'\b(percent|%|\$|million|billion)\b'  # Numbers/statistics
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, sentence):
                return True
        
        return False
    
    def _has_factual_patterns(self, doc, sent) -> bool:
        """Check if sentence has factual patterns using spaCy analysis."""
        sent_text = sent.text.lower()
        
        # Look for named entities (organizations, dates, etc.)
        entities = [ent.label_ for ent in sent.ents]
        factual_entities = ['ORG', 'DATE', 'MONEY', 'PERCENT', 'GPE', 'PERSON']
        
        if any(entity in factual_entities for entity in entities):
            return True
        
        # Look for declarative verbs
        declarative_verbs = ['announce', 'declare', 'confirm', 'state', 'report']
        for token in sent:
            if token.lemma_ in declarative_verbs:
                return True
        
        return False
    
    def _deduplicate_and_filter(self, claims: List[str]) -> List[str]:
        """Remove duplicates and apply final filtering."""
        # Remove exact duplicates
        unique_claims = list(dict.fromkeys(claims))
        
        # Remove very similar claims (simple approach)
        filtered_claims = []
        for claim in unique_claims:
            is_duplicate = False
            for existing in filtered_claims:
                # Simple similarity check
                if self._are_similar(claim, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_claims.append(claim)
        
        return filtered_claims
    
    def _are_similar(self, claim1: str, claim2: str, threshold: float = 0.8) -> bool:
        """Check if two claims are similar (simple word overlap)."""
        words1 = set(claim1.lower().split())
        words2 = set(claim2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0
        return similarity > threshold


def extract_claims(text: str) -> List[str]:
    """
    Convenience function to extract claims from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted claims
    """
    extractor = ClaimExtractor(use_spacy=True, use_transformers=False)
    return extractor.extract_claims(text)