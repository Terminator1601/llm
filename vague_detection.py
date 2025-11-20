"""
Vague claim detection and scoring module for filtering non-verifiable statements.
"""

import re
import spacy
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

from config import SPACY_MODEL
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class VaguenessScore:
    """Represents vagueness analysis of a claim."""
    claim: str
    vagueness_score: float  # 0.0 = specific, 1.0 = very vague
    vagueness_category: str  # "specific", "moderate", "vague", "very_vague"
    indicators: List[str]  # Specific indicators found
    recommendation: str  # "verify", "clarify", "reject"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "claim": self.claim,
            "vagueness_score": self.vagueness_score,
            "vagueness_category": self.vagueness_category,
            "indicators": self.indicators,
            "recommendation": self.recommendation
        }


class VagueClaimDetector:
    """Detect and score claims for vagueness to filter non-verifiable statements."""
    
    def __init__(self, spacy_model: str = SPACY_MODEL):
        """
        Initialize the vague claim detector.
        
        Args:
            spacy_model: spaCy model name for NLP processing
        """
        self.spacy_model = spacy_model
        self.nlp = None
        self.vectorizer = None
        self.model = None
        
        # Vagueness indicators
        self.vague_words = {
            "quantifiers": ["some", "many", "few", "several", "various", "numerous", "multiple", 
                           "certain", "most", "majority", "minority", "plenty", "lots"],
            "temporal": ["recently", "soon", "later", "eventually", "sometime", "often",
                        "frequently", "occasionally", "sometimes", "usually", "generally"],
            "hedging": ["possibly", "probably", "maybe", "perhaps", "might", "could", "would",
                       "seems", "appears", "allegedly", "reportedly", "supposedly", "likely"],
            "vague_refs": ["things", "stuff", "issues", "matters", "aspects", "factors",
                          "elements", "people", "experts", "sources", "studies", "reports"],
            "weak_verbs": ["suggests", "indicates", "implies", "seems", "appears", "tends",
                          "may be", "might be", "could be", "would be"]
        }
        
        self.specific_indicators = {
            "numbers": r'\b\d+(?:\.\d+)?(?:%|percent|million|billion|thousand|crore|lakh)?\b',
            "dates": r'\b(?:19|20)\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*(?:19|20)\d{2}\b',
            "specific_entities": ["WHO", "NASA", "GDP", "CEO", "PM", "President", "Minister"],
            "measurements": r'\b\d+(?:\.\d+)?\s*(?:meters?|km|miles?|degrees?|celsius|fahrenheit|kg|tons?|%)\b'
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load spaCy model and initialize components."""
        try:
            self.nlp = spacy.load(self.spacy_model)
            logger.info(f"Loaded spaCy model: {self.spacy_model}")
        except OSError:
            logger.warning(f"spaCy model {self.spacy_model} not found")
            self.nlp = None
    
    def score_vagueness(self, claim: str) -> VaguenessScore:
        """
        Score a claim for vagueness.
        
        Args:
            claim: Input claim to analyze
            
        Returns:
            VaguenessScore object with analysis
        """
        claim = claim.strip()
        if not claim:
            return VaguenessScore(
                claim=claim,
                vagueness_score=1.0,
                vagueness_category="very_vague",
                indicators=["empty_claim"],
                recommendation="reject"
            )
        
        # Calculate various vagueness indicators
        indicators = []
        scores = []
        
        # 1. Vague word analysis
        vague_word_score, vague_indicators = self._analyze_vague_words(claim)
        scores.append(vague_word_score)
        indicators.extend(vague_indicators)
        
        # 2. Specificity analysis
        specificity_score, specific_indicators = self._analyze_specificity(claim)
        scores.append(1.0 - specificity_score)  # Invert - high specificity = low vagueness
        indicators.extend([f"specific_{ind}" for ind in specific_indicators])
        
        # 3. Linguistic analysis (if spaCy available)
        if self.nlp:
            linguistic_score, ling_indicators = self._analyze_linguistics(claim)
            scores.append(linguistic_score)
            indicators.extend(ling_indicators)
        
        # 4. Structural analysis
        structural_score, struct_indicators = self._analyze_structure(claim)
        scores.append(structural_score)
        indicators.extend(struct_indicators)
        
        # Calculate overall vagueness score (weighted average)
        weights = [0.3, 0.3, 0.2, 0.2] if self.nlp else [0.4, 0.4, 0.2]
        if len(scores) != len(weights):
            weights = weights[:len(scores)]
            
        overall_score = np.average(scores, weights=weights)
        overall_score = max(0.0, min(1.0, overall_score))  # Clamp to [0, 1]
        
        # Categorize vagueness
        if overall_score <= 0.25:
            category = "specific"
            recommendation = "verify"
        elif overall_score <= 0.5:
            category = "moderate"
            recommendation = "verify"
        elif overall_score <= 0.75:
            category = "vague"
            recommendation = "clarify"
        else:
            category = "very_vague"
            recommendation = "reject"
        
        return VaguenessScore(
            claim=claim,
            vagueness_score=overall_score,
            vagueness_category=category,
            indicators=list(set(indicators)),  # Remove duplicates
            recommendation=recommendation
        )
    
    def _analyze_vague_words(self, claim: str) -> Tuple[float, List[str]]:
        """Analyze presence of vague words."""
        claim_lower = claim.lower()
        found_indicators = []
        total_vague_words = 0
        
        for category, words in self.vague_words.items():
            for word in words:
                if f" {word} " in f" {claim_lower} " or claim_lower.startswith(f"{word} ") or claim_lower.endswith(f" {word}"):
                    found_indicators.append(f"vague_{category}_{word}")
                    total_vague_words += 1
        
        # Calculate score based on density of vague words
        word_count = len(claim.split())
        if word_count == 0:
            return 1.0, found_indicators
        
        vague_density = total_vague_words / word_count
        score = min(1.0, vague_density * 2)  # Scale to [0, 1]
        
        return score, found_indicators
    
    def _analyze_specificity(self, claim: str) -> Tuple[float, List[str]]:
        """Analyze presence of specific elements (numbers, dates, entities)."""
        found_indicators = []
        specificity_points = 0
        
        # Check for numbers
        if re.search(self.specific_indicators["numbers"], claim, re.IGNORECASE):
            found_indicators.append("numbers")
            specificity_points += 2
        
        # Check for dates
        if re.search(self.specific_indicators["dates"], claim, re.IGNORECASE):
            found_indicators.append("dates")
            specificity_points += 2
        
        # Check for measurements
        if re.search(self.specific_indicators["measurements"], claim, re.IGNORECASE):
            found_indicators.append("measurements")
            specificity_points += 2
        
        # Check for specific entities
        for entity in self.specific_indicators["specific_entities"]:
            if entity.lower() in claim.lower():
                found_indicators.append(f"entity_{entity}")
                specificity_points += 1
        
        # Normalize score to [0, 1]
        max_possible_points = 8  # Rough estimate
        score = min(1.0, specificity_points / max_possible_points)
        
        return score, found_indicators
    
    def _analyze_linguistics(self, claim: str) -> Tuple[float, List[str]]:
        """Analyze linguistic features using spaCy."""
        doc = self.nlp(claim)
        indicators = []
        vagueness_score = 0.0
        
        # Check for proper nouns (more specific)
        proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
        if proper_nouns:
            indicators.append(f"proper_nouns_{len(proper_nouns)}")
            vagueness_score -= 0.1 * min(len(proper_nouns), 3)  # Reduce vagueness
        
        # Check for pronouns (can increase vagueness)
        pronouns = [token.text for token in doc if token.pos_ == "PRON"]
        vague_pronouns = ["this", "that", "these", "those", "it", "they", "them"]
        vague_pron_count = sum(1 for p in pronouns if p.lower() in vague_pronouns)
        if vague_pron_count > 0:
            indicators.append(f"vague_pronouns_{vague_pron_count}")
            vagueness_score += 0.1 * vague_pron_count
        
        # Check for named entities (more specific)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entities:
            indicators.append(f"named_entities_{len(entities)}")
            vagueness_score -= 0.05 * min(len(entities), 5)
        
        # Normalize to [0, 1] range
        vagueness_score = max(0.0, min(1.0, vagueness_score + 0.5))
        
        return vagueness_score, indicators
    
    def _analyze_structure(self, claim: str) -> Tuple[float, List[str]]:
        """Analyze structural features of the claim."""
        indicators = []
        vagueness_score = 0.0
        
        # Check length - very short or very long claims can be vague
        length = len(claim.split())
        if length < 5:
            indicators.append("too_short")
            vagueness_score += 0.3
        elif length > 40:
            indicators.append("too_long")
            vagueness_score += 0.2
        
        # Check for question marks (questions are not verifiable claims)
        if "?" in claim:
            indicators.append("question")
            vagueness_score += 0.4
        
        # Check for modal verbs (uncertainty)
        modal_verbs = ["should", "would", "could", "might", "may", "will", "shall"]
        claim_lower = claim.lower()
        modal_count = sum(1 for modal in modal_verbs if f" {modal} " in f" {claim_lower} ")
        if modal_count > 0:
            indicators.append(f"modals_{modal_count}")
            vagueness_score += 0.1 * modal_count
        
        # Check for negations (can make claims harder to verify)
        negations = ["not", "no", "never", "none", "nothing", "nowhere", "nobody"]
        neg_count = sum(1 for neg in negations if f" {neg} " in f" {claim_lower} ")
        if neg_count > 0:
            indicators.append(f"negations_{neg_count}")
            vagueness_score += 0.05 * neg_count
        
        return min(1.0, vagueness_score), indicators
    
    def filter_vague_claims(self, claims: List[str], threshold: float = 0.7) -> List[Tuple[str, VaguenessScore]]:
        """
        Filter out vague claims based on threshold.
        
        Args:
            claims: List of claims to filter
            threshold: Vagueness threshold (claims above this are filtered out)
            
        Returns:
            List of (claim, score) tuples for non-vague claims
        """
        filtered_claims = []
        
        for claim in claims:
            score = self.score_vagueness(claim)
            
            if score.vagueness_score <= threshold:
                filtered_claims.append((claim, score))
            else:
                logger.info(f"Filtered vague claim: '{claim}' (score: {score.vagueness_score:.2f})")
        
        logger.info(f"Filtered {len(claims) - len(filtered_claims)} vague claims from {len(claims)} total")
        return filtered_claims
    
    def batch_analyze(self, claims: List[str]) -> List[VaguenessScore]:
        """Analyze multiple claims for vagueness."""
        return [self.score_vagueness(claim) for claim in claims]


def test_vague_claim_detector():
    """Test the vague claim detector with sample claims."""
    detector = VagueClaimDetector()
    
    test_claims = [
        # Specific claims (should score low)
        "The WHO declared COVID-19 a pandemic on March 11, 2020.",
        "Apple announced the iPhone 15 on September 12, 2023.",
        "India's population reached 1.4 billion in 2023.",
        
        # Moderately vague claims
        "The government announced some new policies recently.",
        "Many experts believe that climate change is a serious issue.",
        
        # Very vague claims (should score high)
        "Some people think things are getting better.",
        "There might be issues with various aspects of the system.",
        "Certain factors could possibly influence the outcome.",
        ""  # Empty claim
    ]
    
    logger.info("Testing Vague Claim Detector...")
    
    for claim in test_claims:
        score = detector.score_vagueness(claim)
        print(f"\nClaim: '{claim}'")
        print(f"Score: {score.vagueness_score:.3f} ({score.vagueness_category})")
        print(f"Recommendation: {score.recommendation}")
        print(f"Indicators: {', '.join(score.indicators[:5])}")  # Show first 5
    
    # Test filtering
    filtered = detector.filter_vague_claims(test_claims, threshold=0.6)
    print(f"\n\nFiltered Claims ({len(filtered)} of {len(test_claims)} passed):")
    for claim, score in filtered:
        print(f"✓ '{claim}' (score: {score.vagueness_score:.3f})")


if __name__ == "__main__":
    test_vague_claim_detector()