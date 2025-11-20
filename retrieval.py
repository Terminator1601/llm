"""
Fact retrieval module for finding relevant evidence for claims.
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from embed import Fact, FactBase, load_fact_base
from config import DEFAULT_TOP_K, SIMILARITY_THRESHOLD
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedFact:
    """Represents a retrieved fact with similarity score and metadata."""
    fact: Fact
    similarity_score: float
    relevance_score: float = 0.0
    
    @property
    def text(self) -> str:
        """Get the fact text."""
        return self.fact.text
    
    @property
    def id(self) -> str:
        """Get the fact ID."""
        return self.fact.id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "similarity_score": self.similarity_score,
            "relevance_score": self.relevance_score,
            "metadata": self.fact.metadata
        }


class FactRetriever:
    """Retrieve relevant facts for given claims."""
    
    def __init__(self, fact_base: FactBase = None):
        """
        Initialize the fact retriever.
        
        Args:
            fact_base: Pre-loaded FactBase instance (optional)
        """
        self.fact_base = fact_base or load_fact_base()
        logger.info("FactRetriever initialized")
    
    def retrieve_similar_facts(self, claim: str, top_k: int = DEFAULT_TOP_K) -> List[RetrievedFact]:
        """
        Retrieve facts similar to the given claim.
        
        Args:
            claim: The claim to find similar facts for
            top_k: Number of top facts to retrieve
            
        Returns:
            List of RetrievedFact objects, sorted by similarity
        """
        logger.info(f"Retrieving similar facts for claim: {claim[:100]}...")
        
        # Search in vector store
        results = self.fact_base.search_similar_facts(claim, top_k)
        
        if not results:
            logger.warning("No similar facts found")
            return []
        
        # Convert to RetrievedFact objects
        retrieved_facts = []
        for fact, distance in results:
            # Convert distance to similarity score (lower distance = higher similarity)
            similarity_score = max(0.0, 1.0 - (distance / 10.0))  # Normalize roughly
            
            retrieved_fact = RetrievedFact(
                fact=fact,
                similarity_score=similarity_score
            )
            
            # Apply relevance scoring
            retrieved_fact.relevance_score = self._calculate_relevance(claim, fact.text)
            
            retrieved_facts.append(retrieved_fact)
        
        # Sort by combined similarity and relevance scores
        retrieved_facts.sort(
            key=lambda x: (x.similarity_score + x.relevance_score) / 2,
            reverse=True
        )
        
        # Filter by similarity threshold
        filtered_facts = [
            rf for rf in retrieved_facts 
            if rf.similarity_score >= SIMILARITY_THRESHOLD
        ]
        
        logger.info(f"Retrieved {len(filtered_facts)} relevant facts (filtered from {len(retrieved_facts)})")
        return filtered_facts
    
    def _calculate_relevance(self, claim: str, fact_text: str) -> float:
        """
        Calculate relevance score between claim and fact.
        
        Args:
            claim: The input claim
            fact_text: The fact text to compare against
            
        Returns:
            Relevance score between 0 and 1
        """
        # Simple keyword-based relevance scoring
        claim_words = set(claim.lower().split())
        fact_words = set(fact_text.lower().split())
        
        if not claim_words or not fact_words:
            return 0.0
        
        # Calculate overlap
        common_words = claim_words & fact_words
        total_words = claim_words | fact_words
        
        # Basic Jaccard similarity
        jaccard_score = len(common_words) / len(total_words) if total_words else 0.0
        
        # Boost score for exact phrase matches
        claim_lower = claim.lower()
        fact_lower = fact_text.lower()
        
        # Look for common phrases (3+ words)
        phrase_boost = 0.0
        claim_phrases = self._extract_phrases(claim_lower, min_length=3)
        fact_phrases = self._extract_phrases(fact_lower, min_length=3)
        
        for phrase in claim_phrases:
            if any(phrase in fact_phrase for fact_phrase in fact_phrases):
                phrase_boost += 0.2
        
        # Boost for named entities (simple pattern matching)
        entity_boost = 0.0
        entity_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper names
            r'\b\d{4}\b',  # Years
            r'\b\d+%\b',   # Percentages
            r'\$\d+',      # Dollar amounts
        ]
        
        import re
        for pattern in entity_patterns:
            claim_entities = set(re.findall(pattern, claim))
            fact_entities = set(re.findall(pattern, fact_text))
            
            if claim_entities & fact_entities:
                entity_boost += 0.1
        
        # Combine scores
        relevance_score = min(1.0, jaccard_score + phrase_boost + entity_boost)
        
        return relevance_score
    
    def _extract_phrases(self, text: str, min_length: int = 3) -> List[str]:
        """Extract n-gram phrases from text."""
        words = text.split()
        phrases = []
        
        for i in range(len(words) - min_length + 1):
            phrase = ' '.join(words[i:i + min_length])
            phrases.append(phrase)
        
        return phrases
    
    def retrieve_multiple_claims(self, claims: List[str], top_k: int = DEFAULT_TOP_K) -> Dict[str, List[RetrievedFact]]:
        """
        Retrieve facts for multiple claims.
        
        Args:
            claims: List of claims to retrieve facts for
            top_k: Number of top facts per claim
            
        Returns:
            Dictionary mapping claims to their retrieved facts
        """
        logger.info(f"Retrieving facts for {len(claims)} claims")
        
        results = {}
        for claim in claims:
            retrieved_facts = self.retrieve_similar_facts(claim, top_k)
            results[claim] = retrieved_facts
        
        return results
    
    def get_fact_statistics(self) -> Dict[str, Any]:
        """Get statistics about the fact base."""
        total_facts = len(self.fact_base.vector_store.facts)
        
        return {
            "total_facts": total_facts,
            "embedding_dimension": self.fact_base.vector_store.embedding_dim,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "default_top_k": DEFAULT_TOP_K
        }


def retrieve_similar_facts(claim: str, top_k: int = DEFAULT_TOP_K) -> List[Fact]:
    """
    Convenience function to retrieve similar facts for a claim.
    
    Args:
        claim: The claim to find similar facts for
        top_k: Number of top facts to retrieve
        
    Returns:
        List of Fact objects
    """
    retriever = FactRetriever()
    retrieved_facts = retriever.retrieve_similar_facts(claim, top_k)
    
    # Return just the Fact objects for compatibility
    return [rf.fact for rf in retrieved_facts]