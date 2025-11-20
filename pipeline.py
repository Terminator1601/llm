"""
Main pipeline for end-to-end fact verification using RAG.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from config import USE_LOCAL_LLM, LOCAL_LLM_CONFIG
from extract import ClaimExtractor, extract_claims
from retrieval import FactRetriever, RetrievedFact
from evaluate import LLMEvaluator, VerificationResult, Verdict
from embed import load_fact_base
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class FactVerificationResult:
    """Complete result of fact verification pipeline."""
    input_text: str
    claims: List[str]
    claim_results: List[Dict[str, Any]]
    overall_verdict: str
    processing_time: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


class FactVerificationPipeline:
    """Complete pipeline for fact verification using RAG."""
    
    def __init__(self, 
                 use_spacy: bool = True,
                 use_transformers: bool = False,
                 top_k_retrieval: int = 5,
                 use_local_llm: bool = None,
                 local_llm_config: str = None):
        """
        Initialize the fact verification pipeline.
        
        Args:
            use_spacy: Whether to use spaCy for claim extraction
            use_transformers: Whether to use transformers for claim extraction
            top_k_retrieval: Number of facts to retrieve per claim
            use_local_llm: Whether to use local LLM (overrides config default)
            local_llm_config: Local LLM configuration (overrides config default)
        """
        logger.info("Initializing fact verification pipeline")
        
        # Determine LLM configuration
        self.use_local_llm = use_local_llm if use_local_llm is not None else USE_LOCAL_LLM
        self.local_llm_config = local_llm_config if local_llm_config is not None else LOCAL_LLM_CONFIG
        
        logger.info(f"LLM Mode: {'Local LLM' if self.use_local_llm else 'OpenAI API'}")
        if self.use_local_llm:
            logger.info(f"Local LLM Config: {self.local_llm_config}")
        
        self.top_k_retrieval = top_k_retrieval
        
        # Initialize components
        try:
            self.claim_extractor = ClaimExtractor(use_spacy, use_transformers)
            self.fact_retriever = FactRetriever()
            self.llm_evaluator = LLMEvaluator(
                use_local_llm=self.use_local_llm,
                local_llm_config=self.local_llm_config
            )
            
            logger.info("Pipeline initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def verify_fact(self, text: str) -> Dict[str, Any]:
        """
        Main method to verify facts in the given text.
        
        Args:
            text: Input text containing claims to verify
            
        Returns:
            Dictionary with verification results
        """
        start_time = time.time()
        logger.info(f"Starting fact verification for text: {text[:100]}...")
        
        try:
            # Step 1: Extract claims
            logger.info("Step 1: Extracting claims")
            claims = self.claim_extractor.extract_claims(text)
            
            if not claims:
                logger.warning("No claims extracted from input text")
                return self._create_no_claims_result(text, time.time() - start_time)
            
            logger.info(f"Extracted {len(claims)} claims")
            
            # Step 2: Process each claim
            claim_results = []
            for i, claim in enumerate(claims, 1):
                logger.info(f"Step 2.{i}: Processing claim: {claim[:50]}...")
                
                # Retrieve evidence
                retrieved_facts = self.fact_retriever.retrieve_similar_facts(
                    claim, self.top_k_retrieval
                )
                
                if not retrieved_facts:
                    logger.warning(f"No evidence found for claim: {claim}")
                    claim_result = {
                        "claim": claim,
                        "verdict": Verdict.UNVERIFIABLE.value,
                        "reasoning": "No relevant evidence found in the fact base.",
                        "evidence": [],
                        "confidence_score": 0.0
                    }
                else:
                    # Evaluate claim against evidence
                    verification_result = self.llm_evaluator.evaluate_claim_against_retrieved_facts(
                        claim, retrieved_facts
                    )
                    
                    claim_result = {
                        "claim": claim,
                        "verdict": verification_result.verdict.value,
                        "reasoning": verification_result.reasoning,
                        "evidence": verification_result.evidence_used,
                        "confidence_score": verification_result.confidence_score,
                        "retrieved_facts": [rf.to_dict() for rf in retrieved_facts]
                    }
                
                claim_results.append(claim_result)
            
            # Step 3: Determine overall verdict
            overall_verdict = self._determine_overall_verdict(claim_results)
            
            processing_time = time.time() - start_time
            logger.info(f"Fact verification completed in {processing_time:.2f}s")
            
            # Create final result
            result = FactVerificationResult(
                input_text=text,
                claims=claims,
                claim_results=claim_results,
                overall_verdict=overall_verdict,
                processing_time=processing_time,
                metadata={
                    "pipeline_version": "1.0",
                    "num_claims": len(claims),
                    "retrieval_k": self.top_k_retrieval,
                    "timestamp": time.time()
                }
            )
            
            return result.to_dict()
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error in fact verification pipeline: {e}")
            
            return {
                "input_text": text,
                "claims": [],
                "claim_results": [],
                "overall_verdict": "Error",
                "processing_time": processing_time,
                "error": str(e),
                "metadata": {"error_occurred": True}
            }
    
    def _create_no_claims_result(self, text: str, processing_time: float) -> Dict[str, Any]:
        """Create result when no claims are found."""
        return {
            "input_text": text,
            "claims": [],
            "claim_results": [],
            "overall_verdict": "No verifiable claims found",
            "processing_time": processing_time,
            "metadata": {
                "pipeline_version": "1.0",
                "num_claims": 0,
                "timestamp": time.time()
            }
        }
    
    def _determine_overall_verdict(self, claim_results: List[Dict[str, Any]]) -> str:
        """
        Determine overall verdict based on individual claim results.
        
        Args:
            claim_results: List of claim verification results
            
        Returns:
            Overall verdict string
        """
        if not claim_results:
            return "No verifiable claims found"
        
        verdicts = [result["verdict"] for result in claim_results]
        
        # Count verdicts
        true_count = verdicts.count(Verdict.TRUE.value)
        false_count = verdicts.count(Verdict.FALSE.value)
        unverifiable_count = verdicts.count(Verdict.UNVERIFIABLE.value)
        
        total_claims = len(verdicts)
        
        # Determine overall verdict based on majority and severity
        if false_count > 0:
            if false_count == total_claims:
                return "All claims are false"
            else:
                return f"Mixed results: {false_count} false, {true_count} true, {unverifiable_count} unverifiable"
        
        elif true_count == total_claims:
            return "All claims are true"
        
        elif unverifiable_count == total_claims:
            return "All claims are unverifiable"
        
        else:
            return f"Mixed results: {true_count} true, {unverifiable_count} unverifiable"
    
    def verify_multiple_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Verify facts in multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of verification results
        """
        logger.info(f"Verifying facts in {len(texts)} texts")
        
        results = []
        for i, text in enumerate(texts, 1):
            logger.info(f"Processing text {i}/{len(texts)}")
            result = self.verify_fact(text)
            results.append(result)
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        fact_stats = self.fact_retriever.get_fact_statistics()
        
        return {
            "pipeline_version": "1.0",
            "components": {
                "claim_extractor": {
                    "use_spacy": self.claim_extractor.use_spacy,
                    "use_transformers": self.claim_extractor.use_transformers
                },
                "fact_retriever": {
                    "top_k": self.top_k_retrieval,
                    **fact_stats
                },
                "llm_evaluator": {
                    "model": self.llm_evaluator.model_name
                }
            }
        }


# Convenience functions for direct use
def verify_fact(text: str) -> Dict[str, Any]:
    """
    Convenience function to verify facts in text.
    
    Args:
        text: Input text to verify
        
    Returns:
        Verification results dictionary
    """
    pipeline = FactVerificationPipeline()
    return pipeline.verify_fact(text)


def create_pipeline(top_k: int = 5, 
                   use_spacy: bool = True, 
                   use_transformers: bool = False) -> FactVerificationPipeline:
    """
    Create and return a configured pipeline.
    
    Args:
        top_k: Number of facts to retrieve per claim
        use_spacy: Whether to use spaCy for claim extraction
        use_transformers: Whether to use transformers
        
    Returns:
        Configured FactVerificationPipeline
    """
    return FactVerificationPipeline(
        use_spacy=use_spacy,
        use_transformers=use_transformers,
        top_k_retrieval=top_k
    )