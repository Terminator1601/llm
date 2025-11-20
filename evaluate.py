"""
LLM evaluation module for fact verification using language models.
"""

import json
import openai
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from config import LLM_MODEL, OPENAI_API_KEY, MAX_TOKENS, TEMPERATURE
from retrieval import RetrievedFact
from utils import get_logger

logger = get_logger(__name__)

# Import local LLM support
try:
    from local_llm import LocalLLMManager, LocalLLMConfig
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    logger.warning("Local LLM support not available")


class Verdict(Enum):
    """Possible verdicts for fact verification."""
    TRUE = "True"
    FALSE = "False"
    UNVERIFIABLE = "Unverifiable"


@dataclass
class VerificationResult:
    """Result of fact verification."""
    verdict: Verdict
    reasoning: str
    evidence_used: List[str]
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "verdict": self.verdict.value,
            "reasoning": self.reasoning,
            "evidence": self.evidence_used,
            "confidence_score": self.confidence_score
        }


class LLMEvaluator:
    """Evaluate claims against evidence using language models."""
    
    def __init__(self, model_name: str = LLM_MODEL, api_key: str = None, 
                 use_local_llm: bool = False, local_llm_config: str = "ollama_llama2"):
        """
        Initialize the LLM evaluator.
        
        Args:
            model_name: Name of the LLM model to use
            api_key: OpenAI API key (optional, will use config default)
            use_local_llm: Whether to use local LLM instead of OpenAI
            local_llm_config: Local LLM configuration name to use
        """
        self.model_name = model_name
        self.api_key = api_key or OPENAI_API_KEY
        self.use_local_llm = use_local_llm
        self.local_llm_config = local_llm_config
        self.local_llm_manager = None
        self.local_llm_evaluator = None
        
        # Initialize local LLM if requested
        if use_local_llm and LOCAL_LLM_AVAILABLE:
            try:
                self.local_llm_manager = LocalLLMManager()
                self.local_llm_evaluator = self.local_llm_manager.get_evaluator(local_llm_config)
                if self.local_llm_evaluator:
                    logger.info(f"Using local LLM: {local_llm_config}")
                else:
                    logger.warning(f"Failed to initialize local LLM {local_llm_config}, falling back to OpenAI")
                    self.use_local_llm = False
            except Exception as e:
                logger.error(f"Local LLM setup failed: {e}, falling back to OpenAI")
                self.use_local_llm = False
        elif use_local_llm:
            logger.warning("Local LLM requested but not available, falling back to OpenAI")
            self.use_local_llm = False
        
        # Initialize OpenAI if not using local LLM
        if not self.use_local_llm:
            if not self.api_key:
                logger.warning("No OpenAI API key provided. LLM evaluation will not work.")
            else:
                openai.api_key = self.api_key
                logger.info(f"LLM evaluator initialized with OpenAI model: {model_name}")
    
    def evaluate_claim_against_evidence(
        self, 
        claim: str, 
        evidence: List[str]
    ) -> VerificationResult:
        """
        Evaluate a claim against provided evidence.
        
        Args:
            claim: The claim to verify
            evidence: List of evidence statements
            
        Returns:
            VerificationResult with verdict, reasoning, and evidence used
        """
        logger.info(f"Evaluating claim: {claim[:100]}...")
        
        if not evidence:
            logger.warning("No evidence provided for claim evaluation")
            return VerificationResult(
                verdict=Verdict.UNVERIFIABLE,
                reasoning="No evidence available to verify this claim.",
                evidence_used=[]
            )
        
        # Use local LLM if configured
        if self.use_local_llm and self.local_llm_evaluator:
            try:
                result_dict = self.local_llm_evaluator.evaluate_claim_against_evidence(claim, evidence)
                
                # Convert to VerificationResult
                verdict_map = {
                    "True": Verdict.TRUE,
                    "False": Verdict.FALSE,
                    "Unverifiable": Verdict.UNVERIFIABLE
                }
                
                return VerificationResult(
                    verdict=verdict_map.get(result_dict["verdict"], Verdict.UNVERIFIABLE),
                    reasoning=result_dict["reasoning"],
                    evidence_used=result_dict.get("evidence_used", evidence),
                    confidence_score=result_dict.get("confidence_score", 0.5)
                )
            except Exception as e:
                logger.error(f"Local LLM evaluation failed, falling back to OpenAI: {e}")
                # Continue with OpenAI evaluation
        
        # Use OpenAI API
        return self._evaluate_with_openai(claim, evidence)
    
    def _evaluate_with_openai(self, claim: str, evidence: List[str]) -> VerificationResult:
        """Evaluate using OpenAI API."""
        
        # Create the evaluation prompt
        prompt = self._create_evaluation_prompt(claim, evidence)
        
        try:
            # Call LLM
            response = self._call_llm(prompt)
            
            # Parse response
            result = self._parse_llm_response(response, evidence)
            
            logger.info(f"OpenAI evaluation completed: {result.verdict.value}")
            return result
        
        except Exception as e:
            logger.error(f"Error during OpenAI evaluation: {e}")
            return VerificationResult(
                verdict=Verdict.UNVERIFIABLE,
                reasoning=f"Error during evaluation: {str(e)}",
                evidence_used=[]
            )
    
    def check_local_llm_availability(self) -> Dict[str, bool]:
        """Check availability of local LLMs."""
        if not LOCAL_LLM_AVAILABLE:
            return {}
        
        if not self.local_llm_manager:
            manager = LocalLLMManager()
        else:
            manager = self.local_llm_manager
            
        return manager.check_availability()
    
    def list_local_llm_configs(self) -> List[str]:
        """List available local LLM configurations."""
        if not LOCAL_LLM_AVAILABLE:
            return []
        
        if not self.local_llm_manager:
            manager = LocalLLMManager()
        else:
            manager = self.local_llm_manager
            
        return manager.list_available_configs()
    
    def evaluate_claim_against_retrieved_facts(
        self, 
        claim: str, 
        retrieved_facts: List[RetrievedFact]
    ) -> VerificationResult:
        """
        Evaluate a claim against retrieved facts.
        
        Args:
            claim: The claim to verify
            retrieved_facts: List of RetrievedFact objects
            
        Returns:
            VerificationResult
        """
        evidence = [rf.text for rf in retrieved_facts]
        return self.evaluate_claim_against_evidence(claim, evidence)
    
    def _create_evaluation_prompt(self, claim: str, evidence: List[str]) -> str:
        """Create the prompt for LLM evaluation."""
        evidence_text = "\n".join([f"{i+1}. {ev}" for i, ev in enumerate(evidence)])
        
        prompt = f"""You are a professional fact-checking assistant. Your task is to evaluate whether a claim is true, false, or unverifiable based on the provided evidence.

CLAIM TO VERIFY:
{claim}

AVAILABLE EVIDENCE:
{evidence_text}

INSTRUCTIONS:
1. Carefully analyze the claim and compare it with each piece of evidence
2. Consider the credibility and relevance of the evidence
3. Look for contradictions, confirmations, or insufficient information
4. Classify the claim as TRUE, FALSE, or UNVERIFIABLE
5. Provide clear reasoning for your decision

CLASSIFICATION CRITERIA:
- TRUE: The evidence clearly supports and confirms the claim
- FALSE: The evidence clearly contradicts or refutes the claim  
- UNVERIFIABLE: Insufficient or conflicting evidence to make a determination

Respond in the following JSON format:
{{
    "verdict": "True|False|Unverifiable",
    "reasoning": "Detailed explanation of your decision based on the evidence",
    "evidence_used": ["List of evidence statements that were most relevant"],
    "confidence_score": 0.95
}}

Ensure your response is valid JSON and includes all required fields."""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        if not self.api_key:
            raise ValueError("No OpenAI API key available")
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional fact-checking assistant. Always respond in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _parse_llm_response(self, response: str, evidence: List[str]) -> VerificationResult:
        """Parse the LLM response into a VerificationResult."""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            
            # Extract verdict
            verdict_str = data.get("verdict", "Unverifiable").lower()
            if "true" in verdict_str:
                verdict = Verdict.TRUE
            elif "false" in verdict_str:
                verdict = Verdict.FALSE
            else:
                verdict = Verdict.UNVERIFIABLE
            
            # Extract other fields
            reasoning = data.get("reasoning", "No reasoning provided")
            evidence_used = data.get("evidence_used", evidence[:3])  # Fallback to first 3
            confidence_score = float(data.get("confidence_score", 0.0))
            
            return VerificationResult(
                verdict=verdict,
                reasoning=reasoning,
                evidence_used=evidence_used,
                confidence_score=confidence_score
            )
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            
            # Fallback parsing
            response_lower = response.lower()
            
            if "true" in response_lower and "false" not in response_lower:
                verdict = Verdict.TRUE
            elif "false" in response_lower:
                verdict = Verdict.FALSE
            else:
                verdict = Verdict.UNVERIFIABLE
            
            return VerificationResult(
                verdict=verdict,
                reasoning=response[:500],  # Truncate if too long
                evidence_used=evidence[:3],
                confidence_score=0.5  # Default moderate confidence
            )
    
    def batch_evaluate(
        self, 
        claims_evidence: List[tuple]
    ) -> List[VerificationResult]:
        """
        Evaluate multiple claims in batch.
        
        Args:
            claims_evidence: List of (claim, evidence_list) tuples
            
        Returns:
            List of VerificationResult objects
        """
        logger.info(f"Batch evaluating {len(claims_evidence)} claims")
        
        results = []
        for claim, evidence in claims_evidence:
            result = self.evaluate_claim_against_evidence(claim, evidence)
            results.append(result)
        
        return results


class LocalLLMEvaluator(LLMEvaluator):
    """Evaluator using local language models (placeholder for future implementation)."""
    
    def __init__(self, model_name: str = "mistral-7b"):
        """
        Initialize local LLM evaluator.
        
        Args:
            model_name: Name of the local model
        """
        self.model_name = model_name
        logger.warning("LocalLLMEvaluator is not implemented yet. Use LLMEvaluator instead.")
    
    def _call_llm(self, prompt: str) -> str:
        """Placeholder for local LLM implementation."""
        raise NotImplementedError("Local LLM evaluation not implemented yet")


def evaluate_claim_against_evidence(claim: str, evidence: List[str]) -> Dict[str, Any]:
    """
    Convenience function for evaluating a claim against evidence.
    
    Args:
        claim: The claim to verify
        evidence: List of evidence statements
        
    Returns:
        Dictionary with verification results
    """
    evaluator = LLMEvaluator()
    result = evaluator.evaluate_claim_against_evidence(claim, evidence)
    return result.to_dict()