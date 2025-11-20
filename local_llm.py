"""
Local LLM support for fact verification using Ollama and other local models.
"""

import requests
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

from utils import get_logger

logger = get_logger(__name__)


@dataclass 
class LocalLLMConfig:
    """Configuration for local LLM."""
    provider: str  # "ollama", "llamacpp", "huggingface"
    model_name: str
    base_url: str
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


class LocalLLMEvaluator:
    """Evaluate claims using local LLM models."""
    
    def __init__(self, config: LocalLLMConfig):
        """
        Initialize local LLM evaluator.
        
        Args:
            config: Local LLM configuration
        """
        self.config = config
        self.session = requests.Session()
        
        # Set up headers based on provider
        if config.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {config.api_key}"
            })
        
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        
        logger.info(f"Initialized LocalLLMEvaluator with {config.provider} - {config.model_name}")
    
    def evaluate_claim_against_evidence(self, 
                                      claim: str, 
                                      evidence: List[str]) -> Dict[str, Any]:
        """
        Evaluate claim against evidence using local LLM.
        
        Args:
            claim: Claim to evaluate
            evidence: List of evidence strings
            
        Returns:
            Evaluation result dictionary
        """
        try:
            # Prepare evidence text
            evidence_text = "\n".join([f"- {ev}" for ev in evidence])
            
            # Create evaluation prompt
            prompt = self._create_evaluation_prompt(claim, evidence_text)
            
            # Call local LLM
            response = self._call_local_llm(prompt)
            
            # Parse response
            result = self._parse_llm_response(response)
            
            logger.info(f"Local LLM evaluation completed for claim: {claim[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Local LLM evaluation failed: {e}")
            return {
                "verdict": "Unverifiable",
                "reasoning": f"Evaluation failed due to error: {str(e)}",
                "evidence": evidence,
                "confidence_score": 0.0
            }
    
    def _create_evaluation_prompt(self, claim: str, evidence: str) -> str:
        """Create prompt for fact verification."""
        prompt = f"""You are a professional fact-checking assistant. Evaluate the following claim against the provided evidence.

CLAIM TO EVALUATE:
{claim}

AVAILABLE EVIDENCE:
{evidence}

TASK:
Analyze whether the claim is supported, contradicted, or cannot be determined from the evidence.

RESPONSE FORMAT:
Please respond with a valid JSON object containing:
- "verdict": "True" (supported by evidence), "False" (contradicted by evidence), or "Unverifiable" (insufficient evidence)
- "reasoning": Detailed explanation of your analysis (2-3 sentences)
- "confidence_score": Number between 0.0 and 1.0 indicating confidence
- "evidence_used": List of specific evidence points that influenced your decision

Example response:
{{
    "verdict": "True",
    "reasoning": "The claim is directly supported by the evidence which confirms the specific details.",
    "confidence_score": 0.9,
    "evidence_used": ["Evidence point 1", "Evidence point 2"]
}}

Your response:"""
        return prompt
    
    def _call_local_llm(self, prompt: str) -> str:
        """Call local LLM based on provider."""
        if self.config.provider.lower() == "ollama":
            return self._call_ollama(prompt)
        elif self.config.provider.lower() == "llamacpp":
            return self._call_llamacpp(prompt)
        elif self.config.provider.lower() == "huggingface":
            return self._call_huggingface(prompt)
        else:
            raise ValueError(f"Unsupported local LLM provider: {self.config.provider}")
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API."""
        url = f"{self.config.base_url}/api/generate"
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        response = self.session.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    
    def _call_llamacpp(self, prompt: str) -> str:
        """Call llama.cpp server API."""
        url = f"{self.config.base_url}/completion"
        
        payload = {
            "prompt": prompt,
            "n_predict": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stop": ["</s>", "\n\n\n"],
            "stream": False
        }
        
        response = self.session.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result.get("content", "")
    
    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API."""
        url = f"{self.config.base_url}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "return_full_text": False
            }
        }
        
        response = self.session.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return ""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        if not response:
            return {
                "verdict": "Unverifiable",
                "reasoning": "No response from local LLM",
                "confidence_score": 0.0,
                "evidence_used": []
            }
        
        try:
            # Try to extract JSON from response
            response = response.strip()
            
            # Look for JSON object in response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Validate required fields
                if "verdict" in result and "reasoning" in result:
                    return {
                        "verdict": result.get("verdict", "Unverifiable"),
                        "reasoning": result.get("reasoning", "No reasoning provided"),
                        "confidence_score": float(result.get("confidence_score", 0.5)),
                        "evidence_used": result.get("evidence_used", [])
                    }
            
            # Fallback: try to extract verdict from text
            response_lower = response.lower()
            if "true" in response_lower and "false" not in response_lower:
                verdict = "True"
            elif "false" in response_lower:
                verdict = "False"
            else:
                verdict = "Unverifiable"
            
            return {
                "verdict": verdict,
                "reasoning": response[:200] + "..." if len(response) > 200 else response,
                "confidence_score": 0.5,
                "evidence_used": []
            }
            
        except Exception as e:
            logger.error(f"Failed to parse local LLM response: {e}")
            return {
                "verdict": "Unverifiable",
                "reasoning": f"Failed to parse response: {response[:100]}...",
                "confidence_score": 0.0,
                "evidence_used": []
            }


class LocalLLMManager:
    """Manager for local LLM configurations and instances."""
    
    def __init__(self):
        """Initialize local LLM manager."""
        self.configs = {}
        self.evaluators = {}
        
        # Add default configurations
        self._add_default_configs()
    
    def _add_default_configs(self):
        """Add default local LLM configurations."""
        # Ollama configurations
        self.configs["ollama_llama2"] = LocalLLMConfig(
            provider="ollama",
            model_name="llama2",
            base_url="http://localhost:11434"
        )
        
        self.configs["ollama_llama3"] = LocalLLMConfig(
            provider="ollama", 
            model_name="llama3",
            base_url="http://localhost:11434"
        )
        
        self.configs["ollama_mistral"] = LocalLLMConfig(
            provider="ollama",
            model_name="mistral",
            base_url="http://localhost:11434"
        )
        
        # llama.cpp server configuration
        self.configs["llamacpp"] = LocalLLMConfig(
            provider="llamacpp",
            model_name="local_model",
            base_url="http://localhost:8080"
        )
    
    def add_config(self, name: str, config: LocalLLMConfig):
        """Add a local LLM configuration."""
        self.configs[name] = config
        logger.info(f"Added local LLM config: {name}")
    
    def get_evaluator(self, config_name: str) -> Optional[LocalLLMEvaluator]:
        """Get or create local LLM evaluator."""
        if config_name not in self.configs:
            logger.error(f"Unknown local LLM config: {config_name}")
            return None
        
        if config_name not in self.evaluators:
            try:
                config = self.configs[config_name]
                evaluator = LocalLLMEvaluator(config)
                
                # Test connection
                if self._test_connection(evaluator):
                    self.evaluators[config_name] = evaluator
                    logger.info(f"Created and validated local LLM evaluator: {config_name}")
                else:
                    logger.warning(f"Failed to connect to local LLM: {config_name}")
                    return None
                    
            except Exception as e:
                logger.error(f"Failed to create local LLM evaluator {config_name}: {e}")
                return None
        
        return self.evaluators.get(config_name)
    
    def _test_connection(self, evaluator: LocalLLMEvaluator) -> bool:
        """Test connection to local LLM."""
        try:
            # Simple test prompt
            test_response = evaluator._call_local_llm("Test: respond with 'OK'")
            return bool(test_response and len(test_response.strip()) > 0)
        except Exception as e:
            logger.error(f"Local LLM connection test failed: {e}")
            return False
    
    def list_available_configs(self) -> List[str]:
        """List available local LLM configurations."""
        return list(self.configs.keys())
    
    def check_availability(self) -> Dict[str, bool]:
        """Check availability of all configured local LLMs."""
        availability = {}
        
        for name, config in self.configs.items():
            try:
                evaluator = LocalLLMEvaluator(config)
                availability[name] = self._test_connection(evaluator)
            except Exception:
                availability[name] = False
        
        return availability


def setup_local_llm_example():
    """Example setup for local LLM integration."""
    manager = LocalLLMManager()
    
    # Check availability
    availability = manager.check_availability()
    
    print("Local LLM Availability Check:")
    for name, is_available in availability.items():
        status = "✅ Available" if is_available else "❌ Not available"
        print(f"  {name}: {status}")
    
    # Try to get an available evaluator
    available_configs = [name for name, available in availability.items() if available]
    
    if available_configs:
        config_name = available_configs[0]
        evaluator = manager.get_evaluator(config_name)
        
        if evaluator:
            print(f"\n✅ Successfully connected to {config_name}")
            
            # Test evaluation
            test_claim = "The Earth is round"
            test_evidence = ["Scientific observations confirm Earth's spherical shape"]
            
            result = evaluator.evaluate_claim_against_evidence(test_claim, test_evidence)
            print(f"\nTest evaluation result:")
            print(f"  Claim: {test_claim}")
            print(f"  Verdict: {result['verdict']}")
            print(f"  Reasoning: {result['reasoning']}")
    else:
        print("\n❌ No local LLMs are currently available")
        print("\nTo use local LLMs:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Pull a model: ollama pull llama2")
        print("3. Start Ollama service")
        print("4. Or set up llama.cpp server")
    
    return manager


if __name__ == "__main__":
    setup_local_llm_example()