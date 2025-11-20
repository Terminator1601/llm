"""
Test local LLM functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from local_llm import setup_local_llm_example
from evaluate import LLMEvaluator
from pipeline import FactVerificationPipeline
from utils import get_logger

logger = get_logger(__name__)

def test_local_llm():
    """Test local LLM setup and functionality."""
    print("🧪 Testing Local LLM Support")
    print("=" * 50)
    
    # Test 1: Check basic local LLM setup
    print("\n1. Checking Local LLM Availability:")
    manager = setup_local_llm_example()
    
    # Test 2: Test LLM evaluator with local LLM
    print("\n2. Testing LLM Evaluator with Local LLM:")
    try:
        evaluator = LLMEvaluator(use_local_llm=True, local_llm_config="ollama_llama2")
        
        # Check availability
        availability = evaluator.check_local_llm_availability()
        print(f"   Available configs: {list(availability.keys())}")
        for config, available in availability.items():
            status = "✅" if available else "❌"
            print(f"   {status} {config}")
        
        # Test with fallback to OpenAI
        if not any(availability.values()):
            print("   ⚠️ No local LLMs available, testing fallback to OpenAI")
            result = evaluator.evaluate_claim_against_evidence(
                "Test claim", 
                ["Test evidence"]
            )
            print(f"   Fallback test result: {result.verdict}")
    except Exception as e:
        print(f"   ❌ Error testing LLM evaluator: {e}")
    
    # Test 3: Test pipeline with local LLM configuration
    print("\n3. Testing Pipeline with Local LLM Configuration:")
    try:
        pipeline = FactVerificationPipeline(use_local_llm=True, local_llm_config="ollama_llama2")
        print("   ✅ Pipeline initialized with local LLM support")
        
        # Test verification with simple text
        test_text = "The Earth is round."
        result = pipeline.verify_fact(test_text)
        print(f"   Test verification completed: {len(result['verification_results'])} claims processed")
    except Exception as e:
        print(f"   ❌ Error testing pipeline: {e}")
    
    # Test 4: Environment variable configuration
    print("\n4. Testing Environment Variable Configuration:")
    print(f"   USE_LOCAL_LLM: {os.getenv('USE_LOCAL_LLM', 'not set')}")
    print(f"   LOCAL_LLM_CONFIG: {os.getenv('LOCAL_LLM_CONFIG', 'not set')}")
    
    # Instructions
    print("\n5. Setup Instructions for Local LLM:")
    print("   To use local LLMs, follow these steps:")
    print("   ")
    print("   For Ollama:")
    print("   1. Install Ollama from https://ollama.ai/")
    print("   2. Pull a model: ollama pull llama2")
    print("   3. Verify service: ollama list")
    print("   4. Set environment variables:")
    print("      - USE_LOCAL_LLM=true")
    print("      - LOCAL_LLM_CONFIG=ollama_llama2")
    print("   ")
    print("   For llama.cpp server:")
    print("   1. Build and run llama.cpp server")
    print("   2. Start server: ./server -m model.gguf -c 4096 --port 8080")
    print("   3. Use config: llamacpp")
    print("   ")
    print("   Available configurations:")
    configs = evaluator.list_local_llm_configs() if 'evaluator' in locals() else []
    for config in configs:
        print(f"   - {config}")

if __name__ == "__main__":
    test_local_llm()