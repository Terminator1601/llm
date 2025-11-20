"""
Simple test for local LLM functionality without heavy dependencies
"""

import os
import sys
import requests
import json
from typing import Dict, Any

# Simple local LLM test without importing full pipeline
def test_local_llm_connection():
    """Test direct connection to local LLM services."""
    print("🧪 Testing Local LLM Connections")
    print("=" * 50)
    
    # Test configurations
    configs = [
        {
            "name": "Ollama (llama2)",
            "url": "http://localhost:11434/api/generate",
            "payload": {
                "model": "llama2",
                "prompt": "Test: respond with 'OK'",
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 50}
            }
        },
        {
            "name": "Ollama (llama3)",
            "url": "http://localhost:11434/api/generate", 
            "payload": {
                "model": "llama3",
                "prompt": "Test: respond with 'OK'",
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 50}
            }
        },
        {
            "name": "llama.cpp server",
            "url": "http://localhost:8080/completion",
            "payload": {
                "prompt": "Test: respond with 'OK'",
                "n_predict": 50,
                "temperature": 0.1,
                "stream": False
            }
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n🔌 Testing {config['name']}:")
        try:
            response = requests.post(
                config["url"], 
                json=config["payload"], 
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                if "ollama" in config["url"]:
                    response_text = result.get("response", "")
                else:  # llama.cpp
                    response_text = result.get("content", "")
                
                if response_text:
                    print(f"   ✅ Connected successfully")
                    print(f"   📝 Response: {response_text[:100]}...")
                    results[config["name"]] = True
                else:
                    print(f"   ⚠️ Connected but no response content")
                    results[config["name"]] = False
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.text[:100]}...")
                results[config["name"]] = False
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection refused (service not running)")
            results[config["name"]] = False
        except requests.exceptions.Timeout:
            print(f"   ❌ Request timed out")
            results[config["name"]] = False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[config["name"]] = False
    
    # Summary
    print(f"\n📊 Connection Summary:")
    available_services = [name for name, available in results.items() if available]
    unavailable_services = [name for name, available in results.items() if not available]
    
    if available_services:
        print(f"   ✅ Available: {', '.join(available_services)}")
    
    if unavailable_services:
        print(f"   ❌ Unavailable: {', '.join(unavailable_services)}")
    
    return results


def test_local_llm_evaluation():
    """Test a simple fact evaluation using local LLM."""
    print("\n🔬 Testing Local LLM Fact Evaluation")
    print("=" * 50)
    
    # Test if we can import the local LLM module
    try:
        # Import our local LLM classes
        from local_llm import LocalLLMManager, LocalLLMConfig, LocalLLMEvaluator
        
        print("✅ Local LLM module imported successfully")
        
        # Create manager and check availability
        manager = LocalLLMManager()
        availability = manager.check_availability()
        
        print(f"\n📋 Available Configurations:")
        for name, available in availability.items():
            status = "✅" if available else "❌"
            print(f"   {status} {name}")
        
        # Try to get a working evaluator
        available_configs = [name for name, available in availability.items() if available]
        
        if available_configs:
            config_name = available_configs[0]
            print(f"\n🧪 Testing evaluation with {config_name}:")
            
            evaluator = manager.get_evaluator(config_name)
            if evaluator:
                # Test fact evaluation
                test_claim = "The Earth is approximately spherical in shape"
                test_evidence = [
                    "Scientific observations confirm Earth's spherical shape",
                    "Satellite images show Earth as a sphere",
                    "The curvature of Earth is visible from high altitudes"
                ]
                
                print(f"   📝 Test claim: {test_claim}")
                print(f"   📚 Test evidence: {len(test_evidence)} items")
                
                result = evaluator.evaluate_claim_against_evidence(test_claim, test_evidence)
                
                print(f"   📊 Evaluation result:")
                print(f"      Verdict: {result['verdict']}")
                print(f"      Confidence: {result['confidence_score']:.2f}")
                print(f"      Reasoning: {result['reasoning'][:150]}...")
                
                return True
            else:
                print(f"   ❌ Failed to create evaluator for {config_name}")
                return False
        else:
            print("   ⚠️ No local LLMs available for testing")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import local LLM module: {e}")
        return False
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False


def print_setup_instructions():
    """Print setup instructions for local LLMs."""
    print("\n📋 Local LLM Setup Instructions")
    print("=" * 50)
    
    print("\n🔧 Option 1: Ollama (Recommended)")
    print("   1. Download and install Ollama:")
    print("      • Windows: https://ollama.ai/download/windows")
    print("      • macOS: https://ollama.ai/download/mac")
    print("      • Linux: curl -fsSL https://ollama.ai/install.sh | sh")
    print("   ")
    print("   2. Pull a model:")
    print("      ollama pull llama2        # ~3.8GB")
    print("      # OR")
    print("      ollama pull llama3        # ~4.7GB")
    print("      # OR") 
    print("      ollama pull mistral       # ~4.1GB")
    print("   ")
    print("   3. Verify installation:")
    print("      ollama list")
    print("      ollama run llama2 \"Hello!\"")
    
    print("\n🔧 Option 2: llama.cpp Server")
    print("   1. Build llama.cpp:")
    print("      git clone https://github.com/ggerganov/llama.cpp")
    print("      cd llama.cpp && make")
    print("   ")
    print("   2. Download a GGUF model (e.g., from Hugging Face)")
    print("   ")
    print("   3. Start server:")
    print("      ./server -m model.gguf -c 4096 --port 8080")
    
    print("\n🔧 Using with Fact Verification System")
    print("   1. Set environment variables:")
    print("      USE_LOCAL_LLM=true")
    print("      LOCAL_LLM_CONFIG=ollama_llama2  # or ollama_llama3, ollama_mistral, llamacpp")
    print("   ")
    print("   2. Or pass parameters directly:")
    print("      pipeline = FactVerificationPipeline(")
    print("          use_local_llm=True,")
    print("          local_llm_config='ollama_llama2'")
    print("      )")
    
    print("\n💡 Tips:")
    print("   • Local LLMs require significant RAM (8GB+ recommended)")
    print("   • Performance varies by model size and hardware")
    print("   • The system falls back to OpenAI API if local LLMs fail")
    print("   • Test connection with this script before using in main app")


def main():
    """Run all local LLM tests."""
    print("🚀 Local LLM Testing Suite")
    print("=" * 60)
    
    # Test 1: Direct connections
    connections = test_local_llm_connection()
    
    # Test 2: Full evaluation (if possible)
    if any(connections.values()):
        evaluation_success = test_local_llm_evaluation()
        
        if evaluation_success:
            print(f"\n🎉 Local LLM testing completed successfully!")
            print(f"   You can now use local LLMs in the fact verification system.")
        else:
            print(f"\n⚠️ Connection successful but evaluation failed.")
            print(f"   Check the local LLM module for issues.")
    else:
        print(f"\n⚠️ No local LLM services detected.")
        print(f"   The system will use OpenAI API as fallback.")
    
    # Always show setup instructions
    print_setup_instructions()


if __name__ == "__main__":
    main()