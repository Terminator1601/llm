"""
Simple test script for basic functionality without OpenAI dependency.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_claim_extraction():
    """Test claim extraction without LLM."""
    print("Testing Claim Extraction...")
    try:
        from extract import extract_claims
        
        test_text = "The WHO declared COVID-19 a pandemic on March 11, 2020."
        claims = extract_claims(test_text)
        
        print(f"✅ Claims extracted: {claims}")
        return True
    except Exception as e:
        print(f"❌ Claim extraction failed: {e}")
        return False

def test_embeddings():
    """Test embedding system."""
    print("Testing Embeddings...")
    try:
        from embed import load_fact_base
        
        fact_base = load_fact_base()
        print("✅ Embeddings system working")
        print(f"   Total facts loaded: {len(fact_base.vector_store.facts)}")
        return True
    except Exception as e:
        print(f"❌ Embeddings failed: {e}")
        return False

def test_retrieval():
    """Test retrieval system."""
    print("Testing Fact Retrieval...")
    try:
        from retrieval import FactRetriever
        
        retriever = FactRetriever()
        results = retriever.retrieve_similar_facts("COVID-19 pandemic", top_k=3)
        
        print("✅ Retrieval system working")
        print(f"   Retrieved {len(results)} facts")
        
        if results:
            print(f"   Top result: {results[0].text[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("🧪 SIMPLE FACT VERIFICATION TESTS")
    print("=" * 50)
    
    tests = [
        test_claim_extraction,
        test_embeddings, 
        test_retrieval
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append(False)
        print()
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All core systems working!")
        print("\nTo test with OpenAI (requires API key):")
        print("  py main.py 'COVID-19 was declared a pandemic in March 2020'")
        print("\nTo run the web interface:")
        print("  streamlit run app.py")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()