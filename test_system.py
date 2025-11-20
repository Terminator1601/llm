"""
Example usage and testing script for the fact verification system.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline import verify_fact, create_pipeline
from extract import extract_claims
from embed import embed_facts
from utils import get_logger

logger = get_logger(__name__)


def test_claim_extraction():
    """Test the claim extraction functionality."""
    print("=" * 50)
    print("TESTING CLAIM EXTRACTION")
    print("=" * 50)
    
    test_texts = [
        "The Indian government has announced free electricity to all farmers starting July 2025.",
        "I think the weather is nice today. The Earth's temperature has increased by 1.1 degrees Celsius since pre-industrial times.",
        "Apple announced the iPhone 15 in September 2023. I believe it's a great phone.",
        "COVID-19 vaccines contain microchips for tracking people. This is absolutely true!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        claims = extract_claims(text)
        print(f"Extracted claims: {claims}")


def test_embedding_system():
    """Test the embedding and vector store system."""
    print("=" * 50)
    print("TESTING EMBEDDING SYSTEM")
    print("=" * 50)
    
    try:
        print("Setting up embeddings...")
        success = embed_facts()
        if success:
            print("✅ Embeddings created successfully!")
        else:
            print("❌ Failed to create embeddings")
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")


def test_full_pipeline():
    """Test the complete fact verification pipeline."""
    print("=" * 50)
    print("TESTING FULL PIPELINE")
    print("=" * 50)
    
    test_statements = [
        "The WHO declared COVID-19 a pandemic on March 11, 2020.",
        "The Earth is flat according to scientific consensus.",
        "Apple announced the iPhone 15 with USB-C in September 2023.",
        "COVID-19 vaccines contain tracking microchips.",
        "The global temperature has increased by about 1.1 degrees since pre-industrial times."
    ]
    
    for i, statement in enumerate(test_statements, 1):
        print(f"\n{'='*20} Test {i} {'='*20}")
        print(f"Statement: {statement}")
        
        try:
            result = verify_fact(statement)
            
            print(f"Overall Verdict: {result.get('overall_verdict', 'Unknown')}")
            print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"Number of Claims: {len(result.get('claims', []))}")
            
            for j, claim_result in enumerate(result.get('claim_results', []), 1):
                print(f"\n  Claim {j}: {claim_result['claim']}")
                print(f"  Verdict: {claim_result['verdict']}")
                print(f"  Reasoning: {claim_result['reasoning'][:100]}...")
                print(f"  Evidence Count: {len(claim_result.get('evidence', []))}")
        
        except Exception as e:
            print(f"❌ Error: {e}")


def setup_environment():
    """Setup the environment for testing."""
    print("Setting up environment...")
    
    # Check if fact base exists
    from config import FACT_BASE_PATH
    if not FACT_BASE_PATH.exists():
        print(f"❌ Fact base not found at {FACT_BASE_PATH}")
        print("Please ensure facts.csv exists in the data directory.")
        return False
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OpenAI API key not found in environment variables.")
        print("Set OPENAI_API_KEY environment variable for LLM evaluation to work.")
    
    print("✅ Environment setup complete!")
    return True


def main():
    """Main testing function."""
    print("🔍 FACT VERIFICATION SYSTEM - TESTING")
    print("=" * 60)
    
    if not setup_environment():
        return
    
    # Test individual components
    test_claim_extraction()
    test_embedding_system()
    
    # Test full pipeline
    test_full_pipeline()
    
    print("\n" + "=" * 60)
    print("✅ TESTING COMPLETE")
    print("=" * 60)
    
    print("\nTo run the Streamlit UI:")
    print("streamlit run app.py")
    
    print("\nTo use the pipeline programmatically:")
    print("from pipeline import verify_fact")
    print('result = verify_fact("Your statement here")')


if __name__ == "__main__":
    main()