"""
Main entry point for the fact verification system.
Provides a simple CLI interface for quick testing.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline import verify_fact
from utils import get_logger

logger = get_logger(__name__)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Fact Verification System using RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "The WHO declared COVID-19 a pandemic in March 2020."
  python main.py --json "Apple announced the iPhone 15 in September 2023."
  
For the web interface, run:
  streamlit run app.py
        """
    )
    
    parser.add_argument(
        "statement",
        help="Statement to verify"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    print("🔍 Fact Verification System")
    print("=" * 50)
    
    if not args.json:
        print(f"Statement: {args.statement}")
        print("-" * 50)
    
    try:
        # Verify the statement
        result = verify_fact(args.statement)
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Format output for readability
            print(f"Overall Verdict: {result.get('overall_verdict', 'Unknown')}")
            print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"Claims Found: {len(result.get('claims', []))}")
            
            if args.verbose and result.get('claim_results'):
                print("\nDetailed Results:")
                for i, claim_result in enumerate(result['claim_results'], 1):
                    print(f"\n{i}. Claim: {claim_result['claim']}")
                    print(f"   Verdict: {claim_result['verdict']}")
                    print(f"   Confidence: {claim_result.get('confidence_score', 0):.1%}")
                    print(f"   Reasoning: {claim_result['reasoning']}")
                    
                    if claim_result.get('evidence'):
                        print(f"   Evidence:")
                        for j, evidence in enumerate(claim_result['evidence'][:3], 1):
                            print(f"     {j}. {evidence}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"CLI error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()