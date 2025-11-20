"""
Streamlit UI for the fact verification system.
"""

import streamlit as st
import json
import time
from typing import Dict, Any, List
import plotly.express as px
import pandas as pd

from pipeline import FactVerificationPipeline, verify_fact
from config import UI_TITLE
from utils import get_logger

# Configure page
st.set_page_config(
    page_title=UI_TITLE,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = get_logger(__name__)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E4057;
        margin-bottom: 2rem;
    }
    .claim-box {
        background-color: #f0f2f6;
        padding: 1rem;
        color: #000000;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .verdict-true {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        color: #000000;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .verdict-false {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        color: #000000;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .verdict-unverifiable {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        color: #000000;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .evidence-box {
        background-color: #e9ecef;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.9rem;
        border-left: 3px solid #007bff;
    }
    .retrieved-fact-card {
        background: #ffffff;
        color: #333333;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 2px solid #e9ecef;
        transition: transform 0.2s ease;
    }
    .retrieved-fact-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        border-color: #667eea;
    }
    .fact-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
        font-weight: bold;
        color: #495057;
    }
    .similarity-badge {
        background-color: #f8f9fa;
        color: #495057;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        border: 1px solid #dee2e6;
    }
    .fact-text {
        font-size: 0.95rem;
        line-height: 1.5;
        color: #333333;
    }
    .details-section {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .metric-container {
        text-align: center;
        padding: 0.8rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin: 0.3rem;
    }
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border: none;
        margin: 1.5rem 0;
        border-radius: 1px;
    }
    /* Ensure expander title text (claim text) is black */
    .streamlit-expanderHeader {
        color: #000000 !important;
    }
    div[data-testid="stExpander"] > div > div > p {
        color: #000000 !important;
    }
    /* Override any Streamlit default text colors */
    .stExpander > div > div > div > p {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load the fact verification pipeline (cached)."""
    try:
        with st.spinner("Loading fact verification pipeline..."):
            pipeline = FactVerificationPipeline()
            return pipeline, None
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        return None, str(e)


def render_verdict_box(verdict: str, reasoning: str, confidence: float = None):
    """Render a styled verdict box."""
    verdict_lower = verdict.lower()
    
    if "true" in verdict_lower and "false" not in verdict_lower:
        css_class = "verdict-true"
        icon = "✅"
    elif "false" in verdict_lower:
        css_class = "verdict-false"
        icon = "❌"
    else:
        css_class = "verdict-unverifiable"
        icon = "❓"
    
    confidence_text = f" (Confidence: {confidence:.1%})" if confidence else ""
    
    st.markdown(f"""
    <div class="{css_class}">
        <h4>{icon} {verdict}{confidence_text}</h4>
        <p>{reasoning}</p>
    </div>
    """, unsafe_allow_html=True)


def render_evidence_section(evidence: List[str]):
    """Render evidence section with enhanced styling."""
    if evidence:
        st.markdown("""
        <h3 style="color: #495057; margin-bottom: 1.5rem;">📚 Supporting Evidence</h3>
        """, unsafe_allow_html=True)
        
        for i, ev in enumerate(evidence, 1):
            st.markdown(f"""
            <div class="evidence-box">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="background-color: #007bff; color: white; padding: 0.2rem 0.6rem; border-radius: 50%; font-size: 0.8rem; font-weight: bold; margin-right: 0.8rem;">{i}</span>
                    <strong style="color: #495057;">Evidence Source</strong>
                </div>
                <p style="margin: 0; line-height: 1.4;">{ev}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; text-align: center;">
            <p style="margin: 0; color: #856404;">⚠️ No evidence found for this claim in our knowledge base.</p>
        </div>
        """, unsafe_allow_html=True)


def render_claim_result(claim_result: Dict[str, Any], claim_num: int):
    """Render individual claim result."""
    with st.expander(f"📝 Claim {claim_num}: {claim_result['claim'][:100]}...", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Verdict and reasoning
            render_verdict_box(
                claim_result["verdict"],
                claim_result["reasoning"],
                claim_result.get("confidence_score")
            )
        
        with col2:
            # Metadata with enhanced styling
            st.markdown("""
            <div class="details-section">
                <h4 style="color: #495057; margin-bottom: 1rem;">📊 Analysis Details</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Styled metrics
            confidence = claim_result.get('confidence_score', 0)
            confidence_color = "#28a745" if confidence > 0.7 else "#ffc107" if confidence > 0.4 else "#dc3545"
            
            st.markdown(f"""
            <div class="metric-container">
                <h3 style="color: {confidence_color}; margin: 0;">{confidence:.1%}</h3>
                <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">Confidence Level</p>
            </div>
            """, unsafe_allow_html=True)
            
            if "retrieved_facts" in claim_result:
                evidence_count = len(claim_result["retrieved_facts"])
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="color: #007bff; margin: 0;">{evidence_count}</h3>
                    <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">Facts Retrieved</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Evidence section
        render_evidence_section(claim_result.get("evidence", []))
        
        # Section divider
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        
        # Retrieved facts details with enhanced styling
        if "retrieved_facts" in claim_result and claim_result["retrieved_facts"]:
            st.markdown("""
            <h3 style="color: #495057; margin-bottom: 1.5rem;">🔍 Retrieved Knowledge Base Facts</h3>
            """, unsafe_allow_html=True)
            
            for i, rf in enumerate(claim_result["retrieved_facts"], 1):
                similarity = rf.get('similarity_score', 0)
                similarity_percent = f"{similarity:.1%}"
                
                # Color-code similarity score
                if similarity > 0.8:
                    similarity_color = "#28a745"  # Green for high similarity
                elif similarity > 0.6:
                    similarity_color = "#ffc107"  # Yellow for medium similarity
                else:
                    similarity_color = "#dc3545"  # Red for low similarity
                
                fact_text = rf.get("text", "No text available")
                
                st.markdown(f"""
                <div class="retrieved-fact-card">
                    <div class="fact-header">
                        <span>📄 Fact {i}</span>
                        <span class="similarity-badge" style="background-color: {similarity_color}; color: white;">
                            {similarity_percent} Match
                        </span>
                    </div>
                    <div class="fact-text">
                        {fact_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)


def render_results_summary(results: Dict[str, Any]):
    """Render summary of all results."""
    st.subheader("📈 Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Claims", len(results.get("claims", [])))
    
    with col2:
        st.metric("Processing Time", f"{results.get('processing_time', 0):.2f}s")
    
    with col3:
        st.metric("Overall Verdict", results.get("overall_verdict", "Unknown"))
    
    with col4:
        if results.get("claim_results"):
            avg_confidence = sum(
                cr.get("confidence_score", 0) 
                for cr in results["claim_results"]
            ) / len(results["claim_results"])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Verdict distribution chart
    if results.get("claim_results"):
        verdict_counts = {}
        for cr in results["claim_results"]:
            verdict = cr["verdict"]
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        if verdict_counts:
            fig = px.pie(
                values=list(verdict_counts.values()),
                names=list(verdict_counts.keys()),
                title="Verdict Distribution",
                color_discrete_map={
                    "True": "#28a745",
                    "False": "#dc3545",
                    "Unverifiable": "#ffc107"
                }
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    # Header
    st.markdown(f'<h1 class="main-header">🔍 {UI_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown("Enter a statement or news claim to verify its accuracy using our RAG-based fact-checking system.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Pipeline options
        st.subheader("Pipeline Settings")
        top_k = st.slider("Number of facts to retrieve", 1, 10, 5)
        use_spacy = st.checkbox("Use spaCy for claim extraction", True)
        use_transformers = st.checkbox("Use transformers for claim extraction", False)
        
        # About section
        st.header("ℹ️ About")
        st.markdown("""
        This fact verification system uses:
        - **Claim Extraction**: Identifies verifiable statements
        - **Evidence Retrieval**: Finds relevant facts from knowledge base
        - **LLM Evaluation**: Uses GPT models to assess truth
        
        **Verdict Types:**
        - ✅ **True**: Claim is supported by evidence
        - ❌ **False**: Claim contradicts evidence  
        - ❓ **Unverifiable**: Insufficient evidence
        """)
    
    # Load pipeline
    pipeline, error = load_pipeline()
    
    if error:
        st.error(f"Failed to load fact verification pipeline: {error}")
        st.stop()
    
    # Main interface
    st.subheader("📝 Enter Statement to Verify")
    
    # Example statements
    example_statements = [
        "The Indian government has announced free electricity to all farmers starting July 2025.",
        "COVID-19 vaccines contain microchips for tracking people.",
        "The Earth's temperature has increased by 1.1 degrees Celsius since pre-industrial times.",
        "Apple announced the iPhone 15 in September 2023.",
        "The population of Tokyo is over 13 million people."
    ]
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Type your own statement", "Select an example"],
        horizontal=True
    )
    
    if input_method == "Type your own statement":
        user_input = st.text_area(
            "Enter statement:",
            height=100,
            placeholder="Type the statement you want to fact-check here..."
        )
    else:
        user_input = st.selectbox(
            "Select an example statement:",
            example_statements
        )
    
    # Process button
    if st.button("🔍 Verify Facts", type="primary", disabled=not user_input.strip()):
        if not user_input.strip():
            st.warning("Please enter a statement to verify.")
            return
        
        # Show input
        st.markdown("### 📋 Input Statement")
        st.markdown(f'<div class="claim-box">{user_input}</div>', unsafe_allow_html=True)
        
        # Process with progress
        with st.spinner("Processing statement..."):
            try:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Extract claims
                status_text.text("Extracting claims...")
                progress_bar.progress(25)
                
                # Step 2: Retrieve evidence
                status_text.text("Retrieving evidence...")
                progress_bar.progress(50)
                
                # Step 3: Evaluate claims
                status_text.text("Evaluating claims...")
                progress_bar.progress(75)
                
                # Run verification
                results = pipeline.verify_fact(user_input)
                
                progress_bar.progress(100)
                status_text.text("Complete!")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"Error during fact verification: {str(e)}")
                logger.error(f"Verification error: {e}")
                return
        
        # Display results
        if results.get("claim_results"):
            # Summary
            render_results_summary(results)
            
            # Individual claim results
            st.subheader("🔍 Detailed Results")
            for i, claim_result in enumerate(results["claim_results"], 1):
                render_claim_result(claim_result, i)
        
        else:
            if "error" in results:
                st.error(f"Error: {results['error']}")
            else:
                st.warning("No verifiable claims found in the input statement.")
        
        # Show raw results in expander
        with st.expander("🔧 Raw Results (JSON)"):
            st.json(results)
        
        # Feedback section
        st.subheader("💬 Feedback")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👍 Helpful"):
                st.success("Thank you for your feedback!")
        
        with col2:
            if st.button("👎 Not Helpful"):
                st.warning("Thank you for your feedback. We'll work to improve!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "Powered by RAG (Retrieval-Augmented Generation) and GPT Models"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()