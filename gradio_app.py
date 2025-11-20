"""
Gradio UI interface for the fact verification system.
"""

import gradio as gr
import json
import pandas as pd
from typing import Dict, Any, List, Tuple
import plotly.express as px
import plotly.graph_objects as go

from pipeline import FactVerificationPipeline
from vague_detection import VagueClaimDetector
from config import UI_TITLE
from utils import get_logger

logger = get_logger(__name__)

# Global variables for caching
pipeline = None
vague_detector = None

def load_components():
    """Load pipeline and vague detector (cached)."""
    global pipeline, vague_detector
    
    if pipeline is None:
        try:
            pipeline = FactVerificationPipeline()
            logger.info("Loaded fact verification pipeline for Gradio")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            return False, str(e)
    
    if vague_detector is None:
        try:
            vague_detector = VagueClaimDetector()
            logger.info("Loaded vague claim detector for Gradio")
        except Exception as e:
            logger.error(f"Failed to load vague detector: {e}")
            vague_detector = None
    
    return True, "Components loaded successfully"

def verify_fact_gradio(text: str, check_vagueness: bool = True) -> Tuple[str, str, str, str]:
    """
    Verify facts using Gradio interface.
    
    Args:
        text: Input text to verify
        check_vagueness: Whether to check for vague claims
        
    Returns:
        Tuple of (verdict_html, details_json, chart_html, log_text)
    """
    if not text or not text.strip():
        return "❌ Please enter a statement to verify.", "{}", "", "Error: Empty input"
    
    try:
        # Load components if not already loaded
        success, message = load_components()
        if not success:
            return f"❌ Error: {message}", "{}", "", f"Failed to load: {message}"
        
        log_messages = [f"Processing: {text[:100]}..."]
        
        # Check vagueness if enabled
        if check_vagueness and vague_detector:
            # Extract potential claims first
            potential_claims = text.split('.') if '.' in text else [text]
            potential_claims = [c.strip() for c in potential_claims if c.strip()]
            
            vague_scores = []
            for claim in potential_claims:
                score = vague_detector.score_vagueness(claim)
                vague_scores.append(score)
                log_messages.append(f"Vagueness check: '{claim[:50]}...' -> {score.vagueness_score:.3f} ({score.vagueness_category})")
                
                if score.vagueness_score > 0.7:
                    return (
                        f"⚠️ **Claim too vague to verify**\n\n"
                        f"**Vagueness Score:** {score.vagueness_score:.3f}/1.0\n"
                        f"**Category:** {score.vagueness_category}\n"
                        f"**Recommendation:** {score.recommendation}\n\n"
                        f"**Issues found:** {', '.join(score.indicators[:5])}\n\n"
                        f"Please provide a more specific claim with concrete details.",
                        json.dumps(score.to_dict(), indent=2),
                        "",
                        "\n".join(log_messages)
                    )
        
        # Run fact verification
        log_messages.append("Running fact verification pipeline...")
        results = pipeline.verify_fact(text)
        
        # Format results
        verdict_html = format_verdict_html(results)
        details_json = json.dumps(results.to_dict() if hasattr(results, 'to_dict') else results, indent=2)
        chart_html = create_results_chart(results)
        
        log_messages.append(f"Verification complete in {results.get('processing_time', 0):.2f}s")
        log_text = "\n".join(log_messages)
        
        return verdict_html, details_json, chart_html, log_text
        
    except Exception as e:
        logger.error(f"Gradio verification error: {e}")
        return f"❌ Error during verification: {str(e)}", "{}", "", f"Error: {str(e)}"

def format_verdict_html(results: Dict[str, Any]) -> str:
    """Format verification results as HTML."""
    if not results or 'claim_results' not in results:
        return "❌ No verifiable claims found in the input text."
    
    html_parts = []
    
    # Overall summary
    overall_verdict = results.get('overall_verdict', 'Unknown')
    processing_time = results.get('processing_time', 0)
    claim_count = len(results.get('claims', []))
    
    if overall_verdict == "All claims are true":
        verdict_color = "#28a745"
        verdict_icon = "✅"
    elif overall_verdict == "All claims are false":
        verdict_color = "#dc3545"
        verdict_icon = "❌"
    else:
        verdict_color = "#ffc107"
        verdict_icon = "⚠️"
    
    html_parts.append(f"""
    <div style="border: 2px solid {verdict_color}; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: {verdict_color}15;">
        <h2 style="color: {verdict_color}; margin-top: 0;">{verdict_icon} {overall_verdict}</h2>
        <p><strong>Claims analyzed:</strong> {claim_count} | <strong>Processing time:</strong> {processing_time:.2f}s</p>
    </div>
    """)
    
    # Individual claim results
    for i, claim_result in enumerate(results.get('claim_results', []), 1):
        verdict = claim_result.get('verdict', 'Unknown')
        confidence = claim_result.get('confidence_score', 0)
        reasoning = claim_result.get('reasoning', 'No reasoning provided')
        claim_text = claim_result.get('claim', 'Unknown claim')
        evidence = claim_result.get('evidence', [])
        
        # Verdict styling
        if verdict.lower() == 'true':
            verdict_color = "#28a745"
            verdict_icon = "✅"
        elif verdict.lower() == 'false':
            verdict_color = "#dc3545"
            verdict_icon = "❌"
        else:
            verdict_color = "#ffc107"
            verdict_icon = "❓"
        
        # Confidence bar
        confidence_percent = confidence * 100
        confidence_color = "#28a745" if confidence > 0.7 else "#ffc107" if confidence > 0.4 else "#dc3545"
        
        html_parts.append(f"""
        <div style="border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; margin: 15px 0; background-color: #f8f9fa;">
            <h3 style="color: #495057;">📝 Claim {i}</h3>
            <div style="background-color: #ffffff; padding: 10px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #007bff;">
                <em>"{claim_text}"</em>
            </div>
            
            <div style="background-color: {verdict_color}15; border: 1px solid {verdict_color}; border-radius: 5px; padding: 15px; margin: 10px 0;">
                <h4 style="color: {verdict_color}; margin-top: 0;">{verdict_icon} {verdict}</h4>
                <p><strong>Confidence:</strong> 
                    <span style="background-color: {confidence_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.9em;">
                        {confidence_percent:.1f}%
                    </span>
                </p>
                <p><strong>Reasoning:</strong> {reasoning}</p>
            </div>
            
            <div style="margin: 10px 0;">
                <strong>📚 Supporting Evidence ({len(evidence)} sources):</strong>
                <ul>
        """)
        
        for j, ev in enumerate(evidence[:3], 1):  # Show first 3 evidence items
            html_parts.append(f"<li>{ev}</li>")
        
        if len(evidence) > 3:
            html_parts.append(f"<li><em>... and {len(evidence) - 3} more evidence sources</em></li>")
        
        html_parts.append("</ul></div></div>")
    
    return "".join(html_parts)

def create_results_chart(results: Dict[str, Any]) -> str:
    """Create a chart visualization of results."""
    if not results or 'claim_results' not in results:
        return "<p>No data to visualize</p>"
    
    try:
        # Count verdicts
        verdicts = [cr.get('verdict', 'Unknown') for cr in results.get('claim_results', [])]
        verdict_counts = {}
        for verdict in verdicts:
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        if not verdict_counts:
            return "<p>No verdict data available</p>"
        
        # Create pie chart
        colors = {
            'True': '#28a745',
            'False': '#dc3545', 
            'Unverifiable': '#ffc107'
        }
        
        fig = px.pie(
            values=list(verdict_counts.values()),
            names=list(verdict_counts.keys()),
            title="Fact Verification Results",
            color_discrete_map=colors
        )
        
        fig.update_layout(
            font=dict(size=14),
            showlegend=True,
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="verdict-chart")
        
    except Exception as e:
        logger.error(f"Chart creation error: {e}")
        return f"<p>Error creating chart: {str(e)}</p>"

def create_gradio_interface():
    """Create and configure Gradio interface."""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        max-width: 1200px;
        margin: auto;
    }
    .output-html {
        max-height: 600px;
        overflow-y: auto;
    }
    """
    
    with gr.Blocks(css=custom_css, title=f"🔍 {UI_TITLE}") as interface:
        gr.Markdown(f"""
        # 🔍 {UI_TITLE}
        
        Enter a statement or news claim to verify its accuracy using our RAG-based fact-checking system.
        
        **Features:**
        - 🧠 AI-powered claim extraction
        - 🔍 Semantic similarity search 
        - 🤖 GPT-based fact verification
        - ⚠️ Vague claim detection
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                with gr.Group():
                    gr.Markdown("## 📝 Input Statement")
                    
                    text_input = gr.Textbox(
                        label="Statement to verify",
                        placeholder="Enter the statement you want to fact-check here...",
                        lines=4
                    )
                    
                    with gr.Row():
                        check_vague = gr.Checkbox(
                            label="Check for vague claims",
                            value=True,
                            info="Filter out vague or non-verifiable statements"
                        )
                    
                    verify_btn = gr.Button(
                        "🔍 Verify Facts", 
                        variant="primary",
                        size="lg"
                    )
                
                # Example statements
                with gr.Group():
                    gr.Markdown("## 📋 Example Statements")
                    
                    examples = [
                        "The WHO declared COVID-19 a pandemic on March 11, 2020.",
                        "Apple announced the iPhone 15 in September 2023.",
                        "COVID-19 vaccines contain microchips for tracking people.",
                        "The Earth is flat and NASA is hiding the truth.",
                        "India achieved 100% village electrification by April 2018."
                    ]
                    
                    for example in examples:
                        gr.Button(
                            f"📄 {example[:60]}{'...' if len(example) > 60 else ''}",
                            size="sm"
                        ).click(
                            fn=lambda x=example: x,
                            outputs=text_input
                        )
            
            with gr.Column(scale=3):
                # Results section
                gr.Markdown("## 📊 Verification Results")
                
                # Main results display
                verdict_display = gr.HTML(
                    label="Verification Results",
                    value="<p>👆 Enter a statement above and click 'Verify Facts' to see results here.</p>"
                )
                
                # Tabs for different views
                with gr.Tabs():
                    with gr.Tab("📈 Chart"):
                        chart_display = gr.HTML(
                            label="Results Visualization"
                        )
                    
                    with gr.Tab("🔧 Raw Data"):
                        json_display = gr.Code(
                            label="Raw JSON Results",
                            language="json"
                        )
                    
                    with gr.Tab("📝 Processing Log"):
                        log_display = gr.Code(
                            label="Processing Log",
                            language="text"
                        )
        
        # Footer
        gr.Markdown("""
        ---
        
        **About this system:**
        - 🤖 Powered by RAG (Retrieval-Augmented Generation)
        - 📚 Knowledge base with verified facts and PIB releases  
        - 🎯 Uses GPT models for intelligent fact verification
        - ⚡ Fast FAISS vector search for evidence retrieval
        
        **Disclaimer:** This tool provides AI-assisted fact verification. Always cross-reference important claims with authoritative sources.
        """)
        
        # Event handlers
        verify_btn.click(
            fn=verify_fact_gradio,
            inputs=[text_input, check_vague],
            outputs=[verdict_display, json_display, chart_display, log_display]
        )
        
        # Quick verify on Enter
        text_input.submit(
            fn=verify_fact_gradio,
            inputs=[text_input, check_vague], 
            outputs=[verdict_display, json_display, chart_display, log_display]
        )
    
    return interface

def launch_gradio(share: bool = False, port: int = 7860):
    """Launch Gradio interface."""
    logger.info("Starting Gradio interface...")
    
    try:
        interface = create_gradio_interface()
        
        # Pre-load components
        success, message = load_components()
        if not success:
            logger.warning(f"Failed to pre-load components: {message}")
        
        # Launch interface
        interface.launch(
            share=share,
            server_port=port,
            show_error=True,
            inbrowser=True
        )
        
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {e}")
        raise

if __name__ == "__main__":
    launch_gradio(share=False, port=7860)