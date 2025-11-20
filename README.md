# RAG Fact Verification System

A comprehensive, production-quality fact-verification system using Retrieval-Augmented Generation (RAG) with advanced enhancements and dual LLM support.

## 🚀 Features

### Core Capabilities
- **Advanced Claim Extraction**: Multi-strategy extraction using spaCy NLP and transformers
- **Semantic Retrieval**: FAISS vector database with sentence-transformers embeddings
- **Dual LLM Support**: OpenAI API + Local LLM (Ollama/llama.cpp) with automatic fallback
- **Enhanced Data**: 40 verified facts including government PIB data
- **Smart Filtering**: Vague claim detection and text chunking for optimal processing

### User Interfaces
- **Streamlit UI**: Primary web interface with enhanced styling
- **Gradio UI**: Alternative interface with charts and visualization
- **CLI Interface**: Command-line tool for batch processing

### Enhancements
- **PIB Data Integration**: Government Press Information Bureau fact collection
- **Intelligent Chunking**: NLTK-based text segmentation (150-250 chars)
- **Vague Detection**: ML-based scoring to filter unclear claims
- **Local LLM Support**: Offline processing with Ollama and llama.cpp
- **Production Quality**: Comprehensive error handling and logging

## 🏗️ System Architecture

```
Input Text → Vague Detection → Claim Extraction → Text Chunking → Evidence Retrieval → LLM Evaluation → Verdict
              ↓                    ↓                 ↓                ↓                      ↓
        [ML Scoring]         [spaCy/Transformers] [NLTK Chunker]  [FAISS + Embeddings]  [OpenAI/Local LLM]
                                                                        ↓
                                                               [40 Facts + PIB Data]
```

### Components
- **Vague Detection**: ML scoring (0.0-1.0) to filter unclear claims
- **Claim Extraction**: Multi-strategy extraction (spaCy + transformers)
- **Text Chunking**: Intelligent segmentation for large documents
- **Vector Database**: FAISS with 384-dim sentence-transformers embeddings
- **Fact Base**: 40 verified facts (30 original + 10 PIB government data)
- **LLM Evaluation**: OpenAI GPT-3.5-turbo or local models (Ollama/llama.cpp)
- **Dual UI**: Streamlit (primary) and Gradio (alternative) interfaces

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Set up LLM access** (choose one):
   
   **Option A: OpenAI API**
   ```bash
   # Windows
   set OPENAI_API_KEY=your_openai_api_key_here
   
   # Linux/Mac
   export OPENAI_API_KEY=your_openai_api_key_here
   ```
   
   **Option B: Local LLM (Ollama)**
   ```bash
   # Install Ollama from https://ollama.ai/
   ollama pull llama2  # or llama3, mistral
   
   # Configure environment
   set USE_LOCAL_LLM=true
   set LOCAL_LLM_CONFIG=ollama_llama2
   ```

## Quick Start

### 1. Test the System
```bash
# Test core functionality
python test_system.py

# Test local LLM connections
python test_local_llm_simple.py
```

### 2. Run the Web Interfaces
```bash
# Primary Streamlit UI
streamlit run app.py

# Alternative Gradio UI
python gradio_app.py
```

### 3. Use Programmatically

**Basic Usage (OpenAI)**
```python
from pipeline import FactVerificationPipeline

# Initialize with OpenAI (default)
pipeline = FactVerificationPipeline()
result = pipeline.verify_fact("The WHO declared COVID-19 a pandemic in March 2020.")
print(result)
```

**Local LLM Usage**
```python
# Use local Ollama LLM
pipeline = FactVerificationPipeline(
    use_local_llm=True,
    local_llm_config="ollama_llama3"
)
result = pipeline.verify_fact("Paris is the capital of France.")
```

**Enhanced Features**
```python
# With vague detection and chunking
from vague_detection import detect_vague_claims
from chunking import chunk_text

# Filter vague claims
text = "Some politicians sometimes make statements."
vague_score = detect_vague_claims([text])[0][1]
if vague_score < 0.5:  # Not too vague
    result = pipeline.verify_fact(text)

# Process large documents
large_text = "Very long document with multiple claims..."
chunks = chunk_text(large_text, method="sentence", max_length=200)
results = [pipeline.verify_fact(chunk) for chunk in chunks]
```

## API Response Format

```json
{
  "input_text": "Your input statement",
  "claims": ["Extracted claim 1", "Extracted claim 2"],
  "claim_results": [
    {
      "claim": "Specific claim text",
      "verdict": "True|False|Unverifiable",
      "reasoning": "Detailed explanation of the verdict",
      "evidence": ["Supporting evidence 1", "Supporting evidence 2"],
      "confidence_score": 0.85
    }
  ],
  "overall_verdict": "Summary of all claims",
  "processing_time": 2.34
}
```

## 📁 Project Structure

```
llm/
├── Core Modules (8 files)
│   ├── config.py              # Configuration settings
│   ├── utils.py               # Logging utilities
│   ├── extract.py             # Claim extraction (spaCy/transformers)
│   ├── embed.py               # Embeddings & FAISS vector store
│   ├── retrieval.py           # Semantic search & fact retrieval
│   ├── evaluate.py            # LLM evaluation (OpenAI/Local)
│   ├── pipeline.py            # RAG pipeline orchestrator
│   └── main.py                # CLI interface
├── UI Modules (2 files)
│   ├── app.py                 # Streamlit web interface
│   └── gradio_app.py          # Gradio alternative UI
├── Enhancement Modules (5 files)
│   ├── pib_scraper.py         # Government data collection
│   ├── chunking.py            # Intelligent text segmentation
│   ├── vague_detection.py     # ML-based claim filtering
│   ├── local_llm.py           # Local LLM support (Ollama/llama.cpp)
│   └── test_local_llm_simple.py # LLM connection testing
├── Data & Storage
│   ├── data/
│   │   ├── verified_facts.csv # Original 30 verified facts
│   │   └── pib_facts.csv      # 10 government PIB facts
│   ├── embeddings/            # FAISS vector store (auto-generated)
│   └── logs/                  # System logs (auto-generated)
├── Documentation
│   ├── EVALUATION_REPORT.json # Comprehensive system evaluation
│   ├── EVALUATION_REPORT.md   # Human-readable evaluation
│   └── IMPLEMENTATION_SUMMARY.md # Complete feature summary
└── requirements.txt           # Python dependencies
```

## 🎯 System Performance

### Evaluation Results
- **Feature Completeness**: 35/36 items (97.2%)
- **Core Functionality**: All RAG components operational
- **Enhanced Features**: 5/5 enhancements implemented
- **Fact Base**: 40 verified facts (30 original + 10 PIB government data)
- **UI Options**: Streamlit + Gradio interfaces
- **LLM Support**: OpenAI API + Local LLMs (Ollama/llama.cpp)

### System Capabilities
- **Claim Processing**: Multi-strategy extraction with vague filtering
- **Semantic Search**: FAISS vector database with 384-dim embeddings
- **Evidence Evaluation**: Structured LLM assessment with confidence scoring
- **Batch Processing**: Handle multiple claims and large documents
- **Offline Support**: Local LLM operation without internet dependency

## 🛠️ Advanced Configuration

### Local LLM Setup
```bash
# Install Ollama
# Windows: Download from https://ollama.ai/download/windows
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama2    # ~3.8GB, good balance of speed/quality
ollama pull llama3    # ~4.7GB, higher quality responses  
ollama pull mistral   # ~4.1GB, fast and efficient

# Configure environment
export USE_LOCAL_LLM=true
export LOCAL_LLM_CONFIG=ollama_llama3
```

### Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=1000
OPENAI_TEMPERATURE=0.1

# Local LLM Configuration  
USE_LOCAL_LLM=true
LOCAL_LLM_CONFIG=ollama_llama3

# System Configuration
LOG_LEVEL=INFO
DEFAULT_TOP_K=5
SIMILARITY_THRESHOLD=0.7
```

### Custom Configuration
```python
# config.py customization
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # or "all-mpnet-base-v2"
VECTOR_DB_TYPE = "faiss"  # or "chromadb" 
LLM_MODEL = "gpt-3.5-turbo"  # or "gpt-4"
```

## Extending the Fact Base

Add more facts to `data/facts.csv`:

```csv
id,fact_text
31,"Your new fact here"
32,"Another verified fact"
```

The system will automatically re-index the facts when you run it next.

## Advanced Usage

### Custom Pipeline Configuration
```python
from pipeline import FactVerificationPipeline

# Create custom pipeline
pipeline = FactVerificationPipeline(
    use_spacy=True,
    use_transformers=False,
    top_k_retrieval=10
)

# Get pipeline information
info = pipeline.get_pipeline_info()
print(info)
```

### Batch Processing
```python
statements = [
    "Statement 1 to verify",
    "Statement 2 to verify",
    "Statement 3 to verify"
]

results = pipeline.verify_multiple_texts(statements)
```

### Using Individual Components
```python
# Extract claims only
from extract import extract_claims
claims = extract_claims("Your text here")

# Retrieve similar facts only
from retrieval import retrieve_similar_facts
facts = retrieve_similar_facts("Your claim here", top_k=5)

# Evaluate claim against evidence only
from evaluate import evaluate_claim_against_evidence
result = evaluate_claim_against_evidence("Claim", ["Evidence 1", "Evidence 2"])
```

## Troubleshooting

### Common Issues

1. **OpenAI API errors**: Ensure API key is set and has sufficient credits
2. **spaCy model not found**: Run `python -m spacy download en_core_web_sm`
3. **FAISS installation issues**: Use `pip install faiss-cpu` for CPU-only version
4. **No facts found**: Check that `data/facts.csv` exists and has content

### Performance Tips

- For faster inference, reduce `top_k_retrieval` parameter
- Use CPU-only models if GPU is not available
- Cache the pipeline object for multiple requests

## 📚 Documentation Files

- **`IMPLEMENTATION_SUMMARY.md`** - Complete project overview and achievements
- **`EVALUATION_REPORT.json`** - Detailed 36-item evaluation results
- **`EVALUATION_REPORT.md`** - Human-readable evaluation summary
- **`test_local_llm_simple.py`** - Local LLM connection testing
- **`README.md`** - This comprehensive guide

## 🚀 Future Enhancements

The system is production-ready with 97.2% feature completeness. Potential future improvements:

1. **Multi-language Support** - Fact verification in multiple languages
2. **Real-time Data Sources** - Live news API integration
3. **Advanced Analytics** - Detailed performance dashboards
4. **Enterprise API** - REST API for external system integration
5. **Distributed Processing** - Scale to handle enterprise workloads

## 🤝 Contributing

To extend the system:

1. **Add new extractors**: Extend `ClaimExtractor` class in `extract.py`
2. **Add new evaluators**: Implement evaluation strategies in `evaluate.py`  
3. **Add new vector stores**: Extend storage backends in `embed.py`
4. **Add new UI features**: Enhance Streamlit/Gradio interfaces
5. **Add new data sources**: Implement scrapers like `pib_scraper.py`

## 📝 License

Educational and research purposes. Comply with OpenAI usage policies when using their API.

## 🙏 Credits & Technologies

Built with modern Python ML/NLP stack:
- **[Sentence Transformers](https://www.sbert.net/)** - Semantic embeddings
- **[FAISS](https://faiss.ai/)** - Efficient vector similarity search  
- **[spaCy](https://spacy.io/)** - Industrial-strength NLP
- **[OpenAI API](https://openai.com/)** - GPT language models
- **[Streamlit](https://streamlit.io/)** - Interactive web applications
- **[Gradio](https://gradio.app/)** - Machine learning interfaces
- **[Ollama](https://ollama.ai/)** - Local LLM deployment
- **[NLTK](https://nltk.org/)** - Natural language toolkit
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning utilities