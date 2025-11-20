"""
Configuration settings for the fact verification system.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
EMBEDDINGS_DIR = PROJECT_ROOT / os.getenv("EMBEDDINGS_DIR", "embeddings")
LOGS_DIR = PROJECT_ROOT / os.getenv("LOGS_DIR", "logs")

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Model configurations
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # sentence-transformers model
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # OpenAI model

# Vector store settings
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")  # or "chromadb"
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "facts_index.faiss"
FAISS_METADATA_PATH = EMBEDDINGS_DIR / "facts_metadata.json"

# Fact base settings
FACT_BASE_PATH = DATA_DIR / os.getenv("FACT_BASE_FILE", "facts.csv")

# Retrieval settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# LLM settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

# Local LLM settings
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
LOCAL_LLM_CONFIG = os.getenv("LOCAL_LLM_CONFIG", "ollama_llama2")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "fact_verification.log"

# Claim extraction settings
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
MAX_CLAIM_LENGTH = int(os.getenv("MAX_CLAIM_LENGTH", "500"))

# UI settings
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
UI_TITLE = os.getenv("UI_TITLE", "Fact Verification System")

def get_config() -> Dict[str, Any]:
    """Get all configuration settings as a dictionary."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "vector_db_type": VECTOR_DB_TYPE,
        "fact_base_path": str(FACT_BASE_PATH),
        "top_k": DEFAULT_TOP_K,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "log_level": LOG_LEVEL,
        "spacy_model": SPACY_MODEL,
        "openai_api_key_set": bool(OPENAI_API_KEY),
    }


def validate_config() -> Dict[str, str]:
    """
    Validate configuration and return any issues.
    
    Returns:
        Dictionary with validation results
    """
    issues = {}
    
    if not OPENAI_API_KEY:
        issues["openai_api_key"] = "OpenAI API key not set. Set OPENAI_API_KEY environment variable."
    
    if not FACT_BASE_PATH.exists():
        issues["fact_base"] = f"Fact base file not found at {FACT_BASE_PATH}"
    
    # Check if required directories exist
    for dir_name, dir_path in [("data", DATA_DIR), ("embeddings", EMBEDDINGS_DIR), ("logs", LOGS_DIR)]:
        if not dir_path.exists():
            issues[f"{dir_name}_dir"] = f"{dir_name.capitalize()} directory not found at {dir_path}"
    
    return issues