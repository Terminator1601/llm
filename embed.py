"""
Embedding and vector store module for storing and retrieving fact embeddings.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss

from config import (
    EMBEDDING_MODEL, FACT_BASE_PATH, FAISS_INDEX_PATH, 
    FAISS_METADATA_PATH, EMBEDDINGS_DIR
)
from utils import get_logger

logger = get_logger(__name__)


class FactEmbedder:
    """Handle embedding generation for facts using sentence transformers."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the fact embedder.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def embed_single_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            NumPy array representing the embedding
        """
        return self.embed_texts([text])[0]


class Fact:
    """Represent a single fact with its metadata."""
    
    def __init__(self, fact_id: str, fact_text: str, metadata: Optional[Dict] = None):
        """
        Initialize a fact.
        
        Args:
            fact_id: Unique identifier for the fact
            fact_text: The actual fact text
            metadata: Additional metadata (source, date, etc.)
        """
        self.id = fact_id
        self.text = fact_text
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fact to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        """Create fact from dictionary representation."""
        return cls(
            fact_id=data["id"],
            fact_text=data["text"],
            metadata=data.get("metadata", {})
        )


class VectorStore:
    """FAISS-based vector store for storing and retrieving fact embeddings."""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of the embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.facts = []
        self.embedder = FactEmbedder()
        
        # Initialize FAISS index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index for similarity search."""
        # Use L2 distance for similarity search
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        logger.info(f"Initialized FAISS index with dimension {self.embedding_dim}")
    
    def add_facts(self, facts: List[Fact]):
        """
        Add facts to the vector store.
        
        Args:
            facts: List of Fact objects to add
        """
        if not facts:
            logger.warning("No facts to add")
            return
        
        logger.info(f"Adding {len(facts)} facts to vector store")
        
        # Extract texts for embedding
        fact_texts = [fact.text for fact in facts]
        
        # Generate embeddings
        embeddings = self.embedder.embed_texts(fact_texts)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store fact metadata
        self.facts.extend(facts)
        
        logger.info(f"Added {len(facts)} facts. Total facts: {len(self.facts)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Fact, float]]:
        """
        Search for similar facts given a query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (Fact, distance) tuples, sorted by similarity
        """
        if not self.facts:
            logger.warning("No facts in vector store")
            return []
        
        logger.info(f"Searching for similar facts to: {query[:100]}...")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_single_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.facts)))
        
        # Prepare results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.facts):  # Valid index
                fact = self.facts[idx]
                results.append((fact, float(distance)))
        
        logger.info(f"Found {len(results)} similar facts")
        return results
    
    def save(self, index_path: Path = FAISS_INDEX_PATH, metadata_path: Path = FAISS_METADATA_PATH):
        """
        Save the vector store to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save fact metadata
        """
        logger.info(f"Saving vector store to {index_path} and {metadata_path}")
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save fact metadata
        facts_data = [fact.to_dict() for fact in self.facts]
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(facts_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Vector store saved successfully")
    
    def load(self, index_path: Path = FAISS_INDEX_PATH, metadata_path: Path = FAISS_METADATA_PATH) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to fact metadata file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not index_path.exists() or not metadata_path.exists():
            logger.warning(f"Vector store files not found at {index_path} or {metadata_path}")
            return False
        
        try:
            logger.info(f"Loading vector store from {index_path} and {metadata_path}")
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load fact metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                facts_data = json.load(f)
            
            self.facts = [Fact.from_dict(data) for data in facts_data]
            
            logger.info(f"Loaded {len(self.facts)} facts from vector store")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False


class FactBase:
    """Manage the fact base and vector store operations."""
    
    def __init__(self, fact_base_path: Path = FACT_BASE_PATH):
        """
        Initialize the fact base.
        
        Args:
            fact_base_path: Path to the CSV file containing facts
        """
        self.fact_base_path = fact_base_path
        self.vector_store = VectorStore()
    
    def load_fact_base(self) -> List[Fact]:
        """
        Load facts from CSV file.
        
        Returns:
            List of Fact objects
        """
        if not self.fact_base_path.exists():
            logger.warning(f"Fact base not found at {self.fact_base_path}")
            return []
        
        try:
            logger.info(f"Loading fact base from {self.fact_base_path}")
            df = pd.read_csv(self.fact_base_path)
            
            required_columns = ['id', 'fact_text']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            facts = []
            for _, row in df.iterrows():
                fact = Fact(
                    fact_id=str(row['id']),
                    fact_text=str(row['fact_text']),
                    metadata={col: row[col] for col in df.columns if col not in required_columns}
                )
                facts.append(fact)
            
            logger.info(f"Loaded {len(facts)} facts from CSV")
            return facts
        
        except Exception as e:
            logger.error(f"Failed to load fact base: {e}")
            return []
    
    def embed_facts(self) -> bool:
        """
        Load facts and create embeddings in vector store.
        
        Returns:
            True if successful, False otherwise
        """
        # Try to load existing vector store first
        if self.vector_store.load():
            logger.info("Loaded existing vector store")
            return True
        
        # Load facts from CSV and create new vector store
        facts = self.load_fact_base()
        if not facts:
            logger.error("No facts to embed")
            return False
        
        # Add facts to vector store
        self.vector_store.add_facts(facts)
        
        # Save vector store
        self.vector_store.save()
        
        return True
    
    def search_similar_facts(self, query: str, top_k: int = 5) -> List[Tuple[Fact, float]]:
        """
        Search for facts similar to the query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of (Fact, distance) tuples
        """
        return self.vector_store.search(query, top_k)


def load_fact_base() -> FactBase:
    """
    Convenience function to load and initialize the fact base.
    
    Returns:
        Initialized FactBase instance
    """
    fact_base = FactBase()
    fact_base.embed_facts()
    return fact_base


def embed_facts() -> bool:
    """
    Convenience function to embed facts into vector store.
    
    Returns:
        True if successful, False otherwise
    """
    fact_base = FactBase()
    return fact_base.embed_facts()


def store_embeddings() -> bool:
    """
    Convenience function to store embeddings.
    This is included in embed_facts() automatically.
    
    Returns:
        True if successful, False otherwise
    """
    return embed_facts()