"""
Text chunking utilities for breaking long facts into optimal retrieval chunks.
"""

import re
import nltk
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from utils import get_logger

logger = get_logger(__name__)

# Download required NLTK data
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    chunk_id: str
    text: str
    original_id: str
    chunk_index: int
    total_chunks: int
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "original_id": self.original_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "metadata": self.metadata
        }


class TextChunker:
    """Intelligent text chunking for optimal retrieval performance."""
    
    def __init__(self, 
                 min_chunk_size: int = 150,
                 max_chunk_size: int = 250,
                 overlap_size: int = 20):
        """
        Initialize the text chunker.
        
        Args:
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            overlap_size: Overlap between consecutive chunks
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        
        logger.info(f"TextChunker initialized: {min_chunk_size}-{max_chunk_size} chars, {overlap_size} overlap")
    
    def chunk_text(self, text: str, fact_id: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """
        Chunk text into optimal sizes for retrieval.
        
        Args:
            text: Input text to chunk
            fact_id: Original fact ID
            metadata: Additional metadata to preserve
            
        Returns:
            List of TextChunk objects
        """
        if not text or len(text.strip()) == 0:
            return []
        
        text = text.strip()
        metadata = metadata or {}
        
        # If text is already within optimal size, return as single chunk
        if len(text) <= self.max_chunk_size:
            return [TextChunk(
                chunk_id=f"{fact_id}_chunk_0",
                text=text,
                original_id=fact_id,
                chunk_index=0,
                total_chunks=1,
                start_pos=0,
                end_pos=len(text),
                metadata=metadata
            )]
        
        # Use different chunking strategies based on text structure
        chunks = []
        
        # Strategy 1: Sentence-based chunking (preferred)
        sentence_chunks = self._chunk_by_sentences(text, fact_id, metadata)
        if sentence_chunks:
            chunks = sentence_chunks
        else:
            # Strategy 2: Phrase-based chunking (fallback)
            chunks = self._chunk_by_phrases(text, fact_id, metadata)
        
        # Post-process chunks to ensure quality
        chunks = self._post_process_chunks(chunks)
        
        logger.info(f"Chunked text '{text[:50]}...' into {len(chunks)} chunks")
        return chunks
    
    def _chunk_by_sentences(self, text: str, fact_id: str, metadata: Dict) -> List[TextChunk]:
        """Chunk text by sentences, respecting size constraints."""
        try:
            # Split into sentences
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed max size
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(potential_chunk) <= self.max_chunk_size:
                # Add sentence to current chunk
                current_chunk = potential_chunk
            else:
                # Finalize current chunk if it meets minimum size
                if len(current_chunk) >= self.min_chunk_size:
                    chunk = TextChunk(
                        chunk_id=f"{fact_id}_chunk_{len(chunks)}",
                        text=current_chunk,
                        original_id=fact_id,
                        chunk_index=len(chunks),
                        total_chunks=0,  # Will be updated later
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk),
                        metadata=metadata.copy()
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + (" " if overlap_text else "") + sentence
                else:
                    # Current chunk too small, continue building
                    current_chunk = potential_chunk
        
        # Add final chunk
        if len(current_chunk) >= self.min_chunk_size:
            chunk = TextChunk(
                chunk_id=f"{fact_id}_chunk_{len(chunks)}",
                text=current_chunk,
                original_id=fact_id,
                chunk_index=len(chunks),
                total_chunks=0,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                metadata=metadata.copy()
            )
            chunks.append(chunk)
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_phrases(self, text: str, fact_id: str, metadata: Dict) -> List[TextChunk]:
        """Chunk text by phrases as fallback method."""
        # Split by common phrase boundaries
        phrases = re.split(r'[,;:]\s+', text)
        phrases = [p.strip() for p in phrases if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for phrase in phrases:
            potential_chunk = current_chunk + (" " if current_chunk else "") + phrase
            
            if len(potential_chunk) <= self.max_chunk_size:
                current_chunk = potential_chunk
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    chunk = TextChunk(
                        chunk_id=f"{fact_id}_chunk_{len(chunks)}",
                        text=current_chunk,
                        original_id=fact_id,
                        chunk_index=len(chunks),
                        total_chunks=0,
                        start_pos=current_start,
                        end_pos=current_start + len(current_chunk),
                        metadata=metadata.copy()
                    )
                    chunks.append(chunk)
                    
                    current_start = current_start + len(current_chunk)
                    current_chunk = phrase
                else:
                    current_chunk = potential_chunk
        
        # Add final chunk
        if len(current_chunk) >= self.min_chunk_size:
            chunk = TextChunk(
                chunk_id=f"{fact_id}_chunk_{len(chunks)}",
                text=current_chunk,
                original_id=fact_id,
                chunk_index=len(chunks),
                total_chunks=0,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                metadata=metadata.copy()
            )
            chunks.append(chunk)
        
        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk."""
        if len(text) <= self.overlap_size:
            return text
        
        # Try to find sentence boundary for natural overlap
        overlap_text = text[-self.overlap_size:]
        
        # Find last sentence start in overlap
        sentence_start = overlap_text.rfind('. ')
        if sentence_start != -1:
            return overlap_text[sentence_start + 2:]
        
        return overlap_text
    
    def _post_process_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Post-process chunks to ensure quality."""
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        for chunk in chunks:
            # Clean chunk text
            chunk.text = chunk.text.strip()
            
            # Skip very short chunks unless it's the only chunk
            if len(chunk.text) < self.min_chunk_size and len(chunks) > 1:
                # Try to merge with previous chunk
                if processed_chunks:
                    prev_chunk = processed_chunks[-1]
                    merged_text = prev_chunk.text + " " + chunk.text
                    
                    if len(merged_text) <= self.max_chunk_size:
                        # Merge with previous chunk
                        prev_chunk.text = merged_text
                        prev_chunk.end_pos = chunk.end_pos
                        continue
            
            processed_chunks.append(chunk)
        
        # Update indices
        for i, chunk in enumerate(processed_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(processed_chunks)
            # Update chunk ID to reflect new index
            chunk.chunk_id = f"{chunk.original_id}_chunk_{i}"
        
        return processed_chunks
    
    def chunk_facts_dataset(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk an entire facts dataset.
        
        Args:
            facts: List of fact dictionaries with 'id' and 'fact_text' keys
            
        Returns:
            List of chunked facts
        """
        all_chunks = []
        
        for fact in facts:
            fact_id = str(fact.get('id', ''))
            fact_text = fact.get('fact_text', '')
            
            if not fact_text:
                continue
            
            # Prepare metadata
            metadata = {k: v for k, v in fact.items() if k not in ['id', 'fact_text']}
            
            # Chunk the fact
            chunks = self.chunk_text(fact_text, fact_id, metadata)
            
            # Convert chunks to dictionaries
            for chunk in chunks:
                chunk_dict = chunk.to_dict()
                chunk_dict['fact_text'] = chunk.text  # Maintain consistency with original format
                all_chunks.append(chunk_dict)
        
        logger.info(f"Chunked {len(facts)} facts into {len(all_chunks)} chunks")
        return all_chunks


def chunk_existing_facts():
    """Utility function to chunk existing facts datasets."""
    import pandas as pd
    from pathlib import Path
    
    chunker = TextChunker()
    
    # Chunk main facts.csv
    facts_path = Path("data/facts.csv")
    if facts_path.exists():
        df = pd.read_csv(facts_path)
        facts = df.to_dict('records')
        
        chunked_facts = chunker.chunk_facts_dataset(facts)
        
        # Save chunked facts
        chunked_df = pd.DataFrame(chunked_facts)
        output_path = facts_path.parent / "facts_chunked.csv"
        chunked_df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Saved chunked facts to {output_path}")
    
    # Chunk PIB facts
    pib_path = Path("data/pib_facts.csv")
    if pib_path.exists():
        df = pd.read_csv(pib_path)
        facts = df.to_dict('records')
        
        chunked_facts = chunker.chunk_facts_dataset(facts)
        
        # Save chunked PIB facts
        chunked_df = pd.DataFrame(chunked_facts)
        output_path = pib_path.parent / "pib_facts_chunked.csv"
        chunked_df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Saved chunked PIB facts to {output_path}")


if __name__ == "__main__":
    chunk_existing_facts()