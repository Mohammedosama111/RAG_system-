"""
Text chunking strategies based on tokens_chunks_optimization_guide.md
"""

import re
import nltk
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK data: {e}")

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    start_index: int
    end_index: int
    token_count: int
    metadata: Dict[str, Any]

class TextChunker:
    """
    Advanced text chunking strategies optimized for RAG systems
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using improved method from the guide
        """
        # More accurate local estimation
        words = re.findall(r'\S+', text)
        
        token_count = 0
        for word in words:
            if len(word) <= 4:
                token_count += 1
            elif len(word) <= 8:
                token_count += 2
            else:
                token_count += len(word) // 4
        
        return token_count
    
    def fixed_size_chunking(self, text: str) -> List[Chunk]:
        """
        Basic fixed-size chunking with overlap
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Don't break in the middle of words
            if end < len(text) and not text[end].isspace():
                # Find the last space before the cutoff
                last_space = chunk_text.rfind(' ')
                if last_space > start + (self.chunk_size * 0.7):  # At least 70% of chunk size
                    end = start + last_space
                    chunk_text = text[start:end]
            
            chunk = Chunk(
                text=chunk_text.strip(),
                start_index=start,
                end_index=end,
                token_count=self.estimate_tokens(chunk_text),
                metadata={
                    'chunk_method': 'fixed_size',
                    'chunk_index': len(chunks)
                }
            )
            
            chunks.append(chunk)
            
            start = end - self.overlap
            if start >= len(text):
                break
        
        return chunks
    
    def semantic_chunking(self, text: str, similarity_threshold: float = 0.7) -> List[Chunk]:
        """
        Semantic chunking based on sentence similarity
        """
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            # Fallback to simple sentence splitting
            sentences = text.split('. ')
        
        chunks = []
        current_chunk = []
        current_size = 0
        start_pos = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.estimate_tokens(sentence)
            
            # Check if adding this sentence would exceed size limit
            if current_size + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                chunk = Chunk(
                    text=chunk_text,
                    start_index=start_pos,
                    end_index=start_pos + len(chunk_text),
                    token_count=current_size,
                    metadata={
                        'chunk_method': 'semantic',
                        'chunk_index': len(chunks),
                        'sentence_count': len(current_chunk)
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.overlap > 0 and len(current_chunk) > 1:
                    # Keep some sentences for overlap
                    overlap_sentences = max(1, len(current_chunk) // 3)
                    current_chunk = current_chunk[-overlap_sentences:]
                    current_size = sum(self.estimate_tokens(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
                    start_pos = start_pos + len(chunk_text)
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = Chunk(
                text=chunk_text,
                start_index=start_pos,
                end_index=start_pos + len(chunk_text),
                token_count=current_size,
                metadata={
                    'chunk_method': 'semantic',
                    'chunk_index': len(chunks),
                    'sentence_count': len(current_chunk)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def paragraph_chunking(self, text: str) -> List[Chunk]:
        """
        Chunk by paragraphs, respecting natural document structure
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        start_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_tokens = self.estimate_tokens(para)
            
            # If paragraph alone exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                # First, add current chunk if it has content
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = Chunk(
                        text=chunk_text,
                        start_index=start_pos,
                        end_index=start_pos + len(chunk_text),
                        token_count=current_size,
                        metadata={
                            'chunk_method': 'paragraph',
                            'chunk_index': len(chunks),
                            'paragraph_count': len(current_chunk)
                        }
                    )
                    chunks.append(chunk)
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph using fixed chunking
                para_chunks = self.fixed_size_chunking(para)
                for para_chunk in para_chunks:
                    para_chunk.metadata['chunk_method'] = 'paragraph_split'
                    para_chunk.metadata['chunk_index'] = len(chunks)
                    chunks.append(para_chunk)
                
                start_pos += len(para) + 2  # +2 for \n\n
                
            elif current_size + para_tokens > self.chunk_size and current_chunk:
                # Create chunk from current paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                chunk = Chunk(
                    text=chunk_text,
                    start_index=start_pos,
                    end_index=start_pos + len(chunk_text),
                    token_count=current_size,
                    metadata={
                        'chunk_method': 'paragraph',
                        'chunk_index': len(chunks),
                        'paragraph_count': len(current_chunk)
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = [para]
                current_size = para_tokens
                start_pos = start_pos + len(chunk_text) + 2
                
            else:
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_size += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk = Chunk(
                text=chunk_text,
                start_index=start_pos,
                end_index=start_pos + len(chunk_text),
                token_count=current_size,
                metadata={
                    'chunk_method': 'paragraph',
                    'chunk_index': len(chunks),
                    'paragraph_count': len(current_chunk)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def adaptive_chunking(self, text: str, content_type: str = 'general') -> List[Chunk]:
        """
        Adaptive chunking based on content type (from optimization guide)
        """
        # Detect content type if not provided
        if content_type == 'auto':
            content_type = self._detect_content_type(text)
        
        # Adjust parameters based on content type
        original_chunk_size = self.chunk_size
        original_overlap = self.overlap
        
        if content_type == 'technical':
            self.chunk_size = int(original_chunk_size * 1.5)  # Larger chunks for technical content
            self.overlap = 300
        elif content_type == 'structured':
            self.chunk_size = int(original_chunk_size * 0.5)  # Smaller chunks for structured content
            self.overlap = 50
        elif content_type == 'narrative':
            # Use paragraph chunking for narrative content
            chunks = self.paragraph_chunking(text)
        else:
            chunks = self.semantic_chunking(text)
        
        # Restore original settings
        self.chunk_size = original_chunk_size
        self.overlap = original_overlap
        
        # Update metadata with content type
        if 'chunks' in locals():
            for chunk in chunks:
                chunk.metadata['content_type'] = content_type
        
        return chunks
    
    def _detect_content_type(self, text: str) -> str:
        """
        Simple content type detection
        """
        # Count structural elements
        line_breaks = text.count('\n') / len(text) if len(text) > 0 else 0
        bullet_points = len(re.findall(r'^[\s]*[-*•]\s', text, re.MULTILINE))
        code_blocks = len(re.findall(r'```|`[^`]+`', text))
        
        if line_breaks > 0.05:  # High line break ratio
            return 'structured'
        elif code_blocks > 5 or 'def ' in text or 'class ' in text:
            return 'technical'
        elif bullet_points > 3:
            return 'structured'
        else:
            return 'narrative'
    
    def chunk_text(self, text: str, method: str = 'semantic') -> List[Chunk]:
        """
        Main chunking method that delegates to specific strategies
        """
        if method == 'fixed':
            return self.fixed_size_chunking(text)
        elif method == 'semantic':
            return self.semantic_chunking(text)
        elif method == 'paragraph':
            return self.paragraph_chunking(text)
        elif method == 'adaptive':
            return self.adaptive_chunking(text, 'auto')
        else:
            logger.warning(f"Unknown chunking method: {method}, using semantic")
            return self.semantic_chunking(text)