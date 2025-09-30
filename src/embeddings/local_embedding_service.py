"""
Local Embedding Service using Sentence Transformers
Replaces Gemini embeddings with free local model
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)

class LocalEmbeddingService:
    """
    Local embedding service using all-MiniLM-L6-v2 model
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize local embedding model"""
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        logger.info(f"Initializing local embedding service with model: {model_name}")
    
    def _load_model(self) -> None:
        """Load the model on first use (lazy loading)"""
        if self.model is None:
            try:
                logger.info(f"Loading {self.model_name} model... (this may take a moment on first run)")
                self.model = SentenceTransformer(self.model_name)
                logger.info("✅ Local embedding model loaded successfully!")
                
                # Log model info
                logger.info(f"Model dimension: {self.embedding_dim}")
                logger.info(f"Model max sequence length: {self.model.get_max_seq_length()}")
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise RuntimeError(f"Could not load embedding model {self.model_name}: {e}")
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts to embeddings
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            List of embedding vectors (list of floats)
        """
        if not texts:
            return []
        
        self._load_model()
        
        try:
            logger.info(f"Encoding {len(texts)} texts with local model...")
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10  # Show progress for large batches
            )
            
            # Convert to list format for compatibility
            embedding_list = embeddings.tolist()
            
            logger.info(f"✅ Successfully encoded {len(texts)} texts locally")
            return embedding_list
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise RuntimeError(f"Failed to encode texts: {e}")
    
    def encode_query(self, query: str) -> List[float]:
        """
        Encode a single query text
        
        Args:
            query: Query text to encode
            
        Returns:
            Embedding vector as list of floats
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return [0.0] * self.embedding_dim
        
        self._load_model()
        
        try:
            logger.info("Encoding query with local model...")
            
            # Generate embedding for single query
            embedding = self.model.encode([query], convert_to_numpy=True)[0]
            
            logger.info("✅ Query encoded successfully")
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            raise RuntimeError(f"Failed to encode query: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {
                "model_name": self.model_name,
                "status": "not_loaded",
                "embedding_dimension": self.embedding_dim
            }
        
        return {
            "model_name": self.model_name,
            "status": "loaded",
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.model.get_max_seq_length(),
            "model_type": "local_transformer"
        }
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def test_model(self) -> Dict[str, Any]:
        """
        Test the model with sample texts
        
        Returns:
            Test results
        """
        logger.info("Testing local embedding model...")
        
        test_texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Python is a popular programming language"
        ]
        
        try:
            # Test encoding
            embeddings = self.encode_texts(test_texts)
            
            # Test similarity
            sim1_2 = self.similarity(embeddings[0], embeddings[1])  # Should be high (both ML)
            sim1_3 = self.similarity(embeddings[0], embeddings[2])  # Should be lower
            
            results = {
                "status": "success",
                "model_info": self.get_model_info(),
                "test_embeddings_count": len(embeddings),
                "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                "similarity_ml_dl": round(sim1_2, 3),
                "similarity_ml_python": round(sim1_3, 3),
                "test_passed": sim1_2 > sim1_3  # ML-DL should be more similar than ML-Python
            }
            
            logger.info("✅ Model test completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }