"""
Vector operations utilities
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logger import get_logger

logger = get_logger(__name__)

class VectorOperations:
    """
    Utilities for vector operations and similarity calculations
    """
    
    @staticmethod
    def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vector1).reshape(1, -1)
            v2 = np.array(vector2).reshape(1, -1)
            
            similarity = cosine_similarity(v1, v2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    @staticmethod
    def batch_cosine_similarity(
        query_vector: List[float], 
        document_vectors: List[List[float]]
    ) -> List[float]:
        """Calculate cosine similarity between query and multiple documents"""
        try:
            query = np.array(query_vector).reshape(1, -1)
            docs = np.array(document_vectors)
            
            similarities = cosine_similarity(query, docs)[0]
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating batch cosine similarity: {e}")
            return [0.0] * len(document_vectors)
    
    @staticmethod
    def euclidean_distance(vector1: List[float], vector2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors"""
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            distance = np.linalg.norm(v1 - v2)
            return float(distance)
            
        except Exception as e:
            logger.error(f"Error calculating Euclidean distance: {e}")
            return float('inf')
    
    @staticmethod
    def normalize_vector(vector: List[float]) -> List[float]:
        """Normalize vector to unit length"""
        try:
            v = np.array(vector)
            norm = np.linalg.norm(v)
            
            if norm == 0:
                return vector
            
            normalized = v / norm
            return normalized.tolist()
            
        except Exception as e:
            logger.error(f"Error normalizing vector: {e}")
            return vector
    
    @staticmethod
    def rerank_by_similarity(
        query_vector: List[float],
        documents: List[str],
        document_vectors: List[List[float]],
        metadatas: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Re-rank documents by similarity to query vector
        """
        try:
            # Calculate similarities
            similarities = VectorOperations.batch_cosine_similarity(
                query_vector, document_vectors
            )
            
            # Create tuples for sorting
            doc_sim_meta = list(zip(documents, similarities, metadatas))
            
            # Sort by similarity (descending)
            doc_sim_meta.sort(key=lambda x: x[1], reverse=True)
            
            # Take top k
            top_results = doc_sim_meta[:top_k]
            
            # Unpack results
            top_docs = [item[0] for item in top_results]
            top_similarities = [item[1] for item in top_results]
            top_metadatas = [item[2] for item in top_results]
            
            return top_docs, top_metadatas, top_similarities
            
        except Exception as e:
            logger.error(f"Error re-ranking documents: {e}")
            # Return original order if error
            return documents[:top_k], metadatas[:top_k], [0.0] * min(top_k, len(documents))
    
    @staticmethod
    def filter_by_threshold(
        documents: List[str],
        similarities: List[float],
        metadatas: List[Dict[str, Any]],
        threshold: float = 0.7
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Filter results by similarity threshold
        """
        try:
            filtered_docs = []
            filtered_meta = []
            filtered_sim = []
            
            for doc, sim, meta in zip(documents, similarities, metadatas):
                if sim >= threshold:
                    filtered_docs.append(doc)
                    filtered_sim.append(sim)
                    filtered_meta.append(meta)
            
            return filtered_docs, filtered_meta, filtered_sim
            
        except Exception as e:
            logger.error(f"Error filtering by threshold: {e}")
            return documents, metadatas, similarities
    
    @staticmethod
    def deduplicate_documents(
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        similarities: List[float],
        similarity_threshold: float = 0.95
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        Remove duplicate documents based on content similarity
        """
        try:
            unique_docs = []
            unique_meta = []
            unique_sim = []
            
            for i, (doc, meta, sim) in enumerate(zip(documents, metadatas, similarities)):
                is_duplicate = False
                
                # Check against already selected unique documents
                for unique_doc in unique_docs:
                    # Simple text similarity check
                    text_similarity = len(set(doc.split()) & set(unique_doc.split())) / len(set(doc.split()) | set(unique_doc.split()))
                    
                    if text_similarity >= similarity_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_docs.append(doc)
                    unique_meta.append(meta)
                    unique_sim.append(sim)
            
            logger.info(f"Removed {len(documents) - len(unique_docs)} duplicate documents")
            return unique_docs, unique_meta, unique_sim
            
        except Exception as e:
            logger.error(f"Error deduplicating documents: {e}")
            return documents, metadatas, similarities