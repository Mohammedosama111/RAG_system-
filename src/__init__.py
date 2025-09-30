"""
RAG System with Gemini API
A comprehensive Retrieval-Augmented Generation system using Google's Gemini API
"""

__version__ = "1.0.0"
__author__ = "RAG System Developer"

from .gemini import GeminiClient
from .vectordb import ChromaDBClient
from .processing import DocumentProcessor
from .retrieval import RAGPipeline
from .embeddings import LocalEmbeddingService

__all__ = [
    "GeminiClient",
    "ChromaDBClient", 
    "DocumentProcessor",
    "RAGPipeline",
    "LocalEmbeddingService"
]