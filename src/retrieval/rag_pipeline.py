"""
Main RAG Pipeline integrating all components
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from ..gemini.client import GeminiClient
from ..vectordb.chroma_client import ChromaDBClient
from ..vectordb.vector_operations import VectorOperations
from ..processing.document_processor import DocumentProcessor
from .prompt_templates import PromptTemplates, PromptTemplate
from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class RAGResponse:
    """Response from RAG pipeline"""
    answer: str
    sources: List[str]
    metadata: List[Dict[str, Any]]
    similarities: List[float]
    processing_time: float
    tokens_used: int

class RAGPipeline:
    """
    Complete RAG pipeline integrating Gemini, ChromaDB, and document processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize RAG pipeline"""
        self.config = config or ConfigLoader.load_config()
        
        # Initialize components
        self.gemini_client = GeminiClient(self.config)
        self.vector_db = ChromaDBClient(self.config)
        self.document_processor = DocumentProcessor(self.config)
        
        # Retrieval settings
        self.retrieval_config = self.config.get('retrieval', {})
        self.top_k = self.retrieval_config.get('top_k', 5)
        self.similarity_threshold = self.retrieval_config.get('similarity_threshold', 0.7)
        self.rerank_results = self.retrieval_config.get('rerank_results', True)
        
        # Create collection
        self.vector_db.create_collection()
        
        logger.info("Initialized RAG Pipeline")
    
    async def ingest_documents(
        self, 
        file_paths: List[str], 
        chunking_method: str = 'semantic'
    ) -> Dict[str, Any]:
        """
        Ingest documents into the vector database
        """
        start_time = time.time()
        
        logger.info(f"Starting document ingestion for {len(file_paths)} files")
        
        all_chunks = []
        successful_files = []
        failed_files = []
        
        # Process each file
        for file_path in file_paths:
            try:
                chunks = self.document_processor.process_file(file_path, chunking_method)
                all_chunks.extend(chunks)
                successful_files.append(file_path)
                logger.info(f"Processed {file_path}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                failed_files.append({"file": file_path, "error": str(e)})
        
        if not all_chunks:
            logger.warning("No chunks generated from any files")
            return {
                "status": "failed",
                "message": "No chunks generated",
                "successful_files": successful_files,
                "failed_files": failed_files
            }
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
        
        texts = [chunk.text for chunk in all_chunks]
        metadatas = [chunk.metadata for chunk in all_chunks]
        
        try:
            embeddings = await self.gemini_client.generate_embeddings(texts)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return {
                "status": "failed",
                "message": f"Embedding generation failed: {e}",
                "successful_files": successful_files,
                "failed_files": failed_files
            }
        
        # Store in vector database
        try:
            self.vector_db.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"Document ingestion completed in {processing_time:.2f}s")
            
            return {
                "status": "success",
                "chunks_processed": len(all_chunks),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "processing_time": processing_time,
                "collection_stats": self.vector_db.get_collection_stats()
            }
            
        except Exception as e:
            logger.error(f"Failed to store documents: {e}")
            return {
                "status": "failed",
                "message": f"Document storage failed: {e}",
                "successful_files": successful_files,
                "failed_files": failed_files
            }
    
    async def ingest_directory(
        self, 
        directory_path: str, 
        recursive: bool = True,
        chunking_method: str = 'semantic'
    ) -> Dict[str, Any]:
        """
        Ingest all supported files from a directory
        """
        logger.info(f"Ingesting directory: {directory_path}")
        
        try:
            chunks = self.document_processor.process_directory(
                directory_path, recursive, chunking_method
            )
            
            if not chunks:
                return {
                    "status": "failed",
                    "message": "No documents found or processed"
                }
            
            # Generate embeddings and store
            texts = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            embeddings = await self.gemini_client.generate_embeddings(texts)
            
            self.vector_db.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return {
                "status": "success",
                "chunks_processed": len(chunks),
                "collection_stats": self.vector_db.get_collection_stats()
            }
            
        except Exception as e:
            logger.error(f"Directory ingestion failed: {e}")
            return {
                "status": "failed",
                "message": str(e)
            }
    
    async def query(
        self, 
        question: str, 
        prompt_type: str = 'basic',
        custom_prompt: Optional[str] = None,
        chat_history: Optional[str] = None
    ) -> RAGResponse:
        """
        Query the RAG system
        """
        start_time = time.time()
        
        logger.info(f"Processing query: {question[:100]}...")
        
        try:
            # Generate query embedding
            query_embedding = await self.gemini_client.generate_embeddings([question])
            query_vector = query_embedding[0]
            
            # Retrieve relevant documents
            results = self.vector_db.query(
                query_embeddings=[query_vector],
                n_results=self.top_k * 2  # Get more for re-ranking
            )
            
            if not results['documents'][0] or len(results['documents'][0]) == 0:
                logger.warning("No relevant documents found")
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    metadata=[],
                    similarities=[],
                    processing_time=time.time() - start_time,
                    tokens_used=0
                )
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results.get('distances', [[]])[0]
            
            # Convert distances to similarities
            # Heuristic: Chroma's default metric is cosine distance => similarity ~ (1 - distance)
            if distances:
                try:
                    raw_distances = distances
                    similarities = [max(0.0, min(1.0, 1.0 - float(d))) for d in raw_distances]
                    try:
                        logger.info(
                            f"Distance stats: min={min(raw_distances):.4f}, max={max(raw_distances):.4f}; "
                            f"similarity stats: min={min(similarities):.4f}, max={max(similarities):.4f}"
                        )
                    except Exception:
                        # Stats logging shouldn't break the flow
                        pass
                except Exception:
                    # Fallback conversion if anything goes wrong
                    similarities = [1 / (1 + float(d)) for d in distances]
            else:
                similarities = [1.0] * len(documents)
            
            # Re-rank if enabled
            if self.rerank_results and len(documents) > self.top_k:
                # Get embeddings for re-ranking (if available in results)
                if 'embeddings' in results and results['embeddings'][0] is not None and len(results['embeddings'][0]) > 0:
                    doc_embeddings = results['embeddings'][0]
                    
                    documents, metadatas, similarities = VectorOperations.rerank_by_similarity(
                        query_vector, documents, doc_embeddings, metadatas, self.top_k
                    )
                else:
                    # Simple truncation if embeddings not available
                    documents = documents[:self.top_k]
                    metadatas = metadatas[:self.top_k]
                    similarities = similarities[:self.top_k]
            
            # Filter by similarity threshold
            threshold = float(self.similarity_threshold)
            filtered_docs, filtered_meta, filtered_sim = VectorOperations.filter_by_threshold(
                documents, similarities, metadatas, threshold
            )
            
            # If nothing passes, try a relaxed threshold once (e.g., -0.05)
            if not filtered_docs and len(documents) > 0:
                relaxed = max(0.0, threshold - 0.05)
                if relaxed != threshold:
                    logger.info(f"Relaxing similarity threshold from {threshold:.2f} to {relaxed:.2f}")
                    filtered_docs, filtered_meta, filtered_sim = VectorOperations.filter_by_threshold(
                        documents, similarities, metadatas, relaxed
                    )
            
            # Fallback: if nothing passes the threshold, use top-k by similarity without threshold
            if not filtered_docs:
                logger.warning(
                    "No documents met similarity threshold; applying fallback to top-k without threshold"
                )
                # Rank by similarity desc
                ranked_indices = sorted(range(len(documents)), key=lambda i: similarities[i], reverse=True)
                k = min(self.top_k, len(documents))
                top_idx = ranked_indices[:k]
                filtered_docs = [documents[i] for i in top_idx]
                filtered_meta = [metadatas[i] for i in top_idx]
                filtered_sim = [similarities[i] for i in top_idx]
            
            if not filtered_docs:
                logger.warning("No documents met similarity threshold")
                return RAGResponse(
                    answer="I couldn't find sufficiently relevant information to answer your question confidently.",
                    sources=[],
                    metadata=[],
                    similarities=[],
                    processing_time=time.time() - start_time,
                    tokens_used=0
                )
            
            # Build context
            context = self._build_context(filtered_docs, filtered_meta)
            
            # Generate response
            if custom_prompt:
                prompt = custom_prompt.format(context=context, query=question)
            else:
                template = PromptTemplates.get_template_by_type(prompt_type)
                
                variables = {
                    'context': context,
                    'query': question
                }
                
                if chat_history:
                    variables['chat_history'] = chat_history
                
                prompt = PromptTemplates.format_prompt(template, variables)
            
            # Generate answer
            response = await self.gemini_client.generate_response(prompt)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Query processed in {processing_time:.2f}s")
            
            return RAGResponse(
                answer=response.content,
                sources=filtered_docs,
                metadata=filtered_meta,
                similarities=filtered_sim,
                processing_time=processing_time,
                tokens_used=response.usage.get('total_tokens', 0)
            )
            
        except ValueError as e:
            # Handle Gemini API specific errors (safety filters, etc.)
            logger.warning(f"Query blocked by safety filters: {e}")
            processing_time = time.time() - start_time
            
            return RAGResponse(
                answer=str(e),  # This will contain the user-friendly message
                sources=filtered_docs if 'filtered_docs' in locals() else [],
                metadata=filtered_meta if 'filtered_meta' in locals() else [],
                similarities=filtered_sim if 'filtered_sim' in locals() else [],
                processing_time=processing_time,
                tokens_used=0
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            processing_time = time.time() - start_time
            
            return RAGResponse(
                answer=f"An error occurred while processing your query: {str(e)}",
                sources=[],
                metadata=[],
                similarities=[],
                processing_time=processing_time,
                tokens_used=0
            )
    
    def _build_context(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved documents
        """
        context_parts = []
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            source_info = meta.get('source_file', f'Document {i+1}')
            
            context_part = f"--- Source: {source_info} ---\n{doc}\n"
            context_parts.append(context_part)
        
        return '\n'.join(context_parts)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'vector_db_stats': self.vector_db.get_collection_stats(),
            'gemini_usage': self.gemini_client.get_usage_stats(),
            'processing_stats': self.document_processor.get_processing_stats(),
            'retrieval_config': {
                'top_k': self.top_k,
                'similarity_threshold': self.similarity_threshold,
                'rerank_results': self.rerank_results
            }
        }
    
    async def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a single text to the vector database
        """
        try:
            chunks = self.document_processor.process_text(text, "direct_input")
            
            if not chunks:
                return False
            
            # Add custom metadata if provided
            if metadata:
                for chunk in chunks:
                    chunk.metadata.update(metadata)
            
            texts = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            embeddings = await self.gemini_client.generate_embeddings(texts)
            
            self.vector_db.add_documents(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add text: {e}")
            return False