"""
ChromaDB client for vector storage and retrieval
"""

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import uuid

from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ChromaDBClient:
    """
    ChromaDB client for vector operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ChromaDB client"""
        self.config = config or ConfigLoader.load_config()
        self.vectordb_config = self.config.get('vectordb', {})
        
        # Setup ChromaDB
        persist_directory = self.vectordb_config.get('persist_directory', './data/vectordb')
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection_name = self.vectordb_config.get('collection_name', 'documents')
        self.collection: Optional[Collection] = None
        
        logger.info(f"Initialized ChromaDB client at: {persist_directory}")
    
    def create_collection(self, collection_name: Optional[str] = None) -> Collection:
        """Create or get collection"""
        name = collection_name or self.collection_name
        
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=name)
            logger.info(f"Retrieved existing collection: {name}")
        except Exception:
            # Create new collection if it doesn't exist
            self.collection = self.client.create_collection(name=name)
            logger.info(f"Created new collection: {name}")
        
        return self.collection
    
    def get_collection(self, collection_name: Optional[str] = None) -> Optional[Collection]:
        """Get existing collection"""
        name = collection_name or self.collection_name
        
        try:
            collection = self.client.get_collection(name=name)
            return collection
        except Exception as e:
            logger.warning(f"Collection {name} not found: {e}")
            return None

    def get_persist_directory(self) -> str:
        """Return the Chroma persist directory path from config"""
        return self.vectordb_config.get('persist_directory', './data/vectordb')

    def ensure_collection(self, collection_name: Optional[str] = None) -> Collection:
        """Ensure self.collection is set to an existing (or newly created) collection"""
        if self.collection is not None and (collection_name is None or self.collection.name == (collection_name or self.collection_name)):
            return self.collection
        col = self.get_collection(collection_name)
        if col is None:
            col = self.create_collection(collection_name)
        self.collection = col
        return col

    def get_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch documents from the collection with pagination.

        Returns a dict with keys: ids, documents, metadatas, embeddings
        """
        col = self.ensure_collection(collection_name)
        if include is None:
            include = ["documents", "metadatas", "embeddings"]
        try:
            result = col.get(limit=limit, offset=offset, where=where, include=include)
            return result
        except Exception as e:
            logger.error(f"Error fetching documents: {e}")
            raise
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to collection"""
        if not self.collection:
            self.create_collection()
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(texts)} documents to collection")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Query the collection for similar documents"""
        if not self.collection:
            raise ValueError("No collection available. Create or get collection first.")
        
        if include is None:
            include = ['documents', 'metadatas', 'distances', 'embeddings']
        
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=include
            )
            
            logger.info(f"Query returned {len(results.get('documents', [[]])[0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            raise
    
    def update_documents(
        self,
        ids: List[str],
        texts: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Update existing documents"""
        if not self.collection:
            raise ValueError("No collection available")
        
        try:
            self.collection.update(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Updated {len(ids)} documents")
            
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            raise
    
    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents by IDs"""
        if not self.collection:
            raise ValueError("No collection available")
        
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.collection:
            return {"error": "No collection available"}
        
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.vectordb_config.get('persist_directory')
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """Delete a collection"""
        name = collection_name or self.collection_name
        
        try:
            self.client.delete_collection(name=name)
            if self.collection and self.collection.name == name:
                self.collection = None
            logger.info(f"Deleted collection: {name}")
            
        except Exception as e:
            logger.error(f"Error deleting collection {name}: {e}")
            raise