"""
Document processor for various file formats
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import mimetypes

from .text_chunker import TextChunker, Chunk
from .pdf_processor import PDFProcessor
from .docx_processor import DOCXProcessor
from .markdown_processor import MarkdownProcessor
from ..utils.config_loader import ConfigLoader
from ..utils.file_utils import FileUtils
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    """
    Main document processor that handles multiple file formats
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize document processor"""
        self.config = config or ConfigLoader.load_config()
        self.processing_config = self.config.get('processing', {})
        
        # Initialize chunker
        chunk_size = self.processing_config.get('chunk_size', 1000)
        chunk_overlap = self.processing_config.get('chunk_overlap', 200)
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        
        # Initialize format processors
        self.pdf_processor = PDFProcessor()
        self.docx_processor = DOCXProcessor()
        self.markdown_processor = MarkdownProcessor()
        
        # Supported formats
        self.supported_formats = self.processing_config.get(
            'supported_formats', 
            ['.pdf', '.txt', '.md', '.docx']
        )
        
        # File size limit
        self.max_file_size_mb = self.processing_config.get('max_file_size_mb', 10)
        
        logger.info(f"Initialized DocumentProcessor with formats: {self.supported_formats}")
    
    def process_file(self, file_path: str, chunking_method: str = 'semantic') -> List[Chunk]:
        """
        Process a single file and return chunks
        """
        file_path = Path(file_path)
        
        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size
        file_size_mb = FileUtils.get_file_size_mb(str(file_path))
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB (max: {self.max_file_size_mb}MB)")
        
        # Check format support
        file_extension = FileUtils.get_file_extension(str(file_path))
        if not FileUtils.is_supported_format(str(file_path), self.supported_formats):
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Processing file: {file_path} ({file_size_mb:.1f}MB)")
        
        # Extract text based on file type
        text_content = self._extract_text(file_path)
        
        if not text_content.strip():
            logger.warning(f"No text content extracted from: {file_path}")
            return []
        
        # Create base metadata
        base_metadata = {
            'source_file': str(file_path),
            'file_name': file_path.name,
            'file_extension': file_extension,
            'file_size_mb': file_size_mb,
            'processing_timestamp': FileUtils.get_file_hash(str(file_path))  # Use as processing ID
        }
        
        # Chunk the text
        chunks = self.chunker.chunk_text(text_content, method=chunking_method)
        
        # Add base metadata to all chunks
        for chunk in chunks:
            chunk.metadata.update(base_metadata)
        
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        return chunks
    
    def process_directory(
        self, 
        directory_path: str, 
        recursive: bool = True, 
        chunking_method: str = 'semantic'
    ) -> List[Chunk]:
        """
        Process all supported files in a directory
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        logger.info(f"Processing directory: {directory_path} (recursive: {recursive})")
        
        # Find all supported files
        files = FileUtils.find_files(str(directory_path), self.supported_formats, recursive)
        
        if not files:
            logger.warning(f"No supported files found in: {directory_path}")
            return []
        
        logger.info(f"Found {len(files)} files to process")
        
        all_chunks = []
        processed_count = 0
        error_count = 0
        
        for file_path in files:
            try:
                chunks = self.process_file(str(file_path), chunking_method)
                all_chunks.extend(chunks)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    logger.info(f"Processed {processed_count}/{len(files)} files")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                error_count += 1
        
        logger.info(f"Processing complete. Processed: {processed_count}, Errors: {error_count}, Total chunks: {len(all_chunks)}")
        return all_chunks
    
    def process_text(self, text: str, source_name: str = "text_input", chunking_method: str = 'semantic') -> List[Chunk]:
        """
        Process raw text and return chunks
        """
        if not text.strip():
            return []
        
        # Create metadata
        base_metadata = {
            'source_file': source_name,
            'file_name': source_name,
            'file_extension': '.txt',
            'file_size_mb': len(text) / (1024 * 1024),
            'processing_timestamp': str(hash(text))  # Use hash as processing ID
        }
        
        # Chunk the text
        chunks = self.chunker.chunk_text(text, method=chunking_method)
        
        # Add metadata to all chunks
        for chunk in chunks:
            chunk.metadata.update(base_metadata)
        
        logger.info(f"Created {len(chunks)} chunks from text input")
        return chunks
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text content from file based on its type"""
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self.pdf_processor.extract_text(str(file_path))
            elif extension == '.docx':
                return self.docx_processor.extract_text(str(file_path))
            elif extension in ['.md', '.markdown']:
                return self.markdown_processor.extract_text(str(file_path))
            elif extension in ['.txt', '.text']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Try to read as text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
        except UnicodeDecodeError:
            try:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Could not read file {file_path}: {e}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing configuration and stats"""
        return {
            'supported_formats': self.supported_formats,
            'max_file_size_mb': self.max_file_size_mb,
            'chunk_size': self.chunker.chunk_size,
            'chunk_overlap': self.chunker.overlap
        }