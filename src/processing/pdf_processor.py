"""
PDF text extraction processor
"""

import PyPDF2
from pathlib import Path
from typing import Optional
import re

from ..utils.logger import get_logger

logger = get_logger(__name__)

class PDFProcessor:
    """
    PDF text extraction with various strategies
    """
    
    def __init__(self):
        self.encoding_fallbacks = ['utf-8', 'latin-1', 'cp1252']
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF file using PyPDF2
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    logger.warning(f"PDF is encrypted: {file_path}")
                    return ""
                
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        
                        if page_text.strip():
                            # Clean the extracted text
                            cleaned_text = self._clean_pdf_text(page_text)
                            text_content.append(cleaned_text)
                            
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1} of {file_path}: {e}")
                        continue
                
                full_text = '\n\n'.join(text_content)
                logger.info(f"Extracted text from {len(pdf_reader.pages)} pages in {Path(file_path).name}")
                
                return full_text
                
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return ""
    
    def _clean_pdf_text(self, text: str) -> str:
        """
        Clean and normalize text extracted from PDF
        """
        if not text:
            return ""
        
        # Remove excessive whitespace and fix line breaks
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix common PDF extraction issues
        text = text.replace('\x00', '')  # Remove null characters
        text = text.replace('', '')      # Remove replacement characters
        
        # Fix hyphenated words at line endings
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*\n', r'\1\n', text)
        
        return text.strip()
    
    def extract_metadata(self, file_path: str) -> dict:
        """
        Extract metadata from PDF
        """
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Basic information
                metadata['page_count'] = len(pdf_reader.pages)
                metadata['is_encrypted'] = pdf_reader.is_encrypted
                
                # Document metadata
                if pdf_reader.metadata:
                    doc_info = pdf_reader.metadata
                    
                    metadata['title'] = str(doc_info.get('/Title', '')) if doc_info.get('/Title') else ''
                    metadata['author'] = str(doc_info.get('/Author', '')) if doc_info.get('/Author') else ''
                    metadata['subject'] = str(doc_info.get('/Subject', '')) if doc_info.get('/Subject') else ''
                    metadata['creator'] = str(doc_info.get('/Creator', '')) if doc_info.get('/Creator') else ''
                    metadata['producer'] = str(doc_info.get('/Producer', '')) if doc_info.get('/Producer') else ''
                    
                    # Creation and modification dates
                    if doc_info.get('/CreationDate'):
                        metadata['creation_date'] = str(doc_info.get('/CreationDate'))
                    if doc_info.get('/ModDate'):
                        metadata['modification_date'] = str(doc_info.get('/ModDate'))
        
        except Exception as e:
            logger.error(f"Error extracting PDF metadata from {file_path}: {e}")
        
        return metadata
    
    def is_text_extractable(self, file_path: str) -> bool:
        """
        Check if text can be extracted from PDF (not just images)
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if pdf_reader.is_encrypted:
                    return False
                
                # Check first few pages for extractable text
                pages_to_check = min(3, len(pdf_reader.pages))
                
                for i in range(pages_to_check):
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text and len(page_text.strip()) > 50:
                        return True
                
                return False
                
        except Exception:
            return False