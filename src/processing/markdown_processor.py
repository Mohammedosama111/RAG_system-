"""
Markdown text extraction processor
"""

import markdown
from pathlib import Path
import re
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)

class MarkdownProcessor:
    """
    Markdown text extraction with optional HTML conversion
    """
    
    def __init__(self):
        # Initialize markdown processor with common extensions
        self.md_processor = markdown.Markdown(
            extensions=[
                'extra',
                'codehilite',
                'toc',
                'tables',
                'fenced_code'
            ]
        )
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract plain text from markdown file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Convert to plain text (remove markdown formatting)
            plain_text = self._markdown_to_text(markdown_content)
            
            logger.info(f"Extracted text from {Path(file_path).name}")
            return plain_text
            
        except Exception as e:
            logger.error(f"Error processing Markdown {file_path}: {e}")
            return ""
    
    def extract_with_structure(self, file_path: str) -> str:
        """
        Extract text while preserving document structure
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Process with structure preservation
            structured_text = self._preserve_structure(markdown_content)
            
            return structured_text
            
        except Exception as e:
            logger.error(f"Error processing structured Markdown {file_path}: {e}")
            return ""
    
    def _markdown_to_text(self, markdown_content: str) -> str:
        """
        Convert markdown to plain text
        """
        # Remove markdown formatting
        text = markdown_content
        
        # Remove headers (# ## ###)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold and italic (**text** *text*)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove links [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove images ![alt](url)
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
        
        # Remove inline code `code`
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove code blocks ```
        text = re.sub(r'```[\s\S]*?```', '', text, flags=re.MULTILINE)
        
        # Remove horizontal rules
        text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\*\*\*+$', '', text, flags=re.MULTILINE)
        
        # Remove blockquotes
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        
        # Clean up lists
        text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _preserve_structure(self, markdown_content: str) -> str:
        """
        Convert markdown while preserving document structure
        """
        text = markdown_content
        
        # Convert headers to structured format
        text = re.sub(r'^(#{1,6})\s+(.+)$', lambda m: f"\n{'='*len(m.group(1))} {m.group(2).upper()} {'='*len(m.group(1))}\n", text, flags=re.MULTILINE)
        
        # Preserve code blocks with labels
        text = re.sub(r'```(\w+)?\n([\s\S]*?)```', r'\nCODE BLOCK (\1):\n\2\nEND CODE BLOCK\n', text, flags=re.MULTILINE)
        
        # Convert lists to structured format
        text = re.sub(r'^([\s]*)[-*+]\s+(.+)$', r'\1• \2', text, flags=re.MULTILINE)
        text = re.sub(r'^([\s]*)\d+\.\s+(.+)$', r'\1\2', text, flags=re.MULTILINE)
        
        # Convert blockquotes
        text = re.sub(r'^>\s*(.+)$', r'QUOTE: \1', text, flags=re.MULTILINE)
        
        # Preserve links with context
        text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\1 (Link: \2)', text)
        
        # Remove other markdown but keep emphasis
        text = re.sub(r'\*\*([^*]+)\*\*', r'BOLD: \1', text)
        text = re.sub(r'\*([^*]+)\*', r'EMPHASIS: \1', text)
        text = re.sub(r'`([^`]+)`', r'CODE: \1', text)
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def extract_metadata(self, file_path: str) -> dict:
        """
        Extract metadata from markdown file
        """
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract YAML front matter if present
            yaml_match = re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)
            if yaml_match:
                try:
                    import yaml
                    yaml_content = yaml.safe_load(yaml_match.group(1))
                    if isinstance(yaml_content, dict):
                        metadata.update(yaml_content)
                except ImportError:
                    logger.warning("PyYAML not installed, skipping YAML front matter parsing")
                except Exception as e:
                    logger.warning(f"Error parsing YAML front matter: {e}")
            
            # Extract titles from headers
            titles = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            if titles:
                metadata['main_title'] = titles[0][1].strip()
                metadata['all_headers'] = [title[1].strip() for title in titles]
            
            # Count elements
            metadata['header_count'] = len(titles)
            metadata['code_block_count'] = len(re.findall(r'```', content)) // 2
            metadata['link_count'] = len(re.findall(r'\[([^\]]+)\]\([^\)]+\)', content))
            
            # Check for tables
            metadata['has_tables'] = '|' in content and re.search(r'\|.*\|', content) is not None
            
        except Exception as e:
            logger.error(f"Error extracting Markdown metadata from {file_path}: {e}")
        
        return metadata
    
    def to_html(self, file_path: str) -> str:
        """
        Convert markdown file to HTML
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            html_content = self.md_processor.convert(markdown_content)
            return html_content
            
        except Exception as e:
            logger.error(f"Error converting Markdown to HTML {file_path}: {e}")
            return ""