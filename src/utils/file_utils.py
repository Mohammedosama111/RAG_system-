"""
File utility functions
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib

class FileUtils:
    """Utility functions for file operations"""
    
    @staticmethod
    def ensure_directory(path: str) -> Path:
        """Ensure directory exists, create if it doesn't"""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Get file extension from path"""
        return Path(file_path).suffix.lower()
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """Get file size in MB"""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    @staticmethod
    def is_supported_format(file_path: str, supported_formats: List[str]) -> bool:
        """Check if file format is supported"""
        extension = FileUtils.get_file_extension(file_path)
        return extension in supported_formats
    
    @staticmethod
    def get_file_hash(file_path: str, algorithm: str = 'md5') -> str:
        """Get file hash for deduplication"""
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def find_files(directory: str, extensions: List[str], recursive: bool = True) -> List[Path]:
        """Find all files with specified extensions in directory"""
        directory_path = Path(directory)
        files = []
        
        if not directory_path.exists():
            return files
        
        pattern = "**/*" if recursive else "*"
        
        for ext in extensions:
            files.extend(directory_path.glob(f"{pattern}{ext}"))
        
        return files
    
    @staticmethod
    def safe_filename(filename: str) -> str:
        """Create a safe filename by removing/replacing invalid characters"""
        invalid_chars = '<>:"/\\|?*'
        safe_name = filename
        
        for char in invalid_chars:
            safe_name = safe_name.replace(char, '_')
        
        return safe_name
    
    @staticmethod
    def get_relative_path(file_path: str, base_path: str) -> str:
        """Get relative path from base path"""
        return os.path.relpath(file_path, base_path)