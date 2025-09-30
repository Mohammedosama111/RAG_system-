"""
Gemini API Client with Rate Limiting and Optimization
Based on the optimization strategies from gemini_pro_optimization.md
"""

import google.generativeai as genai
import time
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .rate_limiter import RateLimiter
from ..utils.config_loader import ConfigLoader
from ..utils.logger import get_logger
from ..embeddings.local_embedding_service import LocalEmbeddingService

logger = get_logger(__name__)

@dataclass
class GeminiResponse:
    """Response data structure for Gemini API calls"""
    content: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]

class GeminiClient:
    """
    Optimized Gemini API client with intelligent rate limiting and caching
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Gemini client with configuration"""
        self.config = config or ConfigLoader.load_config()
        self.gemini_config = self.config.get('gemini', {})
        
        # Configure Gemini API
        api_key = self.gemini_config.get('api_key')
        if not api_key:
            raise ValueError("Gemini API key not found in configuration")
        
        genai.configure(api_key=api_key)
        
        # Initialize models with safety settings
        self.model_name = self.gemini_config.get('model_name', 'gemini-2.5-pro')
        self.embedding_model_name = self.gemini_config.get('embedding_model', 'models/embedding-001')
        
        # Configure safety settings to be more permissive for educational content
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ]
        
        self.model = genai.GenerativeModel(
            self.model_name,
            safety_settings=safety_settings
        )
        
        # Initialize rate limiter
        rpm_limit = self.gemini_config.get('rpm_limit', 5)
        daily_limit = self.gemini_config.get('daily_limit', 100)
        
        self.rate_limiter = RateLimiter(
            rpm_limit=rpm_limit,
            daily_limit=daily_limit
        )
        
        # Initialize local embedding service
        self.local_embeddings = LocalEmbeddingService()
        
        # Cache settings
        cache_config = self.config.get('cache', {})
        self.cache_enabled = cache_config.get('enabled', True)
        self.cache_dir = Path(cache_config.get('cache_directory', './data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Gemini client with model: {self.model_name}")
    
    def _get_cache_key(self, text: str, model_type: str = "generation") -> str:
        """Generate cache key for request"""
        content = f"{model_type}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load response from cache if available"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            cache_ttl = self.config.get('cache', {}).get('cache_ttl_hours', 24)
            
            if datetime.now() - cache_time > timedelta(hours=cache_ttl):
                cache_file.unlink()  # Remove expired cache
                return None
            
            logger.info(f"Using cached response for key: {cache_key[:10]}...")
            return cache_data['response']
            
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, response: Any) -> None:
        """Save response to cache"""
        if not self.cache_enabled:
            return
        
        cache_data = {
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    async def generate_response(self, prompt: str, **kwargs) -> GeminiResponse:
        """
        Generate response using Gemini with rate limiting and caching
        """
        # Check cache first
        cache_key = self._get_cache_key(prompt, "generation")
        cached_response = self._load_from_cache(cache_key)
        
        if cached_response:
            return GeminiResponse(
                content=cached_response['content'],
                usage=cached_response.get('usage', {}),
                metadata={'source': 'cache', 'cache_key': cache_key}
            )
        
        # Wait for rate limit if necessary
        await self.rate_limiter.wait_if_needed()
        
        try:
            generation_config = {
                'temperature': self.gemini_config.get('temperature', 0.1),
                'max_output_tokens': self.gemini_config.get('max_tokens', 1000),
                **kwargs
            }
            
            logger.info("Making Gemini API call for text generation")
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Record successful request
            self.rate_limiter.record_request()
            
            # Check if response is valid
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                logger.error(f"Gemini response blocked or empty. Finish reason: {finish_reason}")
                
                # Provide helpful error messages based on finish_reason
                if finish_reason == 2:  # SAFETY
                    error_msg = "Response was blocked due to safety filters. Please try rephrasing your question."
                elif finish_reason == 3:  # RECITATION
                    error_msg = "Response was blocked due to recitation concerns. Please try a different question."
                elif finish_reason == 4:  # OTHER
                    error_msg = "Response was blocked for other policy reasons. Please try rephrasing your question."
                else:
                    error_msg = f"Response generation failed (reason: {finish_reason}). Please try again."
                
                raise ValueError(error_msg)
            
            # Prepare response
            gemini_response = GeminiResponse(
                content=response.text,
                usage={'prompt_token_count': response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                      'candidates_token_count': response.usage_metadata.candidates_token_count if response.usage_metadata else 0},
                metadata={'model': self.model_name, 'source': 'api'}
            )
            
            # Cache the response
            self._save_to_cache(cache_key, {
                'content': gemini_response.content,
                'usage': gemini_response.usage,
                'metadata': gemini_response.metadata
            })
            
            return gemini_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using LOCAL model (no API quota limits!)
        """
        logger.info(f"🎯 Using LOCAL embeddings for {len(texts)} texts - No API quota used!")
        
        try:
            # Use local embedding service instead of Gemini API
            embeddings = self.local_embeddings.encode_texts(texts)
            logger.info(f"✅ Successfully generated {len(embeddings)} local embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating local embeddings: {e}")
            raise
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using Gemini's tokenizer"""
        try:
            result = self.model.count_tokens(text)
            return result.total_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens, using estimation: {e}")
            # Fallback estimation: ~4 characters per token
            return len(text) // 4
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        return self.rate_limiter.get_usage_stats()
    
    def reset_daily_usage(self) -> None:
        """Reset daily usage counter (useful for testing)"""
        self.rate_limiter.reset_daily_usage()