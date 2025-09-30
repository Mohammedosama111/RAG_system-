"""
Gemini API client module
Handles interactions with Google's Gemini API for embeddings and text generation
"""

from .client import GeminiClient
from .rate_limiter import RateLimiter

__all__ = ["GeminiClient", "RateLimiter"]