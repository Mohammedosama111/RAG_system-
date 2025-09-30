"""
Rate Limiter for Gemini API
Handles RPM and daily rate limiting with intelligent waiting
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class RateLimit:
    """Rate limit configuration"""
    rpm_limit: int
    daily_limit: int
    request_history: list
    daily_usage: int
    last_reset_date: str

class RateLimiter:
    """
    Intelligent rate limiter for API requests
    """
    
    def __init__(self, rpm_limit: int = 5, daily_limit: int = 100):
        self.rpm_limit = rpm_limit
        self.daily_limit = daily_limit
        self.request_history = []
        self.daily_usage = 0
        self.last_reset_date = datetime.now().strftime('%Y-%m-%d')
        
        # Load usage from file
        self.usage_file = Path('./data/usage_stats.json')
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_usage_stats()
    
    def _load_usage_stats(self) -> None:
        """Load usage statistics from file"""
        try:
            if self.usage_file.exists():
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                
                today = datetime.now().strftime('%Y-%m-%d')
                
                # Reset daily usage if new day
                if data.get('last_reset_date') != today:
                    self.daily_usage = 0
                    self.last_reset_date = today
                else:
                    self.daily_usage = data.get('daily_usage', 0)
                    self.last_reset_date = data.get('last_reset_date', today)
                    
        except Exception:
            # If file doesn't exist or is corrupted, start fresh
            self.daily_usage = 0
            self.last_reset_date = datetime.now().strftime('%Y-%m-%d')
    
    def _save_usage_stats(self) -> None:
        """Save usage statistics to file"""
        try:
            data = {
                'daily_usage': self.daily_usage,
                'last_reset_date': self.last_reset_date,
                'rpm_limit': self.rpm_limit,
                'daily_limit': self.daily_limit
            }
            
            with open(self.usage_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save usage stats: {e}")
    
    def _clean_old_requests(self) -> None:
        """Remove requests older than 1 minute from history"""
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=1)
        
        self.request_history = [
            req_time for req_time in self.request_history 
            if req_time > cutoff_time
        ]
    
    def can_make_request(self) -> Tuple[bool, str]:
        """
        Check if a request can be made now
        Returns: (can_make_request, message)
        """
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        
        # Reset daily usage if new day
        if self.last_reset_date != today:
            self.daily_usage = 0
            self.last_reset_date = today
            self._save_usage_stats()
        
        # Check daily limit
        if self.daily_usage >= self.daily_limit:
            return False, f"Daily limit of {self.daily_limit} requests reached"
        
        # Clean old requests and check RPM limit
        self._clean_old_requests()
        
        if len(self.request_history) >= self.rpm_limit:
            # Calculate wait time until oldest request expires
            oldest_request = min(self.request_history)
            wait_until = oldest_request + timedelta(minutes=1)
            wait_seconds = (wait_until - now).total_seconds()
            
            return False, f"RPM limit reached. Wait {wait_seconds:.1f} seconds"
        
        return True, "OK"
    
    async def wait_if_needed(self) -> None:
        """Wait if rate limit requires it"""
        can_request, message = self.can_make_request()
        
        if not can_request:
            if "Wait" in message:
                # Extract wait time and wait
                wait_time = float(message.split("Wait ")[1].split(" ")[0])
                print(f"Rate limit hit. Waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time + 1)  # Add 1 second buffer
            else:
                # Daily limit reached
                raise Exception(message)
    
    def record_request(self) -> None:
        """Record that a request was made"""
        now = datetime.now()
        
        # Add to request history
        self.request_history.append(now)
        
        # Increment daily usage
        self.daily_usage += 1
        
        # Save stats
        self._save_usage_stats()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        now = datetime.now()
        self._clean_old_requests()
        
        return {
            'daily_usage': self.daily_usage,
            'daily_limit': self.daily_limit,
            'remaining_daily': self.daily_limit - self.daily_usage,
            'rpm_usage': len(self.request_history),
            'rpm_limit': self.rpm_limit,
            'remaining_rpm': max(0, self.rpm_limit - len(self.request_history)),
            'last_reset_date': self.last_reset_date
        }
    
    def reset_daily_usage(self) -> None:
        """Reset daily usage (useful for testing)"""
        self.daily_usage = 0
        self.last_reset_date = datetime.now().strftime('%Y-%m-%d')
        self._save_usage_stats()