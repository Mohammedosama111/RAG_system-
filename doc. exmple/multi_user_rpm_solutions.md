# Multi-User RPM Management: Complete Solutions

## The RPM Challenge Explained

### Single API Key = Shared Rate Limit
- **5 RPM for Gemini Pro** = shared across ALL users
- **10 RPM for Gemini Flash** = shared across ALL users
- Each user query typically uses 2-3 API calls:
  1. Query embedding generation (1 RPM)
  2. Response generation (1 RPM)
  3. Sometimes document re-ranking (1 RPM)

### Real-World Impact:
```
With 5 RPM limit:
- 2-3 simultaneous users = Immediate bottleneck
- User 4+ gets "rate limit exceeded" errors
- Everyone waits 60 seconds for reset
```

## Solution 1: Advanced Request Queue System

### Intelligent Queue Manager
```python
# src/queue/request_queue.py
import asyncio
import time
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from typing import Optional
import uuid

@dataclass
class QueuedRequest:
    id: str
    user_id: str
    query: str
    context: str
    timestamp: datetime
    priority: int = 0
    retries: int = 0

class IntelligentRequestQueue:
    def __init__(self, rpm_limit=5, max_queue_size=50):
        self.rpm_limit = rpm_limit
        self.max_queue_size = max_queue_size
        self.request_queue = deque()
        self.processing_queue = deque()
        self.request_history = []
        self.active_requests = {}
        
        # Start background processor
        asyncio.create_task(self.process_queue())
    
    def can_process_now(self) -> tuple[bool, float]:
        """Check if we can process request now"""
        now = datetime.now()
        
        # Clean old requests (older than 1 minute)
        cutoff_time = now - timedelta(minutes=1)
        self.request_history = [req_time for req_time in self.request_history if req_time > cutoff_time]
        
        if len(self.request_history) < self.rpm_limit:
            return True, 0
        
        # Calculate wait time until oldest request expires
        wait_until = self.request_history[0] + timedelta(minutes=1)
        wait_seconds = (wait_until - now).total_seconds()
        return False, max(0, wait_seconds)
    
    async def add_request(self, user_id: str, query: str, context: str = "", priority: int = 0) -> str:
        """Add request to queue"""
        if len(self.request_queue) >= self.max_queue_size:
            raise Exception("Queue is full. Please try again later.")
        
        request_id = str(uuid.uuid4())
        request = QueuedRequest(
            id=request_id,
            user_id=user_id,
            query=query,
            context=context,
            timestamp=datetime.now(),
            priority=priority
        )
        
        # Insert based on priority (higher priority = processed first)
        if priority > 0:
            # Find insertion point for priority requests
            insertion_point = 0
            for i, req in enumerate(self.request_queue):
                if req.priority < priority:
                    insertion_point = i
                    break
                insertion_point = i + 1
            
            self.request_queue.insert(insertion_point, request)
        else:
            self.request_queue.append(request)
        
        self.active_requests[request_id] = {
            'status': 'queued',
            'position': len(self.request_queue),
            'estimated_wait': self.estimate_wait_time(len(self.request_queue))
        }
        
        return request_id
    
    def estimate_wait_time(self, queue_position: int) -> int:
        """Estimate wait time in seconds"""
        requests_per_minute = self.rpm_limit
        minutes_to_wait = queue_position / requests_per_minute
        return int(minutes_to_wait * 60)
    
    async def process_queue(self):
        """Background task to process queued requests"""
        while True:
            if not self.request_queue:
                await asyncio.sleep(1)
                continue
            
            can_process, wait_time = self.can_process_now()
            
            if not can_process:
                print(f"Rate limit hit. Waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time + 0.1)  # Small buffer
                continue
            
            # Process next request
            request = self.request_queue.popleft()
            self.active_requests[request.id]['status'] = 'processing'
            
            try:
                # Process the request (your RAG pipeline here)
                result = await self.process_single_request(request)
                
                self.active_requests[request.id] = {
                    'status': 'completed',
                    'result': result,
                    'completed_at': datetime.now()
                }
                
                # Record successful request
                self.request_history.append(datetime.now())
                
            except Exception as e:
                request.retries += 1
                if request.retries < 3:
                    # Retry with lower priority
                    request.priority = max(0, request.priority - 1)
                    self.request_queue.appendleft(request)
                    self.active_requests[request.id]['status'] = 'retrying'
                else:
                    self.active_requests[request.id] = {
                        'status': 'failed',
                        'error': str(e),
                        'failed_at': datetime.now()
                    }
            
            # Update queue positions for remaining requests
            for i, req in enumerate(self.request_queue):
                if req.id in self.active_requests:
                    self.active_requests[req.id]['position'] = i + 1
                    self.active_requests[req.id]['estimated_wait'] = self.estimate_wait_time(i + 1)
    
    async def process_single_request(self, request: QueuedRequest):
        """Process individual request - implement your RAG logic here"""
        # This is where you'd call your actual RAG pipeline
        # For now, simulating processing time
        await asyncio.sleep(2)  # Simulate API call time
        
        return {
            'answer': f"Processed query: {request.query}",
            'processing_time': 2.0,
            'user_id': request.user_id
        }
    
    def get_request_status(self, request_id: str) -> Optional[dict]:
        """Get status of a specific request"""
        return self.active_requests.get(request_id)
    
    def get_user_requests(self, user_id: str) -> list:
        """Get all requests for a specific user"""
        user_requests = []
        
        # Check queue
        for req in self.request_queue:
            if req.user_id == user_id:
                user_requests.append({
                    'id': req.id,
                    'status': 'queued',
                    'query': req.query,
                    'position': self.active_requests[req.id]['position']
                })
        
        # Check completed/failed requests
        for req_id, status in self.active_requests.items():
            if req_id not in [r['id'] for r in user_requests]:
                # This is a processed request, check if it belongs to user
                # You'd need to store user_id in active_requests for this to work
                user_requests.append({
                    'id': req_id,
                    'status': status['status'],
                    'result': status.get('result', status.get('error'))
                })
        
        return user_requests
    
    def get_queue_stats(self) -> dict:
        """Get current queue statistics"""
        return {
            'queue_length': len(self.request_queue),
            'current_rpm_usage': len(self.request_history),
            'rpm_limit': self.rpm_limit,
            'estimated_wait_for_new_request': self.estimate_wait_time(len(self.request_queue) + 1),
            'active_requests': len([req for req in self.active_requests.values() if req['status'] == 'processing'])
        }
```

## Solution 2: Aggressive Caching + Pre-computation

### Smart Cache with User Context
```python
# src/cache/multi_user_cache.py
import hashlib
import json
import time
from typing import Optional, List
from collections import defaultdict

class MultiUserSmartCache:
    def __init__(self, cache_dir="./data/user_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # User-specific caches
        self.user_cache = defaultdict(dict)
        self.global_cache = {}
        self.popular_queries = defaultdict(int)
        
        self.load_popular_queries()
    
    def load_popular_queries(self):
        """Load frequently asked questions"""
        try:
            with open(self.cache_dir / "popular_queries.json", 'r') as f:
                self.popular_queries = defaultdict(int, json.load(f))
        except FileNotFoundError:
            pass
    
    def save_popular_queries(self):
        """Save popular queries for pre-computation"""
        with open(self.cache_dir / "popular_queries.json", 'w') as f:
            json.dump(dict(self.popular_queries), f)
    
    def get_semantic_cache_key(self, query: str) -> str:
        """Generate semantic cache key"""
        # Normalize query
        normalized = query.lower().strip()
        # Remove common words
        stop_words = {'what', 'how', 'when', 'where', 'why', 'is', 'are', 'the', 'a', 'an'}
        words = [w for w in normalized.split() if w not in stop_words]
        semantic_key = ' '.join(sorted(words))
        return hashlib.md5(semantic_key.encode()).hexdigest()
    
    def find_similar_cached_response(self, query: str, similarity_threshold: float = 0.8) -> Optional[dict]:
        """Find semantically similar cached responses"""
        query_key = self.get_semantic_cache_key(query)
        
        # Check exact semantic match first
        if query_key in self.global_cache:
            cached_data = self.global_cache[query_key]
            if time.time() - cached_data['timestamp'] < 3600:  # 1 hour freshness
                return cached_data
        
        # Check for partial matches
        query_words = set(query.lower().split())
        best_match = None
        best_similarity = 0
        
        for cached_key, cached_data in self.global_cache.items():
            if time.time() - cached_data['timestamp'] > 3600:
                continue
                
            cached_words = set(cached_data['original_query'].lower().split())
            intersection = len(query_words.intersection(cached_words))
            union = len(query_words.union(cached_words))
            
            if union > 0:
                similarity = intersection / union
                if similarity > best_similarity and similarity >= similarity_threshold:
                    best_similarity = similarity
                    best_match = cached_data
        
        return best_match
    
    def cache_response(self, user_id: str, query: str, response: dict):
        """Cache response with user context"""
        semantic_key = self.get_semantic_cache_key(query)
        
        cache_data = {
            'original_query': query,
            'response': response,
            'timestamp': time.time(),
            'user_id': user_id,
            'usage_count': 1
        }
        
        # Store in global cache
        if semantic_key in self.global_cache:
            self.global_cache[semantic_key]['usage_count'] += 1
        else:
            self.global_cache[semantic_key] = cache_data
        
        # Store in user cache
        self.user_cache[user_id][semantic_key] = cache_data
        
        # Update popular queries
        self.popular_queries[semantic_key] += 1
        if self.popular_queries[semantic_key] % 5 == 0:  # Save every 5 uses
            self.save_popular_queries()
    
    def get_cached_response(self, user_id: str, query: str) -> Optional[dict]:
        """Get cached response for user"""
        # Try user-specific cache first
        semantic_key = self.get_semantic_cache_key(query)
        
        if semantic_key in self.user_cache[user_id]:
            cached_data = self.user_cache[user_id][semantic_key]
            if time.time() - cached_data['timestamp'] < 7200:  # 2 hours for user cache
                return cached_data['response']
        
        # Try global semantic cache
        similar_response = self.find_similar_cached_response(query)
        if similar_response:
            return similar_response['response']
        
        return None
    
    def get_popular_queries_for_precomputation(self, limit: int = 20) -> List[str]:
        """Get most popular queries for pre-computation"""
        sorted_queries = sorted(self.popular_queries.items(), key=lambda x: x[1], reverse=True)
        
        popular_queries = []
        for semantic_key, count in sorted_queries[:limit]:
            # Find original query from cache
            if semantic_key in self.global_cache:
                popular_queries.append(self.global_cache[semantic_key]['original_query'])
        
        return popular_queries
    
    def precompute_popular_responses(self, rag_pipeline, max_precompute: int = 10):
        """Pre-compute responses for popular queries during off-peak"""
        popular_queries = self.get_popular_queries_for_precomputation(max_precompute)
        
        for query in popular_queries:
            semantic_key = self.get_semantic_cache_key(query)
            
            # Check if we need fresh computation
            if semantic_key in self.global_cache:
                cached_data = self.global_cache[semantic_key]
                if time.time() - cached_data['timestamp'] < 1800:  # 30 minutes
                    continue  # Still fresh
            
            print(f"Pre-computing response for: {query}")
            try:
                # This uses your API quota, but during off-peak hours
                response = rag_pipeline.retrieve_and_generate(query)
                self.cache_response("system", query, response)
                
                # Add delay to respect rate limits
                time.sleep(15)  # Wait 15 seconds between pre-computations
                
            except Exception as e:
                print(f"Pre-computation failed for '{query}': {e}")
```

## Solution 3: Hybrid Local + API System

### Combine Local Models with Gemini Pro
```python
# src/hybrid/hybrid_rag.py
from sentence_transformers import SentenceTransformer
import ollama

class HybridRAGSystem:
    def __init__(self, gemini_client, use_local_embeddings=True, use_local_llm_fallback=True):
        self.gemini_client = gemini_client
        self.use_local_embeddings = use_local_embeddings
        self.use_local_llm_fallback = use_local_llm_fallback
        
        # Local models for when API is rate-limited
        if use_local_embeddings:
            self.local_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        if use_local_llm_fallback:
            try:
                # Check if Ollama is available
                ollama.list()  # This will throw if Ollama isn't running
                self.local_llm_available = True
            except:
                self.local_llm_available = False
                print("Ollama not available. Gemini-only mode.")
    
    def generate_embeddings_smart(self, texts: List[str], prefer_local=False):
        """Smart embedding generation - local first if rate limited"""
        
        if prefer_local and self.use_local_embeddings:
            print("Using local embeddings to save API quota")
            return self.local_embedding_model.encode(texts).tolist()
        
        try:
            # Try Gemini first
            return self.gemini_client.generate_embeddings(texts)
        except Exception as e:
            if "rate limit" in str(e).lower() and self.use_local_embeddings:
                print("Rate limited. Switching to local embeddings.")
                return self.local_embedding_model.encode(texts).tolist()
            raise
    
    def generate_response_smart(self, query: str, context: str, prefer_local=False):
        """Smart response generation with fallback"""
        
        if prefer_local and self.local_llm_available:
            return self.generate_local_response(query, context)
        
        try:
            # Try Gemini Pro first
            prompt = f"""Based on the following context, answer the question:

Context: {context}

Question: {query}

Answer:"""
            return self.gemini_client.generate_response(prompt)
            
        except Exception as e:
            if "rate limit" in str(e).lower() and self.local_llm_available:
                print("Gemini rate limited. Using local LLM.")
                return self.generate_local_response(query, context)
            raise
    
    def generate_local_response(self, query: str, context: str):
        """Generate response using local Ollama model"""
        prompt = f"""Context: {context}

Question: {query}

Based on the context above, provide a clear and accurate answer:"""
        
        response = ollama.generate(
            model='llama2',  # or 'mistral', 'codellama'
            prompt=prompt,
            options={
                'temperature': 0.1,
                'num_predict': 512
            }
        )
        
        return response['response']
```

## Solution 4: User Priority System

### VIP vs Regular Users
```python
# src/users/priority_manager.py
from enum import Enum
from datetime import datetime, timedelta

class UserTier(Enum):
    FREE = 0
    PREMIUM = 1
    VIP = 2

class UserPriorityManager:
    def __init__(self):
        self.user_tiers = {}
        self.user_usage = defaultdict(list)
        self.tier_limits = {
            UserTier.FREE: {'daily_queries': 5, 'priority': 0},
            UserTier.PREMIUM: {'daily_queries': 50, 'priority': 1},
            UserTier.VIP: {'daily_queries': 200, 'priority': 2}
        }
    
    def set_user_tier(self, user_id: str, tier: UserTier):
        """Set user tier"""
        self.user_tiers[user_id] = tier
    
    def get_user_priority(self, user_id: str) -> int:
        """Get user priority for queue ordering"""
        tier = self.user_tiers.get(user_id, UserTier.FREE)
        return self.tier_limits[tier]['priority']
    
    def can_user_make_request(self, user_id: str) -> tuple[bool, str]:
        """Check if user can make request based on tier limits"""
        tier = self.user_tiers.get(user_id, UserTier.FREE)
        daily_limit = self.tier_limits[tier]['daily_queries']
        
        # Count today's usage
        today = datetime.now().date()
        today_usage = [req_time for req_time in self.user_usage[user_id] 
                      if req_time.date() == today]
        
        if len(today_usage) >= daily_limit:
            return False, f"Daily limit reached ({daily_limit} queries)"
        
        return True, "OK"
    
    def record_user_request(self, user_id: str):
        """Record that user made a request"""
        self.user_usage[user_id].append(datetime.now())
        
        # Clean old usage data (keep only last 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        self.user_usage[user_id] = [req_time for req_time in self.user_usage[user_id] 
                                   if req_time > cutoff]
```

## Implementation Strategy:

### Phase 1: Immediate (Queue System)
1. Implement request queue with priority
2. Add smart caching
3. User notification system ("Your request is #3 in queue, estimated wait: 2 minutes")

### Phase 2: Optimization (Hybrid System)  
1. Add local embedding models
2. Pre-compute popular queries
3. Implement user tiers

### Phase 3: Scaling (Multiple API Keys)
1. Get additional API keys for different services
2. Load balancing across multiple keys
3. Separate keys for embeddings vs generation

The key insight: **Turn the rate limit from a problem into a feature** by managing user expectations and providing excellent queue management!