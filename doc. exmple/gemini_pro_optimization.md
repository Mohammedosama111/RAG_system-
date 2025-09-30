# Gemini Pro RAG System: Optimization Strategy

## Pro-Specific Configuration

### Optimized Rate Limiting for Pro
```python
# src/gemini/gemini_pro_client.py
import google.generativeai as genai
import time
from datetime import datetime, timedelta
import json

class GeminiProClient:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.embedding_model = "models/embedding-001"
        
        # Pro-specific rate limiting
        self.rpm_limit = 5  # 5 requests per minute for Pro
        self.daily_limit = 100  # Daily request limit
        self.request_history = []
        self.daily_usage = self.load_daily_usage()
        
    def load_daily_usage(self):
        """Load today's usage from file"""
        try:
            with open('./data/daily_usage.json', 'r') as f:
                usage_data = json.load(f)
                today = datetime.now().strftime('%Y-%m-%d')
                return usage_data.get(today, 0)
        except FileNotFoundError:
            return 0
    
    def save_daily_usage(self):
        """Save daily usage to file"""
        try:
            with open('./data/daily_usage.json', 'r') as f:
                usage_data = json.load(f)
        except FileNotFoundError:
            usage_data = {}
        
        today = datetime.now().strftime('%Y-%m-%d')
        usage_data[today] = self.daily_usage
        
        with open('./data/daily_usage.json', 'w') as f:
            json.dump(usage_data, f)
    
    def can_make_request(self):
        """Check if we can make a request based on rate limits"""
        now = datetime.now()
        
        # Check daily limit
        if self.daily_usage >= self.daily_limit:
            return False, "Daily limit reached"
        
        # Clean old requests (older than 1 minute)
        cutoff_time = now - timedelta(minutes=1)
        self.request_history = [req_time for req_time in self.request_history if req_time > cutoff_time]
        
        # Check RPM limit
        if len(self.request_history) >= self.rpm_limit:
            wait_time = 60 - (now - self.request_history[0]).total_seconds()
            return False, f"Rate limit exceeded. Wait {wait_time:.1f} seconds"
        
        return True, "OK"
    
    def make_request_with_limits(self, request_func, *args, **kwargs):
        """Make request with automatic rate limiting"""
        can_request, message = self.can_make_request()
        
        if not can_request:
            if "Wait" in message:
                wait_time = float(message.split("Wait ")[1].split(" ")[0])
                print(f"Rate limit hit. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time + 1)  # Add 1 second buffer
            else:
                raise Exception(message)
        
        # Make the request
        now = datetime.now()
        try:
            result = request_func(*args, **kwargs)
            self.request_history.append(now)
            self.daily_usage += 1
            self.save_daily_usage()
            return result
        except Exception as e:
            print(f"API Error: {e}")
            # Don't count failed requests
            raise
    
    def generate_response(self, prompt):
        """Generate response with Pro model"""
        def _generate():
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 2048,
                    'top_p': 0.8,
                    'top_k': 40
                }
            )
            return response.text
        
        return self.make_request_with_limits(_generate)
    
    def generate_embeddings(self, texts):
        """Generate embeddings with rate limiting"""
        embeddings = []
        
        for i, text in enumerate(texts):
            print(f"Processing embedding {i+1}/{len(texts)}")
            
            def _embed():
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            
            embedding = self.make_request_with_limits(_embed)
            embeddings.append(embedding)
        
        return embeddings
    
    def get_usage_stats(self):
        """Get current usage statistics"""
        return {
            'daily_usage': self.daily_usage,
            'daily_limit': self.daily_limit,
            'current_rpm_usage': len(self.request_history),
            'rpm_limit': self.rpm_limit,
            'remaining_daily': self.daily_limit - self.daily_usage
        }
```

## Smart Caching Strategy for Pro

### Intelligent Response Caching
```python
# src/utils/pro_cache.py
import hashlib
import json
import time
from pathlib import Path

class ProResponseCache:
    def __init__(self, cache_dir="./data/pro_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load cache metadata"""
        try:
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        except FileNotFoundError:
            self.metadata = {
                'total_cached_responses': 0,
                'cache_hit_rate': 0.0,
                'api_calls_saved': 0
            }
    
    def save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_cache_key(self, query, context):
        """Generate cache key from query and context"""
        # Create a more sophisticated cache key
        content = f"query:{query}\ncontext:{context[:1000]}"  # First 1000 chars of context
        return hashlib.sha256(content.encode()).hexdigest()
    
    def is_similar_query(self, query, cached_query, similarity_threshold=0.8):
        """Check if queries are similar enough to use cache"""
        # Simple similarity check - you could use more sophisticated methods
        query_words = set(query.lower().split())
        cached_words = set(cached_query.lower().split())
        
        if not query_words or not cached_words:
            return False
        
        intersection = len(query_words.intersection(cached_words))
        union = len(query_words.union(cached_words))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= similarity_threshold
    
    def get_cached_response(self, query, context):
        """Get cached response with fuzzy matching"""
        cache_key = self.get_cache_key(query, context)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Try exact match first
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                # Check if cache is still fresh (24 hours)
                if time.time() - cached_data['timestamp'] < 86400:
                    return cached_data['response']
        
        # Try fuzzy matching for similar queries
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name == "cache_metadata.json":
                continue
                
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if self.is_similar_query(query, cached_data['query']):
                        return cached_data['response']
            except (json.JSONDecodeError, KeyError):
                continue
        
        return None
    
    def cache_response(self, query, context, response):
        """Cache the response"""
        cache_key = self.get_cache_key(query, context)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cache_data = {
            'query': query,
            'context_preview': context[:500],  # Store preview for debugging
            'response': response,
            'timestamp': time.time()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        # Update metadata
        self.metadata['total_cached_responses'] += 1
        self.save_metadata()
    
    def clear_old_cache(self, max_age_hours=24):
        """Clear cache entries older than specified hours"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        for cache_file in self.cache_dir.glob("*.json"):
            if cache_file.name == "cache_metadata.json":
                continue
                
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data['timestamp'] < cutoff_time:
                        cache_file.unlink()
            except (json.JSONDecodeError, KeyError):
                cache_file.unlink()  # Remove corrupted files
```

## Pro-Optimized RAG Pipeline

### Context-Aware Retrieval
```python
# src/rag_pipeline/pro_retriever.py
class ProRAGPipeline:
    def __init__(self, gemini_client, vector_db, cache):
        self.gemini_client = gemini_client
        self.vector_db = vector_db
        self.cache = cache
        
    def retrieve_and_generate(self, query, top_k=10, use_advanced_synthesis=True):
        """Enhanced retrieval with Pro's capabilities"""
        
        # Check cache first
        cached_response = self.cache.get_cached_response(query, "")
        if cached_response:
            print("Using cached response")
            return cached_response
        
        # Enhanced retrieval with larger context
        query_embedding = self.gemini_client.generate_query_embedding(query)
        results = self.vector_db.query(query_embedding, n_results=top_k)
        
        # Pro can handle much larger context - use it!
        context = self.build_rich_context(results, query)
        
        if use_advanced_synthesis:
            prompt = self.create_advanced_synthesis_prompt(query, context)
        else:
            prompt = self.create_standard_prompt(query, context)
        
        response = self.gemini_client.generate_response(prompt)
        
        # Cache the response
        self.cache.cache_response(query, context[:1000], {
            'answer': response,
            'sources': results['documents'][0],
            'metadata': results['metadatas'][0]
        })
        
        return {
            'answer': response,
            'sources': results['documents'][0],
            'metadata': results['metadatas'][0]
        }
    
    def build_rich_context(self, results, query):
        """Build comprehensive context using Pro's large context window"""
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # Pro can handle much more context
        rich_context = f"Query: {query}\n\nRelevant Information:\n\n"
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            source = meta.get('source', f'Document {i+1}')
            rich_context += f"--- Source: {source} ---\n{doc}\n\n"
        
        return rich_context
    
    def create_advanced_synthesis_prompt(self, query, context):
        """Create sophisticated prompt leveraging Pro's reasoning"""
        return f"""You are an expert analyst with access to multiple information sources. Your task is to provide a comprehensive, well-reasoned answer based on the provided context.

{context}

QUERY: {query}

INSTRUCTIONS:
1. Analyze all the provided sources carefully
2. Identify key themes, patterns, and relationships across sources
3. Synthesize information from multiple sources where relevant
4. Highlight any conflicting information and explain potential reasons
5. Provide a structured, comprehensive answer
6. Cite specific sources for key claims
7. If information is insufficient, clearly state what additional information would be helpful

RESPONSE FORMAT:
- Start with a direct answer to the query
- Provide supporting evidence from the sources
- Include analysis of relationships between different pieces of information
- End with any important caveats or limitations

Answer:"""
    
    def create_standard_prompt(self, query, context):
        """Standard prompt for simpler queries"""
        return f"""Based on the following context, provide a clear and accurate answer to the user's question.

Context:
{context}

Question: {query}

Instructions:
- Use only information from the provided context
- Be specific and cite relevant sources
- If the answer isn't available, say so clearly
- Provide a well-structured response

Answer:"""
```

## Pro Usage Optimization Tips

### 1. Batch Operations Strategically
```python
# Process documents during off-peak hours
def process_documents_efficiently():
    """Process multiple documents with optimal timing"""
    documents = get_pending_documents()
    
    # Process 5 documents per session (respects daily limit)
    daily_batch_size = 5
    
    for i in range(0, min(len(documents), daily_batch_size)):
        doc = documents[i]
        print(f"Processing document {i+1}/{daily_batch_size}")
        
        # Process document
        chunks = process_document(doc)
        embeddings = gemini_client.generate_embeddings(chunks)
        
        # Store in vector database
        store_in_vectordb(chunks, embeddings, doc.metadata)
        
        # Show remaining quota
        stats = gemini_client.get_usage_stats()
        print(f"Daily usage: {stats['daily_usage']}/{stats['daily_limit']}")
        
        if stats['remaining_daily'] < 10:
            print("Approaching daily limit. Stopping processing.")
            break
```

### 2. Advanced Query Optimization
```python
def optimize_query_for_pro(user_query):
    """Enhance queries to leverage Pro's capabilities"""
    
    # For complex queries, add analytical instructions
    if any(word in user_query.lower() for word in ['compare', 'analyze', 'relationship', 'trend']):
        return f"""
        {user_query}
        
        Please provide a comprehensive analysis including:
        - Key findings and insights
        - Relationships between different aspects
        - Any patterns or trends you identify
        - Comparative analysis where relevant
        """
    
    return user_query
```

## Expected Performance with Pro:

### Advantages:
- **Higher quality responses**: More nuanced understanding
- **Better synthesis**: Can connect information across many sources  
- **Deeper analysis**: More sophisticated reasoning capabilities
- **Larger context handling**: Can work with entire documents

### Rate Limit Management:
- **5 RPM**: Plan for 12-second intervals between requests
- **100 daily requests**: Enough for ~50-80 user interactions daily
- **Cache everything**: Essential to maximize the daily allowance

### Best Use Cases for Pro:
- **Research applications**: Where quality matters more than speed
- **Complex analysis**: Multi-document synthesis and comparison
- **Professional content**: High-stakes applications requiring accuracy
- **Educational tools**: Where comprehensive answers are needed

This setup maximizes your Gemini Pro investment while working efficiently within the rate limits!