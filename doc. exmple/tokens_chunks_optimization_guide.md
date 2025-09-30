# Complete Guide: Tokens, Chunks & RAG Optimization

## Understanding Tokens

### What Are Tokens?
Tokens are the smallest units that language models process. They can be:
- **Whole words**: "hello" = 1 token
- **Parts of words**: "playing" = 2 tokens ("play" + "ing")
- **Characters**: In some cases, special characters or symbols
- **Spaces and punctuation**: Each counts separately

### Token Calculation Rules

#### General Rules (Gemini & Most LLMs):
- **1 token ≈ 4 characters** (rough estimate)
- **1 token ≈ 0.75 words** (for English text)
- **100 words ≈ 130-150 tokens** (typical range)

#### Gemini-Specific Token Counting:
```python
# Method 1: Using Gemini API's count_tokens
import google.generativeai as genai

def count_tokens_gemini(text, model="gemini-2.5-pro"):
    """Accurate token count using Gemini API"""
    genai.configure(api_key="your-api-key")
    model = genai.GenerativeModel(model)
    
    response = model.count_tokens(text)
    return response.total_tokens

# Method 2: Local estimation (no API call)
def estimate_tokens_local(text):
    """Quick local estimation"""
    # Rough calculation: 1 token = 4 characters
    return len(text) // 4

# Method 3: More accurate local estimation
def estimate_tokens_accurate(text):
    """More accurate local estimation"""
    import re
    
    # Split by whitespace and punctuation
    words = re.findall(r'\S+', text)
    
    token_count = 0
    for word in words:
        # Longer words tend to be split into more tokens
        if len(word) <= 4:
            token_count += 1
        elif len(word) <= 8:
            token_count += 2
        else:
            token_count += len(word) // 4
    
    return token_count

# Example usage
text = "This is a sample text for tokenization testing purposes."
print(f"Characters: {len(text)}")
print(f"Words: {len(text.split())}")
print(f"Estimated tokens (simple): {estimate_tokens_local(text)}")
print(f"Estimated tokens (accurate): {estimate_tokens_accurate(text)}")
```

## Understanding Chunks in RAG

### What Are Chunks?
Chunks are segments of your documents that are:
1. **Small enough** for embedding models to process effectively
2. **Large enough** to contain meaningful context
3. **Semantically coherent** - contain related information

### Why Chunking Matters:
- **Embedding models** have input limits (typically 512-8192 tokens)
- **Context windows** of LLMs are finite (1M tokens for Gemini)
- **Search precision** improves with focused, relevant chunks
- **Processing speed** increases with smaller, manageable pieces

## Optimal Chunking Strategies

### 1. Fixed-Size Chunking (Basic)
```python
def fixed_size_chunking(text, chunk_size=1000, overlap=200):
    """Basic fixed-size chunking with overlap"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Don't break in the middle of words
        if end < len(text) and not text[end].isspace():
            # Find the last space before the cutoff
            last_space = chunk.rfind(' ')
            if last_space > chunk_size * 0.8:  # Only if we don't lose too much
                end = start + last_space
                chunk = text[start:end]
        
        chunks.append({
            'text': chunk.strip(),
            'start_pos': start,
            'end_pos': end,
            'token_count': estimate_tokens_accurate(chunk)
        })
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks
```

### 2. Semantic Chunking (Advanced)
```python
import nltk
from sentence_transformers import SentenceTransformer

def semantic_chunking(text, max_chunk_size=1000, similarity_threshold=0.7):
    """Chunk based on semantic similarity"""
    
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Get sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = estimate_tokens_accurate(sentence)
        
        # Check if adding this sentence would exceed size limit
        if current_size + sentence_tokens > max_chunk_size and current_chunk:
            # Start new chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'token_count': current_size,
                'sentences': len(current_chunk),
                'semantic_coherence': calculate_coherence(current_chunk, model)
            })
            current_chunk = [sentence]
            current_size = sentence_tokens
        else:
            # Add to current chunk
            current_chunk.append(sentence)
            current_size += sentence_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'token_count': current_size,
            'sentences': len(current_chunk),
            'semantic_coherence': calculate_coherence(current_chunk, model)
        })
    
    return chunks

def calculate_coherence(sentences, model):
    """Calculate semantic coherence within a chunk"""
    if len(sentences) < 2:
        return 1.0
    
    embeddings = model.encode(sentences)
    
    # Calculate average cosine similarity between sentences
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            similarities.append(sim)
    
    return sum(similarities) / len(similarities) if similarities else 0.0
```

### 3. Hierarchical Chunking (Professional)
```python
def hierarchical_chunking(text, max_levels=3):
    """Create multiple levels of chunks for different granularities"""
    
    chunks_by_level = {}
    
    # Level 1: Large chunks (paragraphs)
    paragraphs = text.split('\n\n')
    level1_chunks = []
    for i, para in enumerate(paragraphs):
        if len(para.strip()) > 0:
            chunk = {
                'text': para.strip(),
                'level': 1,
                'id': f"L1_{i}",
                'parent_id': None,
                'token_count': estimate_tokens_accurate(para),
                'type': 'paragraph'
            }
            level1_chunks.append(chunk)
    
    chunks_by_level[1] = level1_chunks
    
    # Level 2: Medium chunks (sentences within paragraphs)
    level2_chunks = []
    for l1_chunk in level1_chunks:
        sentences = nltk.sent_tokenize(l1_chunk['text'])
        
        for j, sentence in enumerate(sentences):
            if len(sentence.strip()) > 0:
                chunk = {
                    'text': sentence.strip(),
                    'level': 2,
                    'id': f"{l1_chunk['id']}_S{j}",
                    'parent_id': l1_chunk['id'],
                    'token_count': estimate_tokens_accurate(sentence),
                    'type': 'sentence'
                }
                level2_chunks.append(chunk)
    
    chunks_by_level[2] = level2_chunks
    
    # Level 3: Small chunks (phrases within sentences)
    level3_chunks = []
    for l2_chunk in level2_chunks:
        phrases = l2_chunk['text'].split(', ')
        
        for k, phrase in enumerate(phrases):
            if len(phrase.strip()) > 20:  # Only meaningful phrases
                chunk = {
                    'text': phrase.strip(),
                    'level': 3,
                    'id': f"{l2_chunk['id']}_P{k}",
                    'parent_id': l2_chunk['id'],
                    'token_count': estimate_tokens_accurate(phrase),
                    'type': 'phrase'
                }
                level3_chunks.append(chunk)
    
    chunks_by_level[3] = level3_chunks
    
    return chunks_by_level
```

## Token & Chunk Optimization Calculations

### Gemini Context Window Optimization
```python
class GeminiChunkOptimizer:
    def __init__(self, model="gemini-2.5-pro"):
        self.model = model
        self.max_context_tokens = 1_000_000  # Gemini 2.5 Pro limit
        self.max_embedding_tokens = 8192     # Typical embedding limit
        self.response_buffer = 2048          # Reserve for response
        
    def calculate_optimal_chunk_size(self, num_chunks_to_retrieve=5):
        """Calculate optimal chunk size based on context limits"""
        
        # Available tokens for retrieved context
        available_for_context = self.max_context_tokens - self.response_buffer
        
        # Reserve space for system prompt, query, formatting
        system_overhead = 500
        available_for_chunks = available_for_context - system_overhead
        
        # Optimal tokens per chunk
        tokens_per_chunk = available_for_chunks // num_chunks_to_retrieve
        
        # Convert to characters (rough approximation)
        chars_per_chunk = tokens_per_chunk * 4
        
        return {
            'optimal_tokens_per_chunk': tokens_per_chunk,
            'optimal_chars_per_chunk': chars_per_chunk,
            'max_retrievable_chunks': num_chunks_to_retrieve,
            'total_context_budget': available_for_context
        }
    
    def optimize_chunk_parameters(self, document_length, target_chunks=None):
        """Calculate optimal chunking parameters for a document"""
        
        if target_chunks is None:
            # Aim for 20-50 chunks per document for good granularity
            target_chunks = max(20, min(50, document_length // 2000))
        
        optimal_chunk_size = document_length // target_chunks
        
        # Ensure chunk size is within embedding model limits
        max_chunk_chars = self.max_embedding_tokens * 4
        optimal_chunk_size = min(optimal_chunk_size, max_chunk_chars)
        
        # Calculate overlap (10-20% is optimal)
        optimal_overlap = int(optimal_chunk_size * 0.15)
        
        return {
            'chunk_size': optimal_chunk_size,
            'overlap': optimal_overlap,
            'expected_chunks': target_chunks,
            'chunk_size_tokens': optimal_chunk_size // 4
        }

# Example usage
optimizer = GeminiChunkOptimizer()

# For retrieval optimization
retrieval_params = optimizer.calculate_optimal_chunk_size(num_chunks_to_retrieve=7)
print("Optimal retrieval parameters:")
for key, value in retrieval_params.items():
    print(f"  {key}: {value:,}")

# For document chunking optimization
document_text = "Your document content here..." * 1000  # Example long document
doc_params = optimizer.optimize_chunk_parameters(len(document_text))
print("\nOptimal chunking parameters:")
for key, value in doc_params.items():
    print(f"  {key}: {value:,}")
```

### Advanced Chunk Quality Metrics
```python
def evaluate_chunk_quality(chunks, embeddings_model):
    """Evaluate the quality of your chunking strategy"""
    
    metrics = {
        'total_chunks': len(chunks),
        'avg_chunk_size': 0,
        'size_variance': 0,
        'semantic_coherence': [],
        'information_density': [],
        'boundary_quality': []
    }
    
    sizes = [chunk['token_count'] for chunk in chunks]
    metrics['avg_chunk_size'] = sum(sizes) / len(sizes)
    metrics['size_variance'] = sum((s - metrics['avg_chunk_size'])**2 for s in sizes) / len(sizes)
    
    for i, chunk in enumerate(chunks):
        # Semantic coherence within chunk
        sentences = nltk.sent_tokenize(chunk['text'])
        if len(sentences) > 1:
            coherence = calculate_semantic_coherence(sentences, embeddings_model)
            metrics['semantic_coherence'].append(coherence)
        
        # Information density (unique words / total words)
        words = chunk['text'].lower().split()
        if words:
            density = len(set(words)) / len(words)
            metrics['information_density'].append(density)
        
        # Boundary quality (are sentences cut off?)
        boundary_quality = evaluate_boundary_quality(chunk['text'])
        metrics['boundary_quality'].append(boundary_quality)
    
    # Calculate averages
    for metric in ['semantic_coherence', 'information_density', 'boundary_quality']:
        if metrics[metric]:
            metrics[f'avg_{metric}'] = sum(metrics[metric]) / len(metrics[metric])
    
    return metrics

def evaluate_boundary_quality(text):
    """Check if chunk boundaries are clean (don't cut sentences)"""
    if not text:
        return 0.0
    
    # Check if starts with capital letter or continues from previous
    starts_clean = text[0].isupper() or text.startswith((' ', '\n'))
    
    # Check if ends with proper punctuation
    ends_clean = text.rstrip()[-1] in '.!?'
    
    return (starts_clean + ends_clean) / 2
```

## Practical Optimization Strategies

### Strategy 1: Dynamic Chunk Sizing
```python
def dynamic_chunk_sizing(text, base_size=1000):
    """Adjust chunk size based on content type"""
    
    # Detect content type
    def detect_content_type(text_sample):
        if text_sample.count('\n') / len(text_sample) > 0.05:
            return 'structured'  # Code, data, etc.
        elif text_sample.count('.') / len(text_sample.split()) > 0.15:
            return 'technical'   # Academic, technical docs
        else:
            return 'narrative'   # Stories, articles
    
    content_type = detect_content_type(text[:1000])
    
    # Adjust chunk size based on content
    if content_type == 'structured':
        chunk_size = base_size // 2  # Smaller chunks for structured content
        overlap = 50
    elif content_type == 'technical':
        chunk_size = base_size * 1.5  # Larger chunks for technical content
        overlap = 300
    else:
        chunk_size = base_size
        overlap = 200
    
    return semantic_chunking(text, max_chunk_size=chunk_size)
```

### Strategy 2: Query-Aware Chunking
```python
def query_aware_chunking(text, expected_query_types):
    """Optimize chunking based on expected query patterns"""
    
    chunks = []
    
    if 'definition' in expected_query_types:
        # Create smaller, concept-focused chunks
        chunks.extend(create_definition_chunks(text))
    
    if 'comparison' in expected_query_types:
        # Create chunks that maintain comparative context
        chunks.extend(create_comparison_chunks(text))
    
    if 'procedural' in expected_query_types:
        # Create step-by-step chunks
        chunks.extend(create_procedural_chunks(text))
    
    # Remove duplicates and merge overlapping chunks
    return deduplicate_chunks(chunks)

def create_definition_chunks(text):
    """Create chunks optimized for definition queries"""
    chunks = []
    
    # Look for definition patterns
    import re
    definition_patterns = [
        r'(.+?) is (.+?)\.',
        r'(.+?) refers to (.+?)\.',
        r'(.+?) means (.+?)\.',
        r'Define (.+?) as (.+?)\.'
    ]
    
    for pattern in definition_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            definition_text = f"{match[0]} is {match[1]}."
            chunks.append({
                'text': definition_text,
                'type': 'definition',
                'token_count': estimate_tokens_accurate(definition_text)
            })
    
    return chunks
```

### Strategy 3: Performance Monitoring
```python
def monitor_chunking_performance(chunks, queries, retrieval_results):
    """Monitor how well your chunking strategy performs"""
    
    performance_metrics = {
        'retrieval_accuracy': [],
        'response_quality': [],
        'chunk_utilization': defaultdict(int),
        'avg_retrieved_chunks': 0
    }
    
    for query, results in zip(queries, retrieval_results):
        # Track which chunks are being retrieved most often
        for result in results['sources']:
            chunk_id = result.get('chunk_id', 'unknown')
            performance_metrics['chunk_utilization'][chunk_id] += 1
        
        # Calculate average chunks per query
        performance_metrics['avg_retrieved_chunks'] = len(results['sources'])
        
        # Measure retrieval accuracy (manual evaluation needed)
        accuracy_score = evaluate_retrieval_accuracy(query, results)
        performance_metrics['retrieval_accuracy'].append(accuracy_score)
    
    return performance_metrics

def suggest_chunking_improvements(performance_metrics, chunks):
    """Suggest improvements based on performance data"""
    suggestions = []
    
    # Check for underutilized chunks
    total_retrievals = sum(performance_metrics['chunk_utilization'].values())
    underutilized = [chunk_id for chunk_id, count in performance_metrics['chunk_utilization'].items() 
                    if count < total_retrievals * 0.01]  # Less than 1% utilization
    
    if underutilized:
        suggestions.append(f"Consider merging or removing {len(underutilized)} underutilized chunks")
    
    # Check chunk size distribution
    avg_accuracy = sum(performance_metrics['retrieval_accuracy']) / len(performance_metrics['retrieval_accuracy'])
    if avg_accuracy < 0.7:
        suggestions.append("Low retrieval accuracy - consider smaller, more focused chunks")
    
    return suggestions
```

## Best Practices Summary

### Optimal Parameters for Different Use Cases:

1. **General Documents (Articles, Reports)**:
   - Chunk size: 800-1200 tokens
   - Overlap: 10-20% (80-240 tokens)
   - Strategy: Semantic chunking with sentence boundaries

2. **Technical Documentation**:
   - Chunk size: 1200-2000 tokens  
   - Overlap: 20-25% (240-500 tokens)
   - Strategy: Section-aware chunking

3. **Code Documentation**:
   - Chunk size: 400-800 tokens
   - Overlap: 5-10% (20-80 tokens)
   - Strategy: Function/class-based chunking

4. **Academic Papers**:
   - Chunk size: 1000-1500 tokens
   - Overlap: 15-20% (150-300 tokens)
   - Strategy: Paragraph + citation-aware chunking

### Key Optimization Rules:
1. **Test different chunk sizes** with your specific documents
2. **Monitor retrieval performance** and adjust accordingly
3. **Use overlap strategically** - more for complex topics
4. **Consider content type** when choosing strategy
5. **Balance chunk size** with embedding model limits
6. **Optimize for your query types** - definitions need smaller chunks, analysis needs larger ones

This comprehensive approach will help you create an optimized RAG system that provides better, more relevant responses!