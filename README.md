# RAG System with Gemini API

A comprehensive Retrieval-Augmented Generation (RAG) system powered by Google's Gemini API, featuring intelligent document processing, vector search, and optimized response generation.

## ✨ Features

### Core Capabilities
- 🤖 **Gemini API Integration** - Embeddings & text generation with Gemini-2.5-Pro/Flash
- 🗃️ **Vector Database** - ChromaDB for efficient similarity search
- 📄 **Multi-format Support** - PDF, DOCX, Markdown, and TXT processing
- 🧠 **Smart Chunking** - Semantic, paragraph, fixed, and adaptive strategies
- 💬 **Conversational AI** - Context-aware responses with chat history
- 🎯 **Multiple Prompt Types** - Basic, advanced, technical, comparative, and more

### Advanced Features
- ⚡ **Rate Limiting** - Intelligent API quota management
- 💾 **Smart Caching** - Response and embedding caching
- 🔄 **Re-ranking** - Vector similarity re-ranking for better results
- 📊 **Analytics** - Comprehensive usage statistics and monitoring
- 🌐 **Web Interface** - Streamlit-based user interface
- 💻 **CLI Tool** - Command-line interface for batch operations

## 🚀 Quick Start

### 1. Installation (Windows PowerShell)

```powershell
# Clone the repository
git clone <repository-url>
cd RAG_system-

# Create and activate a virtual environment (named "venv")
python -m venv venv
./venv/Scripts/Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Optional: run the guided setup (creates folders, checks imports, downloads NLTK data)
# You can skip this if you prefer manual steps
python setup.py
```

### 2. Configuration

1. Get your Gemini API key from Google AI Studio
2. Create or edit a `.env` file in the project root and add:
  ```
  GEMINI_API_KEY=your-api-key-here
  ENVIRONMENT=development
  LOG_LEVEL=INFO
  ```
  Note: If a `.env` file already exists, just update `GEMINI_API_KEY`.

### 3. Run the Application

#### Web Interface (Recommended)
```powershell
# If your venv is activated
streamlit run app.py

# Or, without activating the venv
./venv/Scripts/streamlit.exe run app.py
```

#### Command Line Interface
```powershell
# Get help
python cli.py --help

# Add documents
python cli.py ingest --directory ./data/documents

# Ask questions
python cli.py query "What is machine learning?"

# Interactive mode
python cli.py interactive

# Or, without activating the venv
./venv/Scripts/python.exe cli.py --help
```

#### Quick Example
```powershell
python examples/quick_start.py
```

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Text           │    │   Gemini API    │
│   Processing    │───▶│   Chunking       │───▶│   Embeddings    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Metadata      │    │   Vector         │    │   ChromaDB      │
│   Extraction    │    │   Storage        │    │   Search        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
         ┌─────────────────────────────────────────────────────┐
         │              RAG Pipeline                           │
         │  Query → Retrieval → Context → Generation → Answer │
         └─────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
RAG_system-/
├── src/                          # Source code
│   ├── gemini/                   # Gemini API client
│   │   ├── client.py            # Main Gemini client
│   │   └── rate_limiter.py      # Rate limiting logic
│   ├── vectordb/                # Vector database
│   │   ├── chroma_client.py     # ChromaDB integration
│   │   └── vector_operations.py # Similarity operations
│   ├── processing/              # Document processing
│   │   ├── document_processor.py# Main processor
│   │   ├── text_chunker.py      # Chunking strategies
│   │   ├── pdf_processor.py     # PDF extraction
│   │   ├── docx_processor.py    # DOCX extraction
│   │   └── markdown_processor.py# Markdown extraction
│   ├── retrieval/               # RAG pipeline
│   │   ├── rag_pipeline.py      # Main RAG logic
│   │   └── prompt_templates.py  # Prompt templates
│   └── utils/                   # Utilities
│       ├── config_loader.py     # Configuration
│       ├── logger.py            # Logging
│       └── file_utils.py        # File operations
├── examples/                    # Example scripts
│   ├── quick_start.py          # Basic usage example
│   └── advanced_features.py    # Advanced features demo
├── data/                       # Data storage
│   ├── documents/              # Input documents
│   ├── vectordb/               # Vector database
│   └── cache/                  # Response cache
├── app.py                      # Streamlit web interface
├── cli.py                      # Command line interface
├── setup.py                    # Automated setup script
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration file
├── .env                        # Environment variables (create/edit with your API key)
└── venv/                       # Python virtual environment
```

## 🔧 Configuration

### Basic Configuration (config.yaml)

```yaml
gemini:
  api_key: "your-api-key-here"
  model_name: "gemini-2.5-pro"  # or "gemini-1.5-flash"
  rpm_limit: 5                  # Requests per minute
  daily_limit: 100              # Daily request limit

vectordb:
  collection_name: "documents"
  persist_directory: "./data/vectordb"

processing:
  chunk_size: 1000              # Tokens per chunk
  chunk_overlap: 200            # Overlap between chunks
  supported_formats: [".pdf", ".txt", ".md", ".docx"]

retrieval:
  top_k: 5                      # Number of results to retrieve
  similarity_threshold: 0.7      # Minimum similarity score
  rerank_results: true          # Enable re-ranking
```

### Environment Variables

```
GEMINI_API_KEY=your-gemini-api-key
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## 🎯 Usage Examples

### Basic Document Processing

```python
import asyncio
from src.retrieval.rag_pipeline import RAGPipeline

async def main():
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Add documents
    result = await rag.ingest_directory("./data/documents")
    print(f"Processed {result['chunks_processed']} chunks")
    
    # Ask questions
    response = await rag.query("What is machine learning?")
    print(response.answer)

asyncio.run(main())
```

### Advanced Query with Different Prompt Types

```python
# Technical documentation query
response = await rag.query(
    "How do I implement a neural network?", 
    prompt_type='technical'
)

# Comparative analysis
response = await rag.query(
    "Compare supervised vs unsupervised learning", 
    prompt_type='comparative'
)

# Conversational with history
response = await rag.query(
    "Can you explain it in simpler terms?", 
    prompt_type='conversational',
    chat_history="Previous conversation context..."
)
```

### Batch Document Processing

```python
# Process multiple files
file_paths = ["doc1.pdf", "doc2.docx", "doc3.md"]
result = await rag.ingest_documents(file_paths, chunking_method='semantic')

# Process entire directory
result = await rag.ingest_directory(
    directory_path="./research_papers",
    recursive=True,
    chunking_method='adaptive'
)
```

## 🧠 Chunking Strategies

### 1. Semantic Chunking (Recommended)
- Preserves sentence boundaries
- Maintains context coherence
- Optimal for most use cases

### 2. Paragraph Chunking
- Natural document structure
- Good for structured documents
- Preserves logical sections

### 3. Fixed Size Chunking
- Consistent chunk sizes
- Predictable token usage
- Good for technical documents

### 4. Adaptive Chunking
- Automatically adjusts based on content type
- Optimizes for different document types
- Balances context and efficiency

## 🎨 Prompt Templates

### Available Templates

1. **Basic** - Standard Q&A responses
2. **Advanced** - Complex analysis with synthesis
3. **Conversational** - Chat-friendly responses with history
4. **Technical** - Code and technical documentation
5. **Comparative** - Side-by-side comparisons
6. **Summary** - Document summarization
7. **Creative** - Creative writing assistance

### Custom Prompts

```python
custom_prompt = """
Based on the context: {context}

Question: {query}

Provide a detailed technical analysis with:
1. Core concepts
2. Implementation details
3. Best practices
4. Common pitfalls

Answer:
"""

response = await rag.query(
    "How to optimize database queries?",
    custom_prompt=custom_prompt
)
```

## 📈 Performance Optimization

### Rate Limiting Best Practices

```python
# For Gemini Pro (5 RPM)
- Plan for 12-second intervals between requests
- Use caching extensively
- Process documents during off-peak hours

# For Gemini Flash (15 RPM) 
- 4-second intervals between requests
- Better for interactive applications
- Higher daily limits available
```

### Chunking Optimization

```python
# For different content types
technical_docs = {"chunk_size": 1500, "overlap": 300}
general_docs = {"chunk_size": 1000, "overlap": 200}
structured_data = {"chunk_size": 500, "overlap": 50}
```

### Caching Strategy

- **Response Caching**: Identical queries return cached results
- **Embedding Caching**: Reuse embeddings for duplicate content
- **Similarity Caching**: Cache similar query results
- **TTL Management**: Automatic cache expiration

## 🔍 Monitoring & Analytics

### System Statistics

```python
stats = rag.get_system_stats()

# Vector database info
print(f"Documents: {stats['vector_db_stats']['document_count']}")

# API usage
print(f"Daily usage: {stats['gemini_usage']['daily_usage']}")
print(f"Remaining RPM: {stats['gemini_usage']['remaining_rpm']}")

# Processing config
print(f"Chunk size: {stats['processing_stats']['chunk_size']}")
```

### Performance Metrics

- Processing time per query
- Token usage tracking
- Source relevance scores
- Cache hit rates
- API quota utilization

## 🚨 Troubleshooting

### Common Issues

#### API Key Issues
```bash
Error: Gemini API key not found
Solution: Set GEMINI_API_KEY environment variable or update config.yaml
```

#### Rate Limit Exceeded
```bash
Error: Rate limit exceeded
Solution: Wait for quota reset or implement longer delays between requests
```

#### Memory Issues
```bash
Error: Out of memory during processing
Solution: Reduce chunk size or process documents in smaller batches
```

#### Import Errors
```bash
Error: Module not found
Solution: Run `pip install -r requirements.txt` or `python setup.py`
```

### Performance Issues

#### Slow Document Processing
- Reduce chunk size
- Process files individually
- Use simpler chunking strategies

#### Poor Search Results
- Adjust similarity threshold
- Enable result re-ranking
- Use better chunking strategies
- Add more diverse content

#### High Token Usage
- Enable response caching
- Use shorter prompts
- Reduce retrieved context size

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini API** for powerful language capabilities
- **ChromaDB** for efficient vector storage
- **Streamlit** for the web interface
- **Community contributors** for documentation and examples

## 📞 Support

- 📧 **Issues**: Report bugs and feature requests via GitHub Issues
- 📚 **Documentation**: Check the examples/ directory for usage patterns
- 💬 **Discussions**: Join GitHub Discussions for questions and tips

## 🗺️ Roadmap

### Phase 1 - Core Features ✅
- [x] Basic RAG pipeline
- [x] Multi-format document support
- [x] Web and CLI interfaces
- [x] Rate limiting and caching

### Phase 2 - Advanced Features 🚧
- [ ] Multi-user support
- [ ] Advanced analytics dashboard
- [ ] Custom model fine-tuning
- [ ] API endpoint creation

### Phase 3 - Enterprise Features 🔄
- [ ] Authentication and authorization
- [ ] Distributed processing
- [ ] Advanced monitoring
- [ ] Multi-language support

---

⭐ **Star this repository** if you find it useful!

Built with ❤️ for the AI community