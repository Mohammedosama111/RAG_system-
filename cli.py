"""
Command Line Interface for the RAG System
Simple CLI for testing and batch operations
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_group = ingest_parser.add_mutually_exclusive_group(required=True)
    ingest_group.add_argument('--file', help='Single file to ingest')
    ingest_group.add_argument('--directory', help='Directory to ingest')
    ingest_parser.add_argument('--chunking', choices=['semantic', 'paragraph', 'fixed', 'adaptive'], 
                              default='semantic', help='Chunking method')
    ingest_parser.add_argument('--recursive', action='store_true', help='Process directory recursively')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--prompt-type', choices=['basic', 'advanced', 'technical', 'summary'], 
                             default='basic', help='Prompt type')
    query_parser.add_argument('--sources', action='store_true', help='Show source documents')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Load configuration
    config = ConfigLoader.load_config()
    
    # Check API key
    if not config.get('gemini', {}).get('api_key'):
        print("❌ Error: Gemini API key not found in configuration or environment variables")
        print("Please set GEMINI_API_KEY environment variable or update config.yaml")
        return
    
    # Initialize RAG pipeline
    try:
        print("🚀 Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline(config)
        print("✅ RAG pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        return
    
    # Execute command
    if args.command == 'ingest':
        await handle_ingest(rag_pipeline, args)
    elif args.command == 'query':
        await handle_query(rag_pipeline, args)
    elif args.command == 'stats':
        handle_stats(rag_pipeline)
    elif args.command == 'interactive':
        await handle_interactive(rag_pipeline)

async def handle_ingest(rag_pipeline, args):
    """Handle document ingestion"""
    print(f"📄 Starting document ingestion...")
    
    try:
        if args.file:
            print(f"Processing file: {args.file}")
            result = await rag_pipeline.ingest_documents([args.file], args.chunking)
        else:
            print(f"Processing directory: {args.directory}")
            result = await rag_pipeline.ingest_directory(args.directory, args.recursive, args.chunking)
        
        if result['status'] == 'success':
            print(f"✅ Successfully processed {result['chunks_processed']} chunks")
            if 'successful_files' in result:
                print(f"📁 Processed {len(result['successful_files'])} files successfully")
            if result.get('failed_files'):
                print(f"⚠️  Failed to process {len(result['failed_files'])} files:")
                for failed in result['failed_files']:
                    print(f"   - {failed['file']}: {failed['error']}")
        else:
            print(f"❌ Ingestion failed: {result['message']}")
            
    except Exception as e:
        print(f"❌ Ingestion error: {e}")

async def handle_query(rag_pipeline, args):
    """Handle query processing"""
    print(f"🔍 Processing query: {args.question}")
    
    try:
        result = await rag_pipeline.query(args.question, args.prompt_type)
        
        print("\n" + "="*80)
        print("🤖 ANSWER:")
        print("="*80)
        print(result.answer)
        
        if args.sources and result.sources:
            print("\n" + "="*80)
            print("📚 SOURCES:")
            print("="*80)
            
            for i, (source, metadata, similarity) in enumerate(zip(result.sources, result.metadata, result.similarities)):
                print(f"\n--- Source {i+1} [{metadata.get('file_name', 'Unknown')}] (Similarity: {similarity:.3f}) ---")
                print(source[:300] + "..." if len(source) > 300 else source)
        
        print("\n" + "="*80)
        print("📊 METRICS:")
        print("="*80)
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"Sources Found: {len(result.sources)}")
        print(f"Tokens Used: {result.tokens_used}")
        
    except Exception as e:
        print(f"❌ Query error: {e}")

def handle_stats(rag_pipeline):
    """Handle stats display"""
    print("📊 System Statistics")
    print("="*80)
    
    try:
        stats = rag_pipeline.get_system_stats()
        
        # Vector DB stats
        db_stats = stats.get('vector_db_stats', {})
        print(f"Vector Database:")
        print(f"  Documents: {db_stats.get('document_count', 0)}")
        print(f"  Collection: {db_stats.get('collection_name', 'N/A')}")
        
        # Gemini usage
        usage = stats.get('gemini_usage', {})
        print(f"\nGemini API Usage:")
        print(f"  Daily Usage: {usage.get('daily_usage', 0)}/{usage.get('daily_limit', 100)}")
        print(f"  RPM Usage: {usage.get('rpm_usage', 0)}/{usage.get('rpm_limit', 5)}")
        print(f"  Remaining Daily: {usage.get('remaining_daily', 0)}")
        print(f"  Remaining RPM: {usage.get('remaining_rpm', 0)}")
        
        # Processing config
        processing = stats.get('processing_stats', {})
        print(f"\nProcessing Configuration:")
        print(f"  Chunk Size: {processing.get('chunk_size', 1000)}")
        print(f"  Chunk Overlap: {processing.get('chunk_overlap', 200)}")
        print(f"  Supported Formats: {processing.get('supported_formats', [])}")
        
        # Retrieval config
        retrieval = stats.get('retrieval_config', {})
        print(f"\nRetrieval Configuration:")
        print(f"  Top K: {retrieval.get('top_k', 5)}")
        print(f"  Similarity Threshold: {retrieval.get('similarity_threshold', 0.7)}")
        print(f"  Re-rank Results: {retrieval.get('rerank_results', True)}")
        
    except Exception as e:
        print(f"❌ Stats error: {e}")

async def handle_interactive(rag_pipeline):
    """Handle interactive mode"""
    print("🚀 Starting interactive mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("="*80)
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print_interactive_help()
                continue
            
            if user_input.lower() == 'stats':
                handle_stats(rag_pipeline)
                continue
            
            if not user_input:
                continue
            
            # Process as query
            print("🤖 Assistant: ", end="", flush=True)
            
            result = await rag_pipeline.query(user_input, 'conversational')
            print(result.answer)
            
            # Show brief metrics
            print(f"\n⏱️  {result.processing_time:.2f}s | 📚 {len(result.sources)} sources | 🔤 {result.tokens_used} tokens")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def print_interactive_help():
    """Print interactive mode help"""
    print("""
📖 Interactive Mode Commands:
- Ask any question about your documents
- 'stats' - Show system statistics
- 'help' - Show this help message  
- 'quit' or 'exit' - Exit interactive mode
- Ctrl+C - Exit interactive mode
    """)

if __name__ == "__main__":
    asyncio.run(main())