"""
Quick Start Example - Basic RAG System Usage
"""

import asyncio
import os
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.config_loader import ConfigLoader

async def main():
    print("🚀 RAG System Quick Start Example")
    print("="*50)
    
    # Set API key (replace with your actual key)
    api_key = input("Enter your Gemini API key: ").strip()
    if not api_key:
        print("❌ API key required")
        return
    
    os.environ['GEMINI_API_KEY'] = api_key
    
    # Initialize RAG pipeline
    print("\n📋 Initializing RAG pipeline...")
    config = ConfigLoader.load_config()
    rag = RAGPipeline(config)
    print("✅ RAG pipeline initialized!")
    
    # Add some sample text
    print("\n📝 Adding sample documents...")
    
    sample_docs = [
        """
        Machine Learning is a subset of artificial intelligence (AI) that focuses on the development of computer systems 
        that can learn and adapt without following explicit instructions. It uses mathematical models and algorithms 
        to perform tasks based on patterns in data. Common applications include image recognition, natural language 
        processing, recommendation systems, and predictive analytics.
        """,
        """
        Python is a high-level, interpreted programming language known for its simplicity and readability. 
        It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms, 
        including procedural, object-oriented, and functional programming. It's widely used in web development, 
        data science, artificial intelligence, automation, and scientific computing.
        """,
        """
        The Retrieval-Augmented Generation (RAG) approach combines the power of large language models with 
        external knowledge retrieval. This technique allows AI systems to access and utilize information from 
        external databases or document collections to generate more accurate and contextually relevant responses. 
        RAG is particularly useful for question-answering systems and knowledge-intensive tasks.
        """
    ]
    
    for i, doc in enumerate(sample_docs):
        success = await rag.add_text(doc, metadata={"source": f"sample_doc_{i+1}"})
        if success:
            print(f"✅ Added sample document {i+1}")
        else:
            print(f"❌ Failed to add sample document {i+1}")
    
    # Test queries
    print("\n💬 Testing queries...")
    
    test_questions = [
        "What is machine learning?",
        "Tell me about Python programming language",
        "What is RAG and how does it work?",
        "Compare Python and machine learning"
    ]
    
    for question in test_questions:
        print(f"\n🔍 Question: {question}")
        print("-" * 40)
        
        result = await rag.query(question, prompt_type='basic')
        
        print(f"🤖 Answer: {result.answer}")
        print(f"⏱️  Processing time: {result.processing_time:.2f}s")
        print(f"📚 Sources used: {len(result.sources)}")
    
    # Show system stats
    print("\n📊 System Statistics:")
    print("-" * 40)
    stats = rag.get_system_stats()
    
    db_stats = stats.get('vector_db_stats', {})
    usage = stats.get('gemini_usage', {})
    
    print(f"Documents in database: {db_stats.get('document_count', 0)}")
    print(f"Daily API usage: {usage.get('daily_usage', 0)}/{usage.get('daily_limit', 100)}")
    print(f"Remaining RPM: {usage.get('remaining_rpm', 0)}")
    
    print("\n✅ Quick start example completed!")
    print("💡 Try running the web interface with: streamlit run app.py")
    print("💡 Or use the CLI with: python cli.py --help")

if __name__ == "__main__":
    asyncio.run(main())