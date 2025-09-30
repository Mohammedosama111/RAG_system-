"""
Advanced Features Example - Demonstrates advanced RAG capabilities
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
    print("🧠 Advanced RAG Features Example")
    print("="*50)
    
    # Set API key (replace with your actual key or set as environment variable)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
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
    
    # Add diverse sample content
    print("\n📚 Adding diverse content...")
    
    technical_doc = """
    def implement_rag_system():
        '''
        Implementation of a Retrieval-Augmented Generation system
        '''
        # Initialize components
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        vector_db = ChromaDB()
        llm = GPT4()
        
        # Document processing pipeline
        def process_documents(documents):
            chunks = []
            for doc in documents:
                text = extract_text(doc)
                doc_chunks = chunk_text(text, chunk_size=512)
                chunks.extend(doc_chunks)
            return chunks
        
        # Retrieval and generation
        def query(question):
            query_embedding = embedding_model.encode(question)
            relevant_docs = vector_db.similarity_search(query_embedding, k=5)
            context = combine_documents(relevant_docs)
            response = llm.generate(context, question)
            return response
    """
    
    business_doc = """
    RAG System Business Benefits:
    
    1. Cost Efficiency: Reduces the need for fine-tuning large models by augmenting with external knowledge
    2. Real-time Updates: Knowledge base can be updated without retraining the entire model
    3. Transparency: Sources of information are traceable and verifiable
    4. Accuracy: Reduces hallucinations by grounding responses in factual data
    5. Scalability: Can handle growing knowledge bases efficiently
    6. Compliance: Easier to maintain data governance and regulatory compliance
    
    Use Cases:
    - Customer support systems
    - Internal knowledge management
    - Research and development
    - Legal document analysis
    - Medical information systems
    """
    
    comparison_doc = """
    RAG vs Fine-tuning Comparison:
    
    RAG Advantages:
    - Dynamic knowledge updates
    - Lower computational requirements
    - Better interpretability
    - Reduced hallucinations
    - Cost-effective for knowledge-intensive tasks
    
    Fine-tuning Advantages:
    - Better task-specific performance
    - No external retrieval latency
    - More compact deployment
    - Better for style and format adaptation
    
    RAG vs Traditional Search:
    - RAG provides natural language answers vs keyword matching
    - Better context understanding
    - Conversational capabilities
    - Synthesis of multiple sources
    """
    
    # Add documents with different metadata
    documents = [
        (technical_doc, {"type": "technical", "topic": "implementation", "difficulty": "advanced"}),
        (business_doc, {"type": "business", "topic": "benefits", "difficulty": "beginner"}),
        (comparison_doc, {"type": "analysis", "topic": "comparison", "difficulty": "intermediate"})
    ]
    
    for i, (doc, metadata) in enumerate(documents):
        success = await rag.add_text(doc, metadata=metadata)
        if success:
            print(f"✅ Added {metadata['type']} document")
        else:
            print(f"❌ Failed to add document {i+1}")
    
    # Demonstrate different prompt types
    print("\n🎯 Testing Different Prompt Types...")
    
    test_cases = [
        {
            "question": "What are the business benefits of RAG systems?",
            "prompt_type": "basic",
            "description": "Basic Q&A"
        },
        {
            "question": "Analyze the advantages and disadvantages of RAG vs fine-tuning approaches",
            "prompt_type": "comparative",
            "description": "Comparative analysis"
        },
        {
            "question": "How would you implement a RAG system in Python?",
            "prompt_type": "technical",
            "description": "Technical implementation"
        },
        {
            "question": "Provide a comprehensive overview of RAG technology",
            "prompt_type": "advanced",
            "description": "Advanced synthesis"
        },
        {
            "question": "Summarize the key points about RAG systems for a business presentation",
            "prompt_type": "summary",
            "description": "Summary generation"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📝 {test_case['description']}: {test_case['question']}")
        print("-" * 60)
        
        result = await rag.query(
            test_case['question'], 
            prompt_type=test_case['prompt_type']
        )
        
        print(f"🤖 Answer: {result.answer[:200]}...")
        print(f"📊 Sources: {len(result.sources)}, Time: {result.processing_time:.2f}s, Tokens: {result.tokens_used}")
        
        results.append(result)
        
        # Show source details for one example
        if test_case['prompt_type'] == 'comparative':
            print("\n📚 Source Details:")
            for i, (source, metadata) in enumerate(zip(result.sources, result.metadata)):
                print(f"  Source {i+1} [{metadata.get('type', 'unknown')}]: {source[:100]}...")
    
    # Demonstrate conversational RAG
    print("\n💬 Testing Conversational RAG...")
    
    conversation_questions = [
        "What is RAG?",
        "What are its main advantages?",
        "How does it compare to fine-tuning?",
        "Can you provide a technical implementation example?"
    ]
    
    chat_history = ""
    
    for question in conversation_questions:
        print(f"\n🔍 Q: {question}")
        
        result = await rag.query(
            question, 
            prompt_type='conversational',
            chat_history=chat_history
        )
        
        print(f"🤖 A: {result.answer}")
        
        # Update chat history
        chat_history += f"Human: {question}\nAssistant: {result.answer}\n"
    
    # Performance analysis
    print("\n📈 Performance Analysis:")
    print("-" * 40)
    
    avg_processing_time = sum(r.processing_time for r in results) / len(results)
    total_tokens = sum(r.tokens_used for r in results)
    
    print(f"Average processing time: {avg_processing_time:.2f}s")
    print(f"Total tokens used: {total_tokens}")
    print(f"Average sources per query: {sum(len(r.sources) for r in results) / len(results):.1f}")
    
    # System stats
    print("\n📊 Final System Statistics:")
    stats = rag.get_system_stats()
    
    db_stats = stats.get('vector_db_stats', {})
    usage = stats.get('gemini_usage', {})
    
    print(f"Documents: {db_stats.get('document_count', 0)}")
    print(f"API Usage: {usage.get('daily_usage', 0)}/{usage.get('daily_limit', 100)}")
    print(f"Remaining RPM: {usage.get('remaining_rpm', 0)}")
    
    print("\n✅ Advanced features demonstration completed!")
    print("💡 Key takeaways:")
    print("  - Different prompt types optimize for different use cases")
    print("  - Metadata enables better document organization and retrieval")
    print("  - Conversational context improves multi-turn interactions")
    print("  - Performance monitoring helps optimize system usage")

if __name__ == "__main__":
    asyncio.run(main())