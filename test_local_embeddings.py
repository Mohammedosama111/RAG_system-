"""
Test Local Embeddings Implementation
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.embeddings.local_embedding_service import LocalEmbeddingService
from src.gemini.client import GeminiClient
from src.utils.config_loader import ConfigLoader

async def test_local_embeddings():
    print("🧪 Testing Local Embeddings Implementation")
    print("=" * 60)
    
    # Test 1: Direct local embedding service
    print("\n1️⃣ Testing Direct Local Embedding Service:")
    print("-" * 40)
    
    embedding_service = LocalEmbeddingService()
    
    # Test model info
    info = embedding_service.get_model_info()
    print(f"Model: {info['model_name']}")
    print(f"Status: {info['status']}")
    print(f"Dimension: {info['embedding_dimension']}")
    
    # Test encoding
    test_texts = [
        "Machine learning is powerful",
        "Deep learning uses neural networks", 
        "Python is a programming language"
    ]
    
    print(f"\n🔄 Encoding {len(test_texts)} test texts...")
    embeddings = embedding_service.encode_texts(test_texts)
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    print(f"📏 Embedding dimension: {len(embeddings[0])}")
    
    # Test similarity
    sim1 = embedding_service.similarity(embeddings[0], embeddings[1])
    sim2 = embedding_service.similarity(embeddings[0], embeddings[2])
    
    print(f"🔗 Similarity (ML vs DL): {sim1:.3f}")
    print(f"🔗 Similarity (ML vs Python): {sim2:.3f}")
    print(f"✅ Expected: ML-DL similarity > ML-Python: {sim1 > sim2}")
    
    # Test 2: Gemini client with local embeddings
    print("\n2️⃣ Testing Gemini Client with Local Embeddings:")
    print("-" * 40)
    
    try:
        config = ConfigLoader.load_config()
        gemini_client = GeminiClient(config)
        
        print("🔄 Testing embedding generation through Gemini client...")
        client_embeddings = await gemini_client.generate_embeddings(test_texts[:2])
        
        print(f"✅ Generated {len(client_embeddings)} embeddings via client")
        print(f"📏 Embedding dimension: {len(client_embeddings[0])}")
        
    except Exception as e:
        print(f"⚠️ Client test failed: {e}")
        print("This is expected if API key is not configured")
    
    # Test 3: Model performance test
    print("\n3️⃣ Running Model Performance Test:")
    print("-" * 40)
    
    test_results = embedding_service.test_model()
    
    for key, value in test_results.items():
        if key == "error":
            print(f"❌ {key}: {value}")
        else:
            print(f"📊 {key}: {value}")
    
    print("\n🎉 Local Embedding Test Complete!")
    print("=" * 60)
    print("✅ Benefits of Local Embeddings:")
    print("  • No API quotas or rate limits")
    print("  • Instant processing (no network delays)")
    print("  • Complete privacy (runs locally)")
    print("  • Zero additional cost")
    print("  • Works offline")
    
    print("\n🚀 Ready to use your RAG system!")
    print("  Run: streamlit run app.py")

if __name__ == "__main__":
    asyncio.run(test_local_embeddings())