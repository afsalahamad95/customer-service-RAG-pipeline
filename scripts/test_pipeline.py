"""Quick test script to verify the RAG pipeline."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import PreprocessingPipeline
from src.retrieval import EmbeddingService
from src.utils import get_logger

logger = get_logger(__name__)


async def test_preprocessing():
    """Test preprocessing pipeline."""
    print("\n=== Testing Preprocessing ===")
    
    pipeline = PreprocessingPipeline()
    
    test_text = "Hi, I need help resetting my password! My email is john@example.com"
    
    result = pipeline.process(test_text, detect_pii=True)
    
    print(f"Original: {result.original_text}")
    print(f"Cleaned: {result.cleaned_text}")
    print(f"Anonymized: {result.anonymized_text}")
    print(f"Language: {result.language}")
    print(f"PII detected: {len(result.detected_pii)} entities")
    
    return True


async def test_embeddings():
    """Test embedding generation."""
    print("\n=== Testing Embeddings ===")
    
    service = EmbeddingService()
    
    text = "How do I reset my password?"
    embedding = service.embed(text)
    
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {service.get_embedding_dimension()}")
    
    # Test similarity
    text2 = "I forgot my password"
    emb2 = service.embed(text2)
    similarity = service.similarity(embedding.flatten(), emb2.flatten())
    print(f"Similarity with '{text2}': {similarity:.3f}")
    
    return True


async def run_tests():
    """Run all tests."""
    print("🚀 Running RAG Pipeline Tests\n")
    
    tests = [
        ("Preprocessing", test_preprocessing),
        ("Embeddings", test_embeddings),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
            print(f"✅ {name} passed")
        except Exception as e:
            results.append((name, False))
            print(f"❌ {name} failed: {e}")
            logger.error(f"Test {name} failed", exc_info=True)
    
    print("\n" + "="*50)
    print("Test Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
