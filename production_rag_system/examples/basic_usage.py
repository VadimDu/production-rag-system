"""
Basic Usage Example

This example demonstrates the basic functionality of the Production RAG System:
- Creating a RAG system
- Checking system health
- Processing a simple query
"""

import sys
from pathlib import Path

from production_rag_system.validation.models import QueryRequest
from production_rag_system.config.settings import Settings
from production_rag_system.core.rag_system import create_production_rag_system


def main():
    """Run basic usage example"""
    print("🎯 Production RAG System - Basic Usage Example")
    print("=" * 50)
    
    try:
        # 1. Create RAG system with default settings
        print("1. Creating RAG system...")
        rag = create_production_rag_system(Settings(
            persist_dir="./chroma_db_example"))
        print("✅ RAG system created successfully")
        
        # 2. Check system health
        print("\n2. Checking system health...")
        health = rag.get_system_health()
        print(f"✅ System health: {health['overall']}")
        print(f"   - Vector DB: {health['vector_db']}")
        print(f"   - Embedding Model: {health['embedding_model']}")
        print(f"   - LLM Connection: {health['llm_connection']}")
        print(f"   - Document Count: {health['document_count']}")
        
        # 3. Process a simple query if vector DB exists
        if health["vector_db"] and health["document_count"] > 0:
            print("\n3. Processing a simple query...")
            query = QueryRequest(
                question="What is artificial intelligence?",
                retriever_type="vec_semantic"
            )
            
            result = rag.process_query(query)
            answer = result.get('result', 'No answer available')
            print(f"✅ Query processed successfully")
            print(f"\nQuestion: {query.question}")
            print(f"Answer: {answer}")
            
            # Show sources if available
            if 'source_documents' in result and result['source_documents']:
                print(f"\nSources ({len(result['source_documents'])}):")
                for i, doc in enumerate(result['source_documents'][:3], 1):  # Show first 3 sources
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"   {i}. {source}")
        else:
            print("\n3. No documents available for querying.")
            print("   Please build a vector database first using the document_processing example.")
        
        # 4. Show performance metrics
        print("\n4. Performance Metrics:")
        metrics = rag.monitor.get_metrics()
        print(f"   - Query Count: {metrics['query_count']}")
        print(f"   - Average Query Time: {metrics['average_query_time']:.2f}s")
        print(f"   - Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
        print(f"   - Error Count: {metrics['error_count']}")
        
        print("\n🎉 Basic usage example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
