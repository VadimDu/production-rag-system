"""
Advanced Retrieval Example

This example demonstrates advanced retrieval features of the Production RAG System:
- Using different retriever types (vector semantic, BM25, ensemble)
- Comparing retrieval results
- Customizing retrieval parameters
- Using different k values
"""

import sys
from pathlib import Path

from production_rag_system.validation.models import QueryRequest
from production_rag_system.config.settings import Settings
from production_rag_system.core.rag_system import create_production_rag_system


def main():
    """Run advanced retrieval example"""
    print("🎯 Production RAG System - Advanced Retrieval Example")
    print("=" * 50)

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    try:
        # 1. Create RAG system
        print("1. Creating RAG system...")
        settings = Settings(
            persist_dir=str(script_dir / "chroma_db_example"),
            log_file=str(script_dir / "rag_system.log"),
            chunk_size_threshold=500,
            chunk_overlap_ratio=0.2,
            log_level="INFO"
        )
        rag = create_production_rag_system(settings)
        print("✅ RAG system created successfully")
        
         # 2. Check system health
        print("\n2. Checking system health...")
        health = rag.get_system_health()
        print(f"✅ System health: {health['overall']}")
        print(f"   - Vector DB: {health['vector_db']}")
        print(f"   - Embedding Model: {health['embedding_model']}")
        print(f"   - LLM Connection: {health['llm_connection']}")
        
        #Build vector database from documents
        print("\n3. Building vector database from documents...")
        
        # Define document paths (relative to script location)
        document_paths = [
            str(script_dir / "test_docs")  # Directory containing documents
        ]
        
        # Check if documents directory exists
        docs_dir = script_dir / "test_docs"
        if not docs_dir.exists():
            print(f"❌ Documents directory not found: {docs_dir}")
            print("   Please create a 'test_docs' directory in the examples folder and add your documents")
            print("   Supported formats: .txt, .md, .pdf, .docx")
            return False
        
        # List available documents
        available_docs = []
        for ext in [".txt", ".md", ".pdf", ".docx"]:
            available_docs.extend(docs_dir.glob(f"**/*{ext}"))
        
        if not available_docs:
            print("❌ No documents found in the test_docs directory")
            print(f"   Directory contents: {list(docs_dir.iterdir()) if docs_dir.exists() else 'Directory does not exist'}")
            print("   Please add some documents (txt, md, pdf, or docx) to the examples/test_docs folder")
            return False

        # Build vector database
        vectordb = rag.build_vector_db_from_docs(document_paths, skip_duplicates=True)
        
        if vectordb:
            print("✅ Vector database built successfully")
            
            # Check document count
            docs_data = rag.list_existing_documents()
            print(f"   - Document chunks: {len(docs_data['ids'])}")
        else:
            print("❌ Failed to build vector database")
            return False

        # 2. Check system health
        print("\n2. Checking system health again...")
        health = rag.get_system_health()
        if not health["overall"] or not health["vector_db"]:
            print("❌ System not ready for advanced retrieval")
            return False
        
        print("✅ System ready for advanced retrieval")
        print(f"   - Document Count: {health['document_count']}")
        
        # 3. Create different retrievers
        print("\n3. Creating different retrievers...")
        retrievers = rag.create_retrievers()
        print(f"   Available retrievers: {list(retrievers.keys())}")
        
        # 4. Compare retrieval results
        print("\n4. Comparing retrieval results...")
        test_query = QueryRequest(
            question="What is machine learning?",
            retriever_type="vec_semantic"
        )
        
        results = {}
        for retriever_name, retriever in retrievers.items():
            # Get documents from retriever
            docs = retriever.get_relevant_documents(test_query.question, k=3)
            results[retriever_name] = docs
            print(f"\n   {retriever_name.upper()} Results:")
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"     {i}. Source: {source}")
                print(f"        Preview: {content_preview}")
        
        # 5. Test different k values
        print("\n5. Testing different k values...")
        k_values = [3, 5, 10]
        k_results = {}
        
        for k in k_values:
            print(f"\n   Testing with k={k}:")
            
            # Create retriever with custom k
            custom_retrievers = rag.create_retrievers(k=k)
            retriever = custom_retrievers.get("vec_semantic")
            
            if retriever:
                docs = retriever.get_relevant_documents(test_query.question, k=k)
                k_results[k] = len(docs)
                print(f"     Retrieved {len(docs)} documents")
        
        # 6. Test different retriever types
        print("\n6. Testing different retriever types...")
        retriever_types = ["vec_semantic", "bm25", "ensemble"]
        
        for retriever_type in retriever_types:
            print(f"\n   Testing {retriever_type} retriever:")
            
            # Build QA chain with specific retriever
            chain = rag.build_qa_chain(
                retriever_type=retriever_type,
                use_conversation=False,
                k=3
            )
            
            # Process query
            result = chain.invoke({"query": test_query.question})
            answer = result.get('result', 'No answer available')
            print(f"     Answer: {answer}")
            
            # Show sources
            if 'source_documents' in result:
                sources = result['source_documents']
                print(f"     Sources: {len(sources)}")
                for i, doc in enumerate(sources, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"       {i}. {source}")
        
        # 7. Test different retriever types with process_query
        print("\n7. Testing retriever types with process_query...")
        
        for retriever_type in retriever_types:
            print(f"\n   Testing {retriever_type} with process_query:")
            
            query = QueryRequest(
                question=test_query.question,
                retriever_type=retriever_type
            )
            
            result = rag.process_query(query)
            answer = result.get('result', 'No answer available')
            print(f"     Answer: {answer}")
        
        # 8. Test ensemble retriever with different weights
        print("\n8. Testing ensemble retriever with different weights...")
        
        if "ensemble" in retrievers:
            # Create custom ensemble retriever
            from langchain.retrievers import EnsembleRetriever
            from langchain_community.retrievers import BM25Retriever
            from langchain.docstore.document import Document
            
            # Get documents from vector store
            docs_data = rag.vectordb.get()
            if docs_data['documents']:
                documents = [
                    Document(page_content=doc, metadata=meta)
                    for doc, meta in zip(docs_data['documents'], docs_data['metadatas'])
                ]
                
                # Create BM25 retriever
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = 3
                
                # Get vector retriever
                vector_retriever = rag.vectordb.as_retriever(search_kwargs={"k": 3})
                
                # Test different weight combinations
                weight_combinations = [
                    (0.3, 0.7),  # More weight to vector
                    (0.5, 0.5),  # Equal weights
                    (0.7, 0.3),  # More weight to BM25
                ]
                
                for bm25_weight, vector_weight in weight_combinations:
                    print(f"\n   Testing ensemble with BM25 weight={bm25_weight}, Vector weight={vector_weight}:")
                    
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, vector_retriever],
                        weights=[bm25_weight, vector_weight]
                    )
                    
                    # Create QA chain with ensemble retriever
                    chain = rag.retriever_factory.build_qa_chain(
                        retriever_type="vec_semantic",
                        use_conversation=False,
                        k=3
                    )
                    
                    # Replace the retriever in the chain
                    chain.retriever = ensemble_retriever
                    
                    # Process query
                    result = chain.invoke({"query": test_query.question})
                    answer = result.get('result', 'No answer available')
                    print(f"     Answer: {answer}")
        
        # 9. Performance comparison
        print("\n9. Performance comparison...")
        
        from time import time
        
        # Test each retriever type for performance
        performance_results = {}
        
        for retriever_type in retriever_types:
            print(f"   Testing {retriever_type} performance...")
            
            query = QueryRequest(
                question=test_query.question,
                retriever_type=retriever_type
            )
            
            # Measure time
            start_time = time.time()
            result = rag.process_query(query)
            end_time = time.time()
            
            execution_time = end_time - start_time
            performance_results[retriever_type] = execution_time
            
            print(f"     Execution time: {execution_time:.3f}s")
        
        # Show performance comparison
        print("\n   Performance Comparison:")
        for retriever_type, time_taken in sorted(performance_results.items(), key=lambda x: x[1]):
            print(f"     {retriever_type}: {time_taken:.3f}s")
        
        # 10. Show performance metrics
        print("\n10. System Performance Metrics:")
        metrics = rag.monitor.get_metrics()
        print(f"   - Query Count: {metrics['query_count']}")
        print(f"   - Average Query Time: {metrics['average_query_time']:.2f}s")
        print(f"   - Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
        print(f"   - Error Count: {metrics['error_count']}")
        
        print("\n🎉 Advanced retrieval example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":

    # Get the script directory for creating relative paths
    script_dir = Path(__file__).parent.absolute()
    
    # Create directories relative to script location
    (script_dir / "chroma_db_example").mkdir(parents=True, exist_ok=True)
    (script_dir/"conversations").mkdir(parents=True, exist_ok=True)

    success = main()
    sys.exit(0 if success else 1)
