"""
Document Processing Example

This example demonstrates how to use the Production RAG System for document processing:
- Building vector databases from documents
- Loading and chunking documents
- Handling duplicate documents
- Updating vector databases
"""

import sys
from pathlib import Path

from production_rag_system.validation.models import QueryRequest
from production_rag_system.config.settings import Settings
from production_rag_system.core.rag_system import create_production_rag_system


def main():
    """Run document processing example"""
    print("🎯 Production RAG System - Document Processing Example")
    print("=" * 50)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    try:
        # 1. Create RAG system with custom settings
        print("1. Creating RAG system...")
        settings = Settings(
            persist_dir=str(script_dir / "chroma_db_example"),
            log_file=str(script_dir / "rag_system.log"),
            chunk_size_threshold=800,
            chunk_overlap_ratio=0.1,
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
        
        # 3. Build vector database from documents
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
        
        # 4. Process a query to test the vector database
        print("\n4. Testing vector database with a query...")
        query = QueryRequest(
            question="What is the main topic of these documents?",
            retriever_type="vec_semantic"
        )
        
        result = rag.process_query(query)
        answer = result.get('result', 'No answer available')
        print(f"✅ Query processed successfully")
        print(f"\nQuestion: {query.question}")
        print(f"Answer: {answer}")
        
        # 5. Show document duplicates info
        print("\n5. Document duplicates information...")
        duplicates_info = rag.get_document_duplicates_info()
        print(f"   - Existing files: {len(duplicates_info['existing_files'])}")
        print(f"   - Duplicate hashes count: {duplicates_info['duplicate_hashes_count']}")
        
        # 6. Demonstrate updating vector database
        print("\n6. Demonstrating vector database update...")
        
        # Create a new document for testing (if it doesn't exist)
        test_doc_path = docs_dir / "new_test_document.txt"
        if not test_doc_path.exists():
            with open(test_doc_path, "w", encoding='utf-8') as f:
                f.write("This is a test document for demonstrating vector database updates.\n")
                f.write("It contains information about testing and validation.\n")
                f.write("This document will be added to the existing vector database.")
        
        print(f"   Created test document: {test_doc_path}")
        
        # Update vector database - use the directory instead of individual file
        # This ensures the document loader can find and process the file correctly
        new_vectordb = rag.update_vector_db([str(docs_dir)], skip_duplicates=True)
        
        if new_vectordb:
            print("✅ Vector database updated successfully")
            
            # Test query after update
            test_query = QueryRequest(
                question="What is the test document about?",
                retriever_type="vec_semantic"
            )
            
            test_result = rag.process_query(test_query)
            test_answer = test_result.get('result', 'No answer available')
            print(f"\nTest Question: {test_query.question}")
            print(f"Test Answer: {test_answer}")
        else:
            print("❌ Failed to update vector database")
        
        # 7. Show performance metrics
        print("\n7. Performance Metrics:")
        metrics = rag.monitor.get_metrics()
        print(f"   - Query Count: {metrics['query_count']}")
        print(f"   - Average Query Time: {metrics['average_query_time']:.2f}s")
        print(f"   - Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
        print(f"   - Error Count: {metrics['error_count']}")
        
        # 8. Show system health after processing
        print("\n8. Final system health:")
        final_health = rag.get_system_health()
        print(f"   - Vector DB: {final_health['vector_db']}")
        print(f"   - Document Count: {final_health['document_count']}")
        print(f"   - Overall: {final_health['overall']}")
        
        print("\n🎉 Document processing example completed successfully!")
        
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
    (script_dir / "test_docs").mkdir(parents=True, exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1)
