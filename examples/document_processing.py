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

# Add the parent directory to the path so we can import the module
#sys.path.append(str(Path(__file__).parent.parent))
production_rag_system_path = Path.home()/"biotax/analysis"
sys.path.append(str(production_rag_system_path))

from production_rag_system import create_production_rag_system, Settings


def main():
    """Run document processing example"""
    print("🎯 Production RAG System - Document Processing Example")
    print("=" * 50)
    
    try:
        # 1. Create RAG system with custom settings
        print("1. Creating RAG system...")
        settings = Settings(
            persist_dir="./examples/chroma_db",
            log_file="./examples/rag_system.log",
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
        
        # Define document paths (replace with your actual document paths)
        document_paths = [
            "./examples/documents"  # Directory containing documents
        ]
        
        # Check if documents directory exists
        docs_dir = Path("./examples/documents")
        if not docs_dir.exists():
            print(f"❌ Documents directory not found: {docs_dir}")
            print("   Please create a 'documents' directory in the examples folder and add your documents")
            print("   Supported formats: .txt, .md, .pdf, .docx")
            return False
        
        # List available documents
        available_docs = []
        for ext in [".txt", ".md", ".pdf", ".docx"]:
            available_docs.extend(docs_dir.glob(f"**/*{ext}"))
        
        if not available_docs:
            print("❌ No documents found in the documents directory")
            print("   Please add some documents (txt, md, pdf, or docx) to the examples/documents folder")
            return False
        
        print(f"   Found {len(available_docs)} documents:")
        for doc in available_docs[:5]:  # Show first 5 documents
            print(f"   - {doc.name}")
        if len(available_docs) > 5:
            print(f"   ... and {len(available_docs) - 5} more")
        
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
        
        # Create a new document for testing
        test_doc_path = docs_dir / "test_document.txt"
        with open(test_doc_path, "w") as f:
            f.write("This is a test document for demonstrating vector database updates.\n")
            f.write("It contains information about testing and validation.\n")
            f.write("This document will be added to the existing vector database.")
        
        print(f"   Created test document: {test_doc_path}")
        
        # Update vector database with the new document
        new_vectordb = rag.update_vector_db([str(test_doc_path)], skip_duplicates=True)
        
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
    # Create necessary directories
    Path("./examples/chroma_db").mkdir(parents=True, exist_ok=True)
    Path("./examples/documents").mkdir(parents=True, exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1)
