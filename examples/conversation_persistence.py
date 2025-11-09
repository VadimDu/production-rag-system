"""
Conversation Persistence Example

This example demonstrates how to use the Production RAG System for conversation persistence:
- Saving conversations to disk
- Loading conversations from disk
- Managing multiple conversations
- Continuing conversations across sessions
"""

import sys
from pathlib import Path
import json

# Add the parent directory to the path so we can import the module
#sys.path.append(str(Path(__file__).parent.parent))
production_rag_system_path = Path.home()/"biotax/analysis"
sys.path.append(str(production_rag_system_path))

from production_rag_system import create_production_rag_system, Settings, QueryRequest


def main():
    """Run conversation persistence example"""
    print("🎯 Production RAG System - Conversation Persistence Example")
    print("=" * 50)
    
    try:
        # 1. Create RAG system
        print("1. Creating RAG system...")
        settings = Settings(
            persist_dir="./examples/chroma_db",
            log_file="./examples/rag_system.log",
            log_level="INFO"
        )
        rag = create_production_rag_system(settings)
        print("✅ RAG system created successfully")
        
        # 2. Check system health
        print("\n2. Checking system health...")
        health = rag.get_system_health()
        if not health["overall"] or not health["vector_db"]:
            print("❌ System not ready for conversational queries")
            return False
        
        print("✅ System ready for conversational queries")
        
        # 3. Create conversations directory
        print("\n3. Creating conversations directory...")
        conversations_dir = Path("./examples/conversations")
        conversations_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created conversations directory: {conversations_dir}")
        
        # 4. Create and save multiple conversations
        print("\n4. Creating and saving conversations...")
        
        # Conversation 1: Python Programming
        print("\n   Conversation 1: Python Programming")
        python_conversation = [
            "What is Python programming language?",
            "What are Python's main features?",
            "Name some popular Python libraries.",
        ]
        
        # Process conversation with buffer memory
        for i, question in enumerate(python_conversation, 1):
            print(f"   Turn {i}: {question}")
            query = QueryRequest(
                question=question,
                retriever_type="vec_semantic"
            )
            result = rag.process_query(query, use_conversation=True, memory_type="buffer")
            print(f"   AI: {result.get('result', 'No answer')}")
        
        # Save conversation
        python_path = conversations_dir / "python_programming.json"
        success = rag.save_conversation_history(
            file_path=str(python_path),
            retriever_type="vec_semantic",
            memory_type="buffer"
        )
        print(f"   ✅ Saved to: {python_path}")
        
        # Clear conversation
        rag.clear_conversation_history(retriever_type="vec_semantic", memory_type="buffer")
        
        # Conversation 2: Machine Learning
        print("\n   Conversation 2: Machine Learning")
        ml_conversation = [
            "What is machine learning?",
            "What is the difference between supervised and unsupervised learning?",
            "Name three common machine learning algorithms.",
        ]
        
        # Process conversation with summary memory
        for i, question in enumerate(ml_conversation, 1):
            print(f"   Turn {i}: {question}")
            query = QueryRequest(
                question=question,
                retriever_type="vec_semantic"
            )
            result = rag.process_query(query, use_conversation=True, memory_type="summary")
            print(f"   AI: {result.get('result', 'No answer')}")
        
        # Save conversation
        ml_path = conversations_dir / "machine_learning.json"
        success = rag.save_conversation_history(
            file_path=str(ml_path),
            retriever_type="vec_semantic",
            memory_type="summary"
        )
        print(f"   ✅ Saved to: {ml_path}")
        
        # Clear conversation
        rag.clear_conversation_history(retriever_type="vec_semantic", memory_type="summary")
        
        # 5. List all saved conversations
        print("\n5. Listing all saved conversations...")
        conversations = rag.list_saved_conversations(str(conversations_dir))
        print(f"   Found {len(conversations)} saved conversations:")
        for conv in conversations:
            print(f"   - {Path(conv['file_path']).name}")
            print(f"     Retriever: {conv['retriever_type']}, Memory: {conv['memory_type']}")
            print(f"     Messages: {conv['message_count']}")
        
        # 6. Load and continue a conversation
        print("\n6. Loading and continuing a conversation...")
        print(f"   Loading from: {python_path}")
        
        # Load conversation
        success = rag.load_conversation_history(
            file_path=str(python_path),
            retriever_type="vec_semantic",
            memory_type="buffer"
        )
        
        if success:
            print("   ✅ Conversation loaded successfully")
            
            # Show conversation history
            history = rag.get_conversation_history(retriever_type="vec_semantic", memory_type="buffer")
            print(f"   Conversation History:\n{history}")
            
            # Continue the conversation
            followup_questions = [
                "Can you summarize what we discussed about Python?",
                "What's one more thing I should know about Python?"
            ]
            
            for i, question in enumerate(followup_questions, 1):
                print(f"\n   Follow-up {i}: {question}")
                query = QueryRequest(
                    question=question,
                    retriever_type="vec_semantic"
                )
                result = rag.process_query(query, use_conversation=True, memory_type="buffer")
                print(f"   AI: {result.get('result', 'No answer')}")
        else:
            print("   ❌ Failed to load conversation")
        
        # 7. Save the continued conversation
        print("\n7. Saving the continued conversation...")
        continued_path = conversations_dir / "python_programming_continued.json"
        success = rag.save_conversation_history(
            file_path=str(continued_path),
            retriever_type="vec_semantic",
            memory_type="buffer"
        )
        print(f"   ✅ Saved to: {continued_path}")
        
        # 8. Demonstrate conversation metadata
        print("\n8. Conversation metadata...")
        for conv in conversations:
            file_path = Path(conv['file_path'])
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                print(f"   File: {file_path.name}")
                print(f"   Retriever Type: {data.get('retriever_type', 'unknown')}")
                print(f"   Memory Type: {data.get('memory_type', 'unknown')}")
                print(f"   Message Count: {len(data.get('messages', []))}")
                print(f"   Timestamp: {data.get('timestamp', 'unknown')}")
                print("   ---")
        
        # 9. Show performance metrics
        print("\n9. Performance Metrics:")
        metrics = rag.monitor.get_metrics()
        print(f"   - Query Count: {metrics['query_count']}")
        print(f"   - Average Query Time: {metrics['average_query_time']:.2f}s")
        print(f"   - Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
        print(f"   - Error Count: {metrics['error_count']}")
        
        print("\n🎉 Conversation persistence example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
