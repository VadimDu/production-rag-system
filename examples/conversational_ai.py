"""
Conversational AI Example

This example demonstrates how to use the Production RAG System for conversational AI:
- Processing conversational queries with memory
- Using different memory types (buffer and summary)
- Getting conversation history
- Saving and loading conversations
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import the module
#sys.path.append(str(Path(__file__).parent.parent))
production_rag_system_path = Path.home()/"biotax/analysis"
sys.path.append(str(production_rag_system_path))

from production_rag_system import create_production_rag_system, QueryRequest


def main():
    """Run conversational AI example"""
    print("🎯 Production RAG System - Conversational AI Example")
    print("=" * 50)
    
    try:
        # 1. Create RAG system
        print("1. Creating RAG system...")
        rag = create_production_rag_system()
        print("✅ RAG system created successfully")
        
        # 2. Check system health
        print("\n2. Checking system health...")
        health = rag.get_system_health()
        if not health["overall"] or not health["vector_db"]:
            print("❌ System not ready for conversational queries")
            return False
        
        print("✅ System ready for conversational queries")
        
        # 3. Process a conversation with buffer memory
        print("\n3. Processing conversation with buffer memory...")
        conversation = [
            "What is artificial intelligence?",
            "What are the main types of machine learning?",
            "How does deep learning differ from traditional machine learning?"
        ]
        
        for i, question in enumerate(conversation, 1):
            print(f"\nTurn {i}: {question}")
            print("-" * 40)
            
            query = QueryRequest(
                question=question,
                retriever_type="vec_semantic"
            )
            
            result = rag.process_query(query, use_conversation=True, memory_type="buffer")
            answer = result.get('result', 'No answer available')
            print(f"AI: {answer}")
        
        # 4. Show conversation history
        print("\n4. Getting conversation history...")
        history = rag.get_conversation_history(retriever_type="vec_semantic", memory_type="buffer")
        print(f"Conversation History:\n{history}")
        
        # 5. Save conversation history
        print("\n5. Saving conversation history...")
        save_path = "./examples/conversations/ai_discussion_buffer.json"
        success = rag.save_conversation_history(
            file_path=save_path,
            retriever_type="vec_semantic",
            memory_type="buffer"
        )
        if success:
            print(f"✅ Conversation history saved to {save_path}")
        else:
            print("❌ Failed to save conversation history")
        
        # 6. Clear conversation and start a new one with summary memory
        print("\n6. Starting new conversation with summary memory...")
        rag.clear_conversation_history(retriever_type="vec_semantic", memory_type="buffer")
        
        new_conversation = [
            "What is natural language processing?",
            "Name three popular NLP libraries",
        ]
        
        for i, question in enumerate(new_conversation, 1):
            print(f"\nTurn {i}: {question}")
            print("-" * 40)
            
            query = QueryRequest(
                question=question,
                retriever_type="vec_semantic"
            )
            
            result = rag.process_query(query, use_conversation=True, memory_type="summary")
            answer = result.get('result', 'No answer available')
            print(f"AI: {answer}")
        
        # 7. Show summary conversation history
        print("\n7. Getting summary conversation history...")
        history = rag.get_conversation_history(retriever_type="vec_semantic", memory_type="summary")
        print(f"Conversation History:\n{history}")
        
        # 8. Save summary conversation
        print("\n8. Saving summary conversation history...")
        save_path = "./examples/conversations/ai_discussion_summary.json"
        success = rag.save_conversation_history(
            file_path=save_path,
            retriever_type="vec_semantic",
            memory_type="summary"
        )
        if success:
            print(f"✅ Summary conversation saved to {save_path}")
        else:
            print("❌ Failed to save summary conversation")
        
        # 9. Demonstrate loading a conversation
        print("\n9. Loading conversation from disk...")
        load_path = "./examples/conversations/ai_discussion_buffer.json"
        success = rag.load_conversation_history(
            file_path=load_path,
            retriever_type="vec_semantic",
            memory_type="buffer"
        )
        
        if success:
            print(f"✅ Conversation loaded from {load_path}")
            
            # Continue the loaded conversation
            followup_query = QueryRequest(
                question="Can you summarize what we discussed about AI?",
                retriever_type="vec_semantic"
            )
            
            result = rag.process_query(followup_query, use_conversation=True, memory_type="buffer")
            answer = result.get('result', 'No answer available')
            print(f"\nFollow-up: {followup_query.question}")
            print(f"AI: {answer}")
        else:
            print("❌ Failed to load conversation from disk")
        
        # 10. Show performance metrics
        print("\n10. Performance Metrics:")
        metrics = rag.monitor.get_metrics()
        print(f"   - Query Count: {metrics['query_count']}")
        print(f"   - Average Query Time: {metrics['average_query_time']:.2f}s")
        print(f"   - Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
        print(f"   - Error Count: {metrics['error_count']}")
        
        print("\n🎉 Conversational AI example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Create conversations directory if it doesn't exist
    Path("./examples/conversations").mkdir(parents=True, exist_ok=True)
    
    success = main()
    sys.exit(0 if success else 1)
