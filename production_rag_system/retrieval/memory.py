"""
Memory management for the RAG system.

This module provides conversation memory management utilities.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import Optional, Any, List, Dict

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manage conversation memory for the RAG system"""
    
    def __init__(self, llm):
        """Initialize memory manager
        
        Args:
            llm: Language model instance for summary memory
        """
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        self.current_memory = None
        self.current_memory_type = None
    
    def set_memory_type(self, memory_type: str = "buffer"):
        """Set conversation memory type
        
        Args:
            memory_type: Type of memory to use ("buffer" or "summary")
            
        Returns:
            Memory instance
            
        Raises:
            ValueError: If memory type is invalid
        """
        if memory_type == "buffer":
            self.current_memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            )
        elif memory_type == "summary":
            self.current_memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            )
        else:
            raise ValueError("Memory type must be 'buffer' or 'summary'")
        
        self.current_memory_type = memory_type
        self.logger.info(f"Memory type set to {memory_type}")
        return self.current_memory
    
    def get_conversation_history(self, chain) -> str:
        """Get formatted conversation history
        
        Args:
            chain: Chain instance with memory
            
        Returns:
            Formatted conversation history
        """
        if hasattr(chain, 'memory') and hasattr(chain.memory, 'chat_memory'):
            messages = chain.memory.chat_memory.messages
            history_str = ""
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history_str += f"======Human Message:======\n{msg.content}\n\n"
                elif isinstance(msg, AIMessage):
                    history_str += f"======AI Message:======\n{msg.content}\n\n"
            return history_str
        else:
            return "Conversation history not available"
    
    def clear_memory(self):
        """Clear the current conversation memory"""
        if self.current_memory:
            if hasattr(self.current_memory, 'clear'):
                self.current_memory.clear()
            elif hasattr(self.current_memory, 'chat_memory'):
                self.current_memory.chat_memory.clear()
            self.logger.info("Conversation memory cleared")
    
    
    def save_conversation_history(self, file_path: str, retriever_type: str = "vec_semantic") -> bool:
        """Save conversation history to disk
        
        Args:
            file_path: Path to save the conversation history
            retriever_type: Type of retriever used for this conversation
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_memory:
            self.logger.warning("No conversation memory to save")
            return False
        
        try:
            # Extract messages from memory
            messages = []
            if hasattr(self.current_memory, 'chat_memory'):
                messages = self.current_memory.chat_memory.messages
            
            # Convert messages to serializable format
            serializable_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    serializable_messages.append({
                        "type": "human",
                        "content": msg.content
                    })
                elif isinstance(msg, AIMessage):
                    serializable_messages.append({
                        "type": "ai",
                        "content": msg.content
                    })
            
            # Create conversation data
            conversation_data = {
                "retriever_type": retriever_type,
                "memory_type": self.current_memory_type,
                "messages": serializable_messages,
                "timestamp": logging.time.time()
            }
            
            # Save to file
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2)
            
            self.logger.info(f"Conversation history saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save conversation history: {e}")
            return False
    
    def load_conversation_history(self, file_path: str) -> bool:
        """Load conversation history from disk
        
        Args:
            file_path: Path to load the conversation history from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                self.logger.warning(f"Conversation history file not found: {file_path}")
                return False
            
            # Load from file
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            # Validate data
            if not all(key in conversation_data for key in ['messages', 'memory_type']):
                self.logger.error("Invalid conversation history file format")
                return False
            
            # Set memory type
            memory_type = conversation_data['memory_type']
            self.set_memory_type(memory_type)
            
            # Convert messages back to LangChain format
            messages = []
            for msg_data in conversation_data['messages']:
                if msg_data['type'] == 'human':
                    messages.append(HumanMessage(content=msg_data['content']))
                elif msg_data['type'] == 'ai':
                    messages.append(AIMessage(content=msg_data['content']))
            
            # Add messages to memory
            if hasattr(self.current_memory, 'chat_memory'):
                self.current_memory.chat_memory.messages = messages
            
            self.logger.info(f"Conversation history loaded from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load conversation history: {e}")
            return False
    
    def list_saved_conversations(self, directory: str) -> List[Dict[str, Any]]:
        """List all saved conversations in a directory
        
        Args:
            directory: Directory to search for conversation files
            
        Returns:
            List of conversation metadata
        """
        conversations = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            return conversations
        
        for file_path in directory_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    conversation_data = json.load(f)
                
                conversations.append({
                    "file_path": str(file_path),
                    "retriever_type": conversation_data.get("retriever_type", "unknown"),
                    "memory_type": conversation_data.get("memory_type", "unknown"),
                    "message_count": len(conversation_data.get("messages", [])),
                    "timestamp": conversation_data.get("timestamp", 0)
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to read conversation file {file_path}: {e}")
        
        return conversations