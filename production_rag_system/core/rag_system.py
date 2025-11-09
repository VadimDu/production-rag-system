"""
Production-Ready Local RAG System with LangChain

This module implements a production-ready Retrieval-Augmented Generation (RAG) system
with comprehensive error handling, input validation, and configuration management.
"""

import os
from pathlib import Path
from typing import List, Dict, Union, Optional, Any

import torch
import chromadb
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import from modular components
from production_rag_system.config.settings import Settings
from production_rag_system.logger_utils.config import setup_logging
from production_rag_system.monitoring.performance import PerformanceMonitor
from production_rag_system.processing.documents import DocumentProcessor
from production_rag_system.retrieval.retrievers import RetrieverFactory
from production_rag_system.retrieval.memory import MemoryManager
from production_rag_system.validation.models import QueryRequest
from production_rag_system.exceptions.base import (
    RAGSystemError,
    ConfigurationError,
    DocumentProcessingError,
    ValidationError,
    LLMError,
    VectorDBError,
    handle_errors,
    retry_on_failure
)
from production_rag_system.core.device import setup_device


class ProductionRAGSystem:
    """
    Production-ready RAG system with comprehensive error handling,
    input validation, and monitoring capabilities.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the production RAG system"""
        self.settings = settings or Settings()
        self.logger = setup_logging(self.settings)
        self.monitor = PerformanceMonitor()
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.settings)
        
        # Initialize device
        self.device = setup_device()
        
        # Initialize core components
        self._initialize_embeddings()
        self._initialize_llm()
        self._initialize_vector_db()
        
        # Initialize retrieval and memory components
        self.memory_manager = MemoryManager(self.llm)
        self.retriever_factory = RetrieverFactory(
            self.settings,
            self.llm,
            self.vectordb,
            self.memory_manager
        )
        
        # Store persistent conversational chains
        self._conversational_chains = {}
        
        # Load existing file hashes if vector DB exists
        self.document_processor.load_existing_file_hashes(self.vectordb)
        
        self.logger.info("Production RAG System initialized successfully")
    
    @handle_errors(ConfigurationError)
    def _initialize_embeddings(self):
        """Initialize embedding model with error handling"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.settings.embedding_model_name,
                model_kwargs={'device': str(self.device)},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': self.settings.embedding_batch_size,
                    'dtype': torch.float32,
                }
            )
            
            # Test embedding
            test_embedding = self.embeddings.embed_query("test")
            if len(test_embedding) == 0:
                raise ConfigurationError("Embedding model failed to generate embeddings")
            
            self.logger.info(f"Embeddings initialized: {self.settings.embedding_model_name}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize embeddings: {str(e)}")
    
    @handle_errors(ConfigurationError)
    @retry_on_failure(max_attempts=3)
    def _initialize_llm(self):
        """Initialize LLM with retry logic"""
        try:
            self.llm = ChatOpenAI(
                model_name=self.settings.llm_model_name,
                openai_api_key=self.settings.llm_api_key,
                openai_api_base=self.settings.llm_base_url,
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_tokens,
                request_timeout=self.settings.request_timeout,
            )
            
            # Test LLM connection
            test_response = self.llm.invoke("test")
            if not test_response.content:
                raise ConfigurationError("LLM failed to generate response")
            
            self.logger.info(f"LLM initialized: {self.settings.llm_model_name}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize LLM: {str(e)}")
    
    @handle_errors(VectorDBError)
    def _initialize_vector_db(self):
        """Initialize vector database"""
        try:
            if os.path.exists(self.settings.persist_dir) and bool(os.listdir(self.settings.persist_dir)):
                self.logger.info("Loading existing vector store...")
                self.vectordb = Chroma(
                    persist_directory=self.settings.persist_dir,
                    embedding_function=self.embeddings,
                    collection_name="personal_knowledge"
                )
                
                # Verify vector store
                collection_info = self.vectordb.get()
                if not collection_info['ids']:
                    raise VectorDBError("Vector store is empty")
                
                self.logger.info(f"Loaded {len(collection_info['ids'])} document chunks")
            else:
                self.logger.info("No existing vector store found. Create one using build_vector_db_from_docs()")
                self.vectordb = None
                
        except Exception as e:
            raise VectorDBError(f"Failed to initialize vector database: {str(e)}")
    
    def load_documents(self, source_dir: Union[str, Path]) -> List:
        """Load and validate documents from directory"""
        return self.document_processor.load_documents(source_dir)
    
    def chunk_docs(self, docs: List) -> List:
        """Split documents into chunks with validation"""
        return self.document_processor.chunk_documents(docs)
    
    def process_query(self, query_request: QueryRequest, use_conversation: bool = False,
                    memory_type: str = "buffer") -> Dict[str, Any]:
        """Process a validated query request with performance monitoring
        
        Args:
            query_request: Validated query request
            use_conversation: Whether to use conversation memory
            memory_type: Type of memory to use for conversation
            
        Returns:
            Query result with answer and sources
        """
        # Use settings.default_k directly
        k_value = self.settings.default_k
        
        # For conversation mode, use persistent chain to maintain memory
        if use_conversation:
            # Create a key for the conversational chain
            chain_key = f"{query_request.retriever_type}_{memory_type}"
            
            # Create or get persistent conversational chain
            if chain_key not in self._conversational_chains:
                self._conversational_chains[chain_key] = self.retriever_factory.build_qa_chain(
                    retriever_type=query_request.retriever_type,
                    use_conversation=True,
                    memory_type=memory_type,
                    k=k_value
                )
            
            chain = self._conversational_chains[chain_key]
        else:
            # For non-conversational queries, create a new chain each time
            chain = self.retriever_factory.build_qa_chain(
                retriever_type=query_request.retriever_type,
                use_conversation=False,
                k=k_value
            )
        
        # Process query with monitoring
        @self.monitor.time_query(query_request.retriever_type)
        def execute_query():
            if use_conversation:
                return chain.invoke({"question": query_request.question})
            else:
                return chain.invoke({"query": query_request.question})
        
        try:
            result = execute_query()
            return result
        except Exception as e:
            self.monitor.increment_errors()
            self.logger.error(f"Query processing failed: {str(e)}")
            raise
    
    def query(self, query_request: QueryRequest, use_conversation: bool = False,
             memory_type: str = "buffer") -> Any:
        """Query method for backward compatibility
        
        Args:
            query_request: Validated query request
            use_conversation: Whether to use conversation memory
            memory_type: Type of memory to use for conversation
        """
        try:
            result = self.process_query(query_request, use_conversation, memory_type)
            # Create a simple response object for compatibility
            class QueryResponse:
                def __init__(self, result_data):
                    self.answer = result_data.get('result', 'No answer available')
                    self.sources = result_data.get('source_documents', [])
            
            return QueryResponse(result)
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise
    
    @handle_errors(VectorDBError)
    def update_vector_db(self, new_docs_path: List[Union[str, Path]], 
                        skip_duplicates: bool = True):
        """Update vector database with new documents with duplicate detection"""
        if not self.vectordb:
            raise VectorDBError("Vector database not initialized. Build it first.")
        
        if not new_docs_path:
            raise VectorDBError("No new document paths provided")
        
        self.logger.info(f"Updating vector database with new documents (skip_duplicates={skip_duplicates})")
        
        try:
            # Load and process new documents
            all_documents = []
            for path in new_docs_path:
                documents = self.load_documents(path)
                all_documents.extend(documents)
            
            # Add file hashes to metadata
            documents_with_hashes = self.document_processor.add_file_hash_to_metadata(all_documents)
            
            # Filter duplicates if requested
            if skip_duplicates:
                unique_documents = self.document_processor.filter_duplicate_documents(documents_with_hashes)
                self.logger.info(f"Processing {len(unique_documents)} unique new document pages (filtered from {len(documents_with_hashes)})")
            else:
                unique_documents = documents_with_hashes
            
            if not unique_documents:
                self.logger.warning("No new unique documents to add")
                return self.vectordb
            
            # Chunk documents
            split_docs = self.chunk_docs(unique_documents)
            
            # Add to vector store
            self.vectordb.add_documents(split_docs)
            self.vectordb.persist()
            
            # Update retriever factory with the updated vector database
            self.retriever_factory.vectordb = self.vectordb
            
            # Update existing file hashes with the newly processed documents
            for doc in unique_documents:
                file_hash = doc.metadata.get('file_hash', '')
                if file_hash:
                    self.document_processor.existing_file_hashes.add(file_hash)
            
            self.logger.info(f"Added {len(split_docs)} new chunks to vector database from {len(unique_documents)} documents")
            return self.vectordb
            
        except Exception as e:
            raise VectorDBError(f"Failed to update vector database: {str(e)}")
    
    def list_existing_documents(self) -> Dict[str, Any]:
        """List all documents in the vector store"""
        if not self.vectordb:
            return {"ids": [], "metadatas": [], "documents": []}
        
        try:
            return self.vectordb.get()
        except Exception as e:
            self.logger.error(f"Failed to list documents: {str(e)}")
            return {"ids": [], "metadatas": [], "documents": []}
    
    def create_retrievers(self, k: Optional[int] = None) -> Dict[str, Any]:
        """Create different types of retrievers with configurable k value"""
        return self.retriever_factory.create_retrievers(k)
    
    def initialize_retrievers(self) -> Dict[str, Any]:
        """Initialize retrievers (alias for create_retrievers for backward compatibility)"""
        return self.create_retrievers()
    
    def set_memory_type(self, memory_type: str = "buffer"):
        """Set conversation memory type"""
        self.memory_manager.set_memory_type(memory_type)
    
    def build_qa_chain(self, 
                      retriever_type: str = "vec_semantic",
                      use_conversation: bool = False,
                      memory_type: str = "buffer",
                      k: Optional[int] = None):
        """Build QA chain with validation"""
        return self.retriever_factory.build_qa_chain(
            retriever_type=retriever_type,
            use_conversation=use_conversation,
            memory_type=memory_type,
            k=k
        )
    
    def build_runnable_qa_chain(self, retriever_type: str = "vec_semantic"):
        """Build runnable QA chain"""
        return self.retriever_factory.build_runnable_qa_chain(retriever_type)
    
    def get_conversation_history(self, retriever_type: str = "vec_semantic",
                            memory_type: str = "buffer", chain=None) -> str:
        """Get formatted conversation history
        
        Args:
            retriever_type: Type of retriever used (required if chain is not provided)
            memory_type: Type of memory used (required if chain is not provided)
            chain: Optional chain instance to get history from
            
        Returns:
            Formatted conversation history
        """
        # If a chain is provided, use it directly
        if chain is not None:
            return self.memory_manager.get_conversation_history(chain)
        
        # Otherwise, use the persistent chain
        chain_key = f"{retriever_type}_{memory_type}"
        if chain_key in self._conversational_chains:
            return self.memory_manager.get_conversation_history(self._conversational_chains[chain_key])
        else:
            return "No conversation history available"
    
    def clear_conversation_history(self, retriever_type: str = "vec_semantic",
                               memory_type: str = "buffer") -> None:
        """Clear conversation history for specific retriever and memory type
        
        Args:
            retriever_type: Type of retriever used
            memory_type: Type of memory used
        """
        chain_key = f"{retriever_type}_{memory_type}"
        if chain_key in self._conversational_chains:
            self._conversational_chains[chain_key].memory.clear()
            self.logger.info(f"Cleared conversation history for {retriever_type}_{memory_type}")
    
    def save_conversation_history(self, file_path: str, retriever_type: str = "vec_semantic",
                                memory_type: str = "buffer") -> bool:
        """Save conversation history to disk
        
        Args:
            file_path: Path to save the conversation history
            retriever_type: Type of retriever used for this conversation
            memory_type: Type of memory used for this conversation
            
        Returns:
            True if successful, False otherwise
        """
        chain_key = f"{retriever_type}_{memory_type}"
        if chain_key in self._conversational_chains:
            return self.memory_manager.save_conversation_history(file_path, retriever_type)
        else:
            self.logger.warning(f"No conversation history found for {retriever_type}_{memory_type}")
            return False
    
    def load_conversation_history(self, file_path: str, retriever_type: str = "vec_semantic",
                                memory_type: str = "buffer") -> bool:
        """Load conversation history from disk and attach to current chain
        
        Args:
            file_path: Path to load the conversation history from
            retriever_type: Type of retriever used for this conversation
            memory_type: Type of memory used for this conversation
            
        Returns:
            True if successful, False otherwise
        """
        # Load conversation history into memory manager
        success = self.memory_manager.load_conversation_history(file_path)
        
        if success:
            # Update the conversational chain with the loaded memory
            chain_key = f"{retriever_type}_{memory_type}"
            if chain_key in self._conversational_chains:
                # Update the chain's memory to use the loaded memory
                self._conversational_chains[chain_key].memory = self.memory_manager.current_memory
                self.logger.info(f"Attached loaded conversation history to {retriever_type}_{memory_type} chain")
            else:
                # Create a new chain with the loaded memory
                self._conversational_chains[chain_key] = self.retriever_factory.build_qa_chain(
                    retriever_type=retriever_type,
                    use_conversation=True,
                    memory_type=memory_type,
                    k=self.settings.default_k
                )
                self.logger.info(f"Created new {retriever_type}_{memory_type} chain with loaded conversation history")
        
        return success
    
    def list_saved_conversations(self, directory: str) -> List[Dict[str, Any]]:
        """List all saved conversations in a directory
        
        Args:
            directory: Directory to search for conversation files
            
        Returns:
            List of conversation metadata
        """
        return self.memory_manager.list_saved_conversations(directory)
    
    def get_current_memory_content(self, retriever_type: str = "vec_semantic", memory_type: str = "summary") -> str:
        """Get the current memory content (summary or buffer) for a specific conversation
        
        Args:
            retriever_type: Type of retriever used
            memory_type: Type of memory used
            
        Returns:
            Current memory content or empty string if not available
        """
        # Get the chain for the specified retriever and memory type
        chain_key = f"{retriever_type}_{memory_type}"
        if chain_key in self._conversational_chains:
            chain = self._conversational_chains[chain_key]
            if hasattr(chain, 'memory'):
                memory = chain.memory
                # Try different possible attributes where content might be stored
                content = ""
                if hasattr(memory, 'summary'):
                    content = memory.summary
                elif hasattr(memory, 'buffer'):
                    content = memory.buffer
                
                return content
        
        return ""
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health information"""
        health_status = {
            "vector_db": self.vectordb is not None,
            "embedding_model": hasattr(self, 'embeddings'),
            "llm_connection": hasattr(self, 'llm'),
            "metrics": self.monitor.get_metrics(),
            "settings": {
                "persist_dir": self.settings.persist_dir,
                "embedding_model": self.settings.embedding_model_name,
                "llm_model": self.settings.llm_model_name,
                "chunk_size": self.settings.chunk_size_threshold,
                "chunk_overlap_ratio": self.settings.chunk_overlap_ratio,
                "default_k": self.settings.default_k,
                "llm_temperature": self.settings.llm_temperature,
                "llm_max_tokens": self.settings.llm_max_tokens,
            }
        }

        # Check document count
        if self.vectordb:
            docs = self.list_existing_documents()
            health_status["document_count"] = len(docs['ids'])
        else:
            health_status["document_count"] = 0
        
        health_status["overall"] = all([
            health_status["vector_db"],
            health_status["embedding_model"],
            health_status["llm_connection"]
        ])
        
        return health_status
    
    @handle_errors(VectorDBError)
    def build_vector_db_from_docs(self, file_paths: List[Union[str, Path]], 
                                  skip_duplicates: bool = True):
        """Build vector database from documents with duplicate detection"""
        if not file_paths:
            raise VectorDBError("No file paths provided")
        
        self.logger.info(f"Building vector database from documents (skip_duplicates={skip_duplicates})")
        
        try:
            # Load and process documents
            all_documents = []
            for path in file_paths:
                documents = self.load_documents(path)
                all_documents.extend(documents)
            
            # Add file hashes to metadata
            documents_with_hashes = self.document_processor.add_file_hash_to_metadata(all_documents)
            
            # Filter duplicates if requested
            if skip_duplicates:
                unique_documents = self.document_processor.filter_duplicate_documents(documents_with_hashes)
                self.logger.info(f"Processing {len(unique_documents)} unique document pages (filtered from {len(documents_with_hashes)})")
            else:
                unique_documents = documents_with_hashes
            
            if not unique_documents:
                self.logger.warning("No unique documents to process after duplicate filtering")
                return self.vectordb if self.vectordb else None
            
            # Chunk documents
            split_docs = self.chunk_docs(unique_documents)
            
            # Create vector store
            self.vectordb = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.settings.persist_dir,
                collection_name="personal_knowledge"
            )
            
            # Update retriever factory with the new vector database
            self.retriever_factory.vectordb = self.vectordb
            
            # Update existing file hashes with the newly processed documents
            for doc in unique_documents:
                file_hash = doc.metadata.get('file_hash', '')
                if file_hash:
                    self.document_processor.existing_file_hashes.add(file_hash)
            
            self.logger.info(f"Vector database built with {len(split_docs)} chunks from {len(unique_documents)} documents")
            return self.vectordb
            
        except Exception as e:
            raise VectorDBError(f"Failed to build vector database: {str(e)}")
    
    def get_document_duplicates_info(self) -> Dict[str, Any]:
        """Get information about existing documents and potential duplicates"""
        if not self.vectordb:
            return {"existing_files": [], "duplicate_hashes_count": 0}
        
        try:
            docs_data = self.vectordb.get()
            existing_files = []
            
            for metadata in docs_data.get('metadatas', []):
                if 'file_hash' in metadata:
                    existing_files.append({
                        'file_path': metadata.get('file_path', 'unknown'),
                        'file_hash': metadata.get('file_hash', ''),
                        'file_size': metadata.get('file_size', 0),
                        'file_modified': metadata.get('file_modified', 0)
                    })
            
            return {
                "existing_files": existing_files,
                "duplicate_hashes_count": len(self.document_processor.existing_file_hashes)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get duplicate info: {e}")
            return {"existing_files": [], "duplicate_hashes_count": 0}
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.monitor.reset_metrics()
        self.logger.info("Metrics reset")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_production_rag_system(config_file_or_settings: Optional[Union[str, Settings]] = None) -> ProductionRAGSystem:
    """Factory function to create production RAG system
    
    Args:
        config_file_or_settings: Either a path to config file or a Settings object.
                                If None, uses default settings.
    
    Returns:
        ProductionRAGSystem instance
    """
    if config_file_or_settings is None:
        # Use default settings
        settings = Settings()
    elif isinstance(config_file_or_settings, Settings):
        # Use provided settings object
        settings = config_file_or_settings
    elif isinstance(config_file_or_settings, str) and os.path.exists(config_file_or_settings):
        # Load settings from config file
        settings = Settings(_env_file=config_file_or_settings)
    else:
        # Invalid parameter
        raise ValueError("config_file_or_settings must be None, a Settings object, or a valid config file path")
    
    return ProductionRAGSystem(settings)


if __name__ == "__main__":
    # Example usage
    import sys
    
    try:
        # Create system
        rag = create_production_rag_system()
        
        # Print health status
        health = rag.get_system_health()
        print("System Health:", health)
        
        # Example query if vector DB exists
        if health["vector_db"] and health["document_count"] > 0:
            query = QueryRequest(question="What is the main topic of these documents?")
            result = rag.process_query(query)
            print("Query Result:", result)
        else:
            print("No documents available. Please build vector database first.")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)