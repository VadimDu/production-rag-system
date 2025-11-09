"""
Retriever components for the RAG system.

This module provides document retrieval utilities and retriever creation.
"""

import logging
from typing import Dict, Any, Optional, Union, List

from langchain_core.documents import Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from ..config.settings import Settings
from ..exceptions.base import LLMError, ValidationError, handle_errors
from ..retrieval.memory import MemoryManager

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """Factory for creating different types of retrievers"""
    
    def __init__(self, settings: Settings, llm: ChatOpenAI, vectordb, memory_manager: MemoryManager):
        """Initialize retriever factory
        
        Args:
            settings: Configuration settings
            llm: Language model instance
            vectordb: Vector database instance
            memory_manager: Memory manager instance
        """
        self.settings = settings
        self.llm = llm
        self.vectordb = vectordb
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
    
    def create_retrievers(self, k: Optional[int] = None) -> Dict[str, Any]:
        """Create different types of retrievers with configurable k value
        
        Args:
            k: Optional k value. If None, uses self.settings.default_k
            
        Returns:
            Dictionary of retrievers (vec_semantic, bm25, ensemble)
            
        Raises:
            ValidationError: If vector database is not initialized
        """
        if not self.vectordb:
            raise ValidationError("Vector database not initialized")
        
        # Use provided k or fall back to settings default
        k_value = k or self.settings.default_k
        
        retrievers = {}
        
        # Vector retriever (always available) with specified k
        retrievers["vec_semantic"] = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k_value}
        )
        
        # BM25 retriever (requires split documents) with specified k
        try:
            # Get documents from vector store for BM25
            docs_data = self.vectordb.get()
            if docs_data['documents']:
                # Create Document objects
                documents = [
                    Document(page_content=doc, metadata=meta)
                    for doc, meta in zip(docs_data['documents'], docs_data['metadatas'])
                ]
                
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = k_value
                retrievers["bm25"] = bm25_retriever
                
                # Ensemble retriever with specified k
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, retrievers["vec_semantic"]],
                    weights=[0.4, 0.6]
                )
                retrievers["ensemble"] = ensemble_retriever
                
        except Exception as e:
            self.logger.warning(f"Failed to create BM25/ensemble retrievers: {str(e)}")
        
        return retrievers
    
    @handle_errors(LLMError)
    def build_qa_chain(self, 
                      retriever_type: str = "vec_semantic",
                      use_conversation: bool = False,
                      memory_type: str = "buffer",
                      k: Optional[int] = None) -> Union[RetrievalQA, ConversationalRetrievalChain]:
        """Build QA chain with validation
        
        Args:
            retriever_type: Type of retriever to use
            use_conversation: Whether to use conversation memory
            memory_type: Type of memory to use
            k: Number of documents to retrieve
            
        Returns:
            QA chain instance
            
        Raises:
            ValidationError: If retriever type is invalid
            LLMError: If chain creation fails
        """
        # Validate retriever type
        valid_retrievers = ["bm25", "vec_semantic", "ensemble"]
        if retriever_type not in valid_retrievers:
            raise ValidationError(f"Invalid retriever type. Must be one of: {valid_retrievers}")
        
        # Use provided k or fall back to settings default
        k_value = k or self.settings.default_k
        
        # Create retrievers with custom k value
        retrievers = self.create_retrievers(k_value)
        if retriever_type not in retrievers:
            raise ValidationError(f"Retriever '{retriever_type}' not available")
        
        # Set memory
        memory = None
        if use_conversation:
            memory = self.memory_manager.set_memory_type(memory_type)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Use ONLY the following context (source documents and metadata) to answer the question. 
                Do not use prior knowledge.
                If the context doesn't contain the answer, respond with "INSUFFICIENT CONTEXT"
                
                Context: {context}"""),
            ("human", "{question}")
        ])
        
        try:
            if use_conversation:
                chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retrievers[retriever_type],
                    memory=memory,
                    combine_docs_chain_kwargs={
                        "prompt": prompt,
                        "document_separator": "\n\n"
                    },
                    return_source_documents=True,
                )
            else:
                chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retrievers[retriever_type],
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": prompt,
                        "document_separator": "\n\n"
                    }
                )
            
            self.logger.info(f"QA chain built with {retriever_type} retriever (k={k_value})")
            return chain
            
        except Exception as e:
            raise LLMError(f"Failed to build QA chain: {str(e)}")
    
    @handle_errors(LLMError)
    def build_runnable_qa_chain(self, retriever_type: str = "vec_semantic"):
        """Build runnable QA chain
        
        Args:
            retriever_type: Type of retriever to use
            
        Returns:
            Runnable QA chain
            
        Raises:
            ValidationError: If retriever type is invalid
            LLMError: If chain creation fails
        """
        retrievers = self.create_retrievers()
        if retriever_type not in retrievers:
            raise ValidationError(f"Retriever '{retriever_type}' not available")
        
        template = '''
        Answer the following question:
        {question}
        
        Use ONLY the following context (source documents and metadata) to answer the question. 
        Do not use prior knowledge.
        If the context doesn't contain the answer, respond with "INSUFFICIENT CONTEXT":
        
        {context}
        
        At the end of the response, specify the name source documents that you used.
        '''
        
        prompt_template = PromptTemplate.from_template(template)
        
        qa_chain = (
            {'context': retrievers[retriever_type], 'question': RunnablePassthrough()}
            | prompt_template 
            | self.llm
            | StrOutputParser()
        )
        
        self.logger.info(f"Runnable QA chain built with {retriever_type} retriever")
        return qa_chain