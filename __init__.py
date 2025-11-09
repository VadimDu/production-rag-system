"""
Production-Ready RAG System

A modular, production-ready Retrieval-Augmented Generation system with
comprehensive error handling, input validation, and monitoring capabilities.
"""

from .core.rag_system import ProductionRAGSystem, create_production_rag_system
from .core.device import setup_device, get_device_info
from .validation.models import QueryRequest, DocumentValidator
from .config.settings import Settings
from .exceptions.base import (
    RAGSystemError,
    ConfigurationError,
    DocumentProcessingError,
    ValidationError,
    LLMError,
    VectorDBError,
    handle_errors,
    retry_on_failure
)
from .logging.config import setup_logging, get_logger
from .monitoring.performance import PerformanceMetrics, PerformanceMonitor
from .processing.documents import DocumentProcessor
from .retrieval.retrievers import RetrieverFactory
from .retrieval.memory import MemoryManager

__version__ = "1.0.0"
__author__ = "Production RAG System Team"

__all__ = [
    # Core components
    "ProductionRAGSystem",
    "create_production_rag_system",
    "setup_device",
    "get_device_info",
    
    # Configuration
    "Settings",
    
    # Validation
    "QueryRequest",
    "DocumentValidator",
    
    # Exceptions
    "RAGSystemError",
    "ConfigurationError",
    "DocumentProcessingError",
    "ValidationError",
    "LLMError",
    "VectorDBError",
    "handle_errors",
    "retry_on_failure",
    
    # Logging
    "setup_logging",
    "get_logger",
    
    # Monitoring
    "PerformanceMetrics",
    "PerformanceMonitor",
    
    # Processing
    "DocumentProcessor",
    
    # Retrieval
    "RetrieverFactory",
    "MemoryManager",
]
