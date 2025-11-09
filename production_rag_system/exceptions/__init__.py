"""
Exception handling for the RAG system.

This module provides custom exceptions and error handling utilities
for the RAG system.
"""

from .base import (
    RAGSystemError,
    ConfigurationError,
    DocumentProcessingError,
    ValidationError,
    LLMError,
    VectorDBError,
    handle_errors,
    retry_on_failure
)

__all__ = [
    "RAGSystemError",
    "ConfigurationError",
    "DocumentProcessingError",
    "ValidationError",
    "LLMError",
    "VectorDBError",
    "handle_errors",
    "retry_on_failure"
]