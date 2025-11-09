"""
Base exceptions and error handling utilities for the RAG system.

This module provides custom exceptions and decorators for consistent
error handling throughout the system.
"""

import logging
from functools import wraps
from typing import Callable, Type, Any

from tenacity import retry, stop_after_attempt, wait_exponential


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class RAGSystemError(Exception):
    """Base exception for RAG system errors"""
    pass


class ConfigurationError(RAGSystemError):
    """Configuration-related errors"""
    pass


class DocumentProcessingError(RAGSystemError):
    """Document processing errors"""
    pass


class ValidationError(RAGSystemError):
    """Input validation errors"""
    pass


class LLMError(RAGSystemError):
    """LLM-related errors"""
    pass


class VectorDBError(RAGSystemError):
    """Vector database errors"""
    pass


# ============================================================================
# ERROR HANDLING DECORATORS
# ============================================================================

def handle_errors(error_type: Type[Exception] = RAGSystemError) -> Callable:
    """Decorator for consistent error handling
    
    Args:
        error_type: Type of exception to raise on error
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise error_type(f"Operation {func.__name__} failed: {str(e)}") from e
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, min_wait: int = 1, max_wait: int = 10) -> Callable:
    """Retry decorator for external API calls
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        
    Returns:
        Decorated function with retry logic
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        reraise=True
    )