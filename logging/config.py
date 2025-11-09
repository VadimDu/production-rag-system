"""
Logging configuration for the RAG system.

This module provides structured logging configuration and utilities.
"""

import logging
from typing import Optional

from ..config.settings import Settings


def setup_logging(settings: Optional[Settings] = None) -> logging.Logger:
    """Configure structured logging
    
    Args:
        settings: Settings object with logging configuration. If None, uses default settings.
        
    Returns:
        Configured logger instance
    """
    if settings is None:
        from ..config.settings import Settings
        settings = Settings()
    
    logger = logging.getLogger(__name__)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set level
    logger.setLevel(getattr(logging, settings.log_level))
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(settings.log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str, settings: Optional[Settings] = None) -> logging.Logger:
    """Get a logger with the specified name
    
    Args:
        name: Logger name
        settings: Settings object with logging configuration
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        setup_logging(settings)
    
    return logger