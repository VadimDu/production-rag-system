"""
Performance monitoring for the RAG system.

This module provides performance metrics tracking and system monitoring.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    query_count: int = 0
    total_query_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    
    @property
    def average_query_time(self) -> float:
        """Calculate average query time"""
        return self.total_query_time / max(self.query_count, 1)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / max(total, 1)
    
    def reset(self):
        """Reset all metrics"""
        self.query_count = 0
        self.total_query_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.error_count = 0


class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.logger = logging.getLogger(__name__)
    
    def time_query(self, retriever_type: str):
        """Decorator to time query execution
        
        Args:
            retriever_type: Type of retriever being used
            
        Returns:
            Decorated function with timing
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    self.metrics.query_count += 1
                    self.logger.debug(f"Query executed with {retriever_type} retriever")
                    return result
                except Exception as e:
                    self.metrics.error_count += 1
                    self.logger.error(f"Query failed: {str(e)}")
                    raise
                finally:
                    duration = time.time() - start_time
                    self.metrics.total_query_time += duration
                    self.logger.debug(f"Query duration: {duration:.2f}s")
            return wrapper
        return decorator
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics
        
        Returns:
            Dictionary of current performance metrics
        """
        return {
            "query_count": self.metrics.query_count,
            "average_query_time": self.metrics.average_query_time,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "error_count": self.metrics.error_count
        }
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        self.metrics.reset()
        self.logger.info("Performance metrics reset")
    
    def increment_cache_hits(self):
        """Increment cache hit count"""
        self.metrics.cache_hits += 1
    
    def increment_cache_misses(self):
        """Increment cache miss count"""
        self.metrics.cache_misses += 1
    
    def increment_errors(self):
        """Increment error count"""
        self.metrics.error_count += 1