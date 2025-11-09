"""
Configuration settings for the RAG system.

This module defines the Settings class with validation for all
configuration parameters used throughout the RAG system.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Database settings - use user home directory for portability
    persist_dir: str = Field(
        default=str(Path.home() / "production_rag_system" / "chroma_db"), 
        env="RAG_PERSIST_DIR"
    )
    chunk_size_threshold: int = Field(default=1200, ge=100, le=10000)
    chunk_overlap_ratio: float = Field(default=0.1, ge=0.0, le=0.5)  # 0.1 means 10% overlap
    
    # Model settings
    embedding_model_name: str = Field(default="mixedbread-ai/mxbai-embed-large-v1", env="RAG_EMBEDDING_MODEL")
    llm_provider: str = Field(default="lmstudio", env="RAG_LLM_PROVIDER")
    llm_model_name: str = Field(default="qwen/qwen3-next-80b", env="RAG_LLM_MODEL")
    llm_api_key: str = Field(default="not-needed", env="RAG_LLM_API_KEY")
    llm_base_url: str = Field(default="http://localhost:1234/v1", env="RAG_LLM_BASE_URL")
    llm_temperature: float = Field(default=1.0, ge=0.0, le=2.0, env="RAG_LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=32000, ge=1, le=100000, env="RAG_LLM_MAX_TOKENS")
    
    # Performance settings
    embedding_batch_size: int = Field(default=32, ge=1, le=128)
    max_concurrent_requests: int = Field(default=10, ge=1, le=100)
    request_timeout: int = Field(default=30, ge=10, le=600)
    
    # Security settings
    max_query_length: int = Field(default=2000, ge=10, le=10000)
    max_document_size: int = Field(default=10_000_000, ge=1000, le=100_000_000)
    
    # Logging settings - use user home directory for portability
    log_level: str = Field(default="INFO", env="RAG_LOG_LEVEL")
    log_file: str = Field(
        default=str(Path.home() / "production_rag_system" / "rag_system.log"), 
        env="RAG_LOG_FILE"
    )
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, env="RAG_ENABLE_METRICS")
    metrics_port: int = Field(default=8000, ge=8000, le=9999, env="RAG_METRICS_PORT")
    
    # Retrieval settings
    default_retriever_type: str = Field(
        default="vec_semantic", 
        pattern="^(bm25|vec_semantic|ensemble)$",
        env="RAG_DEFAULT_RETRIEVER"
    )
    default_k: int = Field(default=5, ge=1, le=50, env="RAG_DEFAULT_K")  # The k parameter controls how many CHUNKS to retrieve (not complete docs!)
    
    # Memory settings
    default_memory_type: str = Field(
        default="buffer",
        pattern="^(buffer|summary)$",
        env="RAG_DEFAULT_MEMORY"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields for flexibility
    
    @field_validator('persist_dir')
    def validate_persist_dir(cls, v):
        """Ensure persist directory exists or can be created"""
        # Convert to absolute path if relative
        if not os.path.isabs(v):
            v = str(Path.home() / "production_rag_system" / v)
        
        if not os.path.exists(v):
            try:
                os.makedirs(v, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Cannot create persist directory: {e}")
        return v
    
    @field_validator('log_file')
    def validate_log_file(cls, v):
        """Ensure log file directory exists"""
        # Convert to absolute path if relative
        if not os.path.isabs(v):
            v = str(Path.home() / "production_rag_system" / v)
        
        log_path = Path(v)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    @field_validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator('llm_base_url')
    def validate_llm_base_url(cls, v):
        """Validate LLM base URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("LLM base URL must start with http:// or https://")
        return v.rstrip('/')
    
    @field_validator('embedding_model_name')
    def validate_embedding_model(cls, v):
        """Validate embedding model name"""
        # Add common embedding models that are known to work
        valid_models = [
            "mixedbread-ai/mxbai-embed-large-v1",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-large-en-v1.5",
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
        ]
        
        # Allow custom models but warn if not in known list
        if v not in valid_models:
            import warnings
            warnings.warn(f"Using custom embedding model: {v}. Ensure it's compatible.")
        
        return v
    
    def get_device_config(self) -> dict:
        """Get device configuration based on available hardware"""
        import torch
        
        if torch.backends.mps.is_available():
            return {"device": "mps", "device_name": "Apple Silicon GPU"}
        elif torch.cuda.is_available():
            return {"device": "cuda:0", "device_name": "NVIDIA CUDA"}
        else:
            return {"device": "cpu", "device_name": "CPU"}
    
    def get_embedding_config(self) -> dict:
        """Get embedding model configuration"""
        device_config = self.get_device_config()
        
        return {
            "model_name": self.embedding_model_name,
            "model_kwargs": {"device": device_config["device"]},
            "encode_kwargs": {
                "normalize_embeddings": True,
                "batch_size": self.embedding_batch_size,
                "dtype": "float32",
            },
            "show_progress": True,
        }
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration"""
        return {
            "model_name": self.llm_model_name,
            "openai_api_key": self.llm_api_key,
            "openai_api_base": self.llm_base_url,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "request_timeout": self.request_timeout,
        }
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled"""
        return self.log_level == "DEBUG"
    
    def to_dict(self) -> dict:
        """Convert settings to dictionary"""
        return {
            "persist_dir": self.persist_dir,
            "chunk_size_threshold": self.chunk_size_threshold,
            "chunk_overlap_ratio": self.chunk_overlap_ratio,
            "embedding_model_name": self.embedding_model_name,
            "llm_provider": self.llm_provider,
            "llm_model_name": self.llm_model_name,
            "llm_base_url": self.llm_base_url,
            "embedding_batch_size": self.embedding_batch_size,
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout": self.request_timeout,
            "max_query_length": self.max_query_length,
            "max_document_size": self.max_document_size,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "enable_metrics": self.enable_metrics,
            "default_retriever_type": self.default_retriever_type,
            "default_k": self.default_k,
            "default_memory_type": self.default_memory_type,
        }
