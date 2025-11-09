"""
Input validation models for the RAG system.

This module provides Pydantic models and validation utilities for ensuring
data integrity and security.
"""

import re
from pathlib import Path
from typing import Union, Optional

from pydantic import BaseModel, Field, validator

from ..exceptions.base import ValidationError


class QueryRequest(BaseModel):
    """Validated query request model"""
    question: str = Field(..., min_length=1, max_length=2000)
    max_context_length: Optional[int] = Field(default=4000, ge=1000, le=32000)
    retriever_type: str = Field(default="vec_semantic", pattern="^(bm25|vec_semantic|ensemble)$")

    @validator('question')
    def validate_question(cls, v):
        """Validate and sanitize question"""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        
        # Check for injection patterns
        dangerous_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'os\.system',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Question contains potentially harmful content")
        
        return v.strip()


class DocumentValidator:
    """Validate and sanitize document content"""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Remove potentially harmful content from documents"""
        if not text:
            return ""
        
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Limit extremely long lines
        lines = text.split('\n')
        sanitized_lines = [line[:10000] for line in lines]
        
        # Remove excessive whitespace
        text = '\n'.join(sanitized_lines)
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Limit consecutive empty lines
        
        return text.strip()
    
    @staticmethod
    def validate_document_size(content: str, max_size: int = 10_000_000) -> bool:
        """Validate document size"""
        if not content:
            return False
        
        size_bytes = len(content.encode('utf-8'))
        return size_bytes <= max_size
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path]) -> bool:
        """Validate file path for security"""
        path = Path(file_path)
        
        # Check for path traversal attempts
        if '..' in path.parts:
            return False
        
        # Check if file exists and is readable
        if not path.exists() or not path.is_file():
            return False
        
        # Check file extension
        allowed_extensions = {'.txt', '.md', '.pdf', '.docx'}
        if path.suffix.lower() not in allowed_extensions:
            return False
        
        return True