"""
Document processing for the RAG system.

This module provides document loading, chunking, and preprocessing utilities.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Set

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config.settings import Settings
from ..exceptions.base import DocumentProcessingError, handle_errors
from ..validation.models import DocumentValidator

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handle document loading, chunking, and preprocessing"""
    
    def __init__(self, settings: Settings):
        """Initialize document processor with settings
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings
        self.validator = DocumentValidator()
        self.logger = logging.getLogger(__name__)
        
        # Track existing document hashes for duplicate detection
        self.existing_file_hashes: Set[str] = set()
    
    @handle_errors(DocumentProcessingError)
    def load_documents(self, source_dir: Union[str, Path]) -> List[Document]:
        """Load and validate documents from directory
        
        Args:
            source_dir: Directory containing documents to load
            
        Returns:
            List of loaded and validated documents
            
        Raises:
            DocumentProcessingError: If document loading fails
        """
        source_dir = Path(source_dir)
        
        if not source_dir.exists():
            raise DocumentProcessingError(f"Source directory does not exist: {source_dir}")
        
        self.logger.info(f"Loading documents from: {source_dir}")
        
        # Define loaders for different file types
        loaders = {
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
        }
        
        documents = []
        total_files = 0
        
        for ext, loader_class in loaders.items():
            files = list(source_dir.glob(f"**/*{ext}"))
            total_files += len(files)
            
            if files:
                try:
                    loader = DirectoryLoader(
                        str(source_dir),
                        glob=f"**/*{ext}",
                        loader_cls=loader_class,
                        show_progress=True,
                        use_multithreading=True
                    )
                    
                    loaded_docs = loader.load()
                    
                    # Validate and sanitize documents
                    for doc in loaded_docs:
                        if self.validator.validate_document_size(doc.page_content, self.settings.max_document_size):
                            doc.page_content = self.validator.sanitize_text(doc.page_content)
                            if doc.page_content.strip():  # Only add non-empty documents
                                documents.append(doc)
                        else:
                            self.logger.warning(f"Document too large, skipping: {doc.metadata.get('source', 'unknown')}")
                    
                    self.logger.info(f"Loaded {len(loaded_docs)} document pages from {ext} file/s")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load {ext} files: {str(e)}")
                    raise DocumentProcessingError(f"Failed to load {ext} files: {str(e)}")
        
        if not documents:
            raise DocumentProcessingError(f"No valid documents found in {source_dir}")
        
        self.logger.info(f"Total document pages loaded: {len(documents)} from {total_files} files")
        return documents
    
    @handle_errors(DocumentProcessingError)
    def chunk_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks with validation
        
        Args:
            docs: List of documents to chunk
            
        Returns:
            List of chunked documents
            
        Raises:
            DocumentProcessingError: If chunking fails
        """
        if not docs:
            raise DocumentProcessingError("No documents to chunk")
        
        self.logger.info(f"Chunking {len(docs)} documents")
        
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.chunk_size_threshold,
                chunk_overlap=int(self.settings.chunk_size_threshold * self.settings.chunk_overlap_ratio),
                length_function=len,
                is_separator_regex=False,
                separators=["\n\n", "\n", " ", "", "."],
            )
            
            split_docs = text_splitter.split_documents(docs)
            
            # Filter out very small chunks
            min_chunk_size = 50
            split_docs = [doc for doc in split_docs if len(doc.page_content.strip()) >= min_chunk_size]
            
            if not split_docs:
                raise DocumentProcessingError("No valid chunks created")
            
            self.logger.info(f"Created {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            raise DocumentProcessingError(f"Failed to chunk documents: {str(e)}")
    
    def calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate MD5 hash of a file for duplicate detection
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash of file content
        """
        file_path = Path(file_path)
        hash_md5 = hashlib.md5()
        
        try:
            with open(file_path, 'rb') as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def load_existing_file_hashes(self, vectordb) -> None:
        """Load existing file hashes from vector database metadata
        
        Args:
            vectordb: Vector database instance
        """
        if not vectordb:
            return
        
        try:
            # Get all documents from vector store
            docs_data = vectordb.get()
            
            # Extract file hashes from metadata
            for metadata in docs_data.get('metadatas', []):
                if 'file_hash' in metadata:
                    self.existing_file_hashes.add(metadata['file_hash'])
            
            self.logger.info(f"Loaded {len(self.existing_file_hashes)} existing file hashes for duplicate detection")
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing file hashes: {e}")
    
    def add_file_hash_to_metadata(self, documents: List[Document]) -> List[Document]:
        """Add file hash to document metadata for duplicate detection
        
        Args:
            documents: List of documents to process
            
        Returns:
            Documents with file hash added to metadata
        """
        documents_with_hashes = []
        
        for doc in documents:
            # Get file path from metadata
            file_path = doc.metadata.get('source', '')
            if file_path and isinstance(file_path, (str, Path)):
                file_hash = self.calculate_file_hash(file_path)
                if file_hash:
                    # Add file hash to metadata
                    doc.metadata['file_hash'] = file_hash
                    doc.metadata['file_path'] = str(Path(file_path).name)
                    try:
                        doc.metadata['file_size'] = Path(file_path).stat().st_size
                        doc.metadata['file_modified'] = Path(file_path).stat().st_mtime
                    except (OSError, TypeError) as e:
                        self.logger.warning(f"Failed to get file stats for {file_path}: {e}")
            documents_with_hashes.append(doc)
        
        return documents_with_hashes
    
    def filter_duplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Filter out duplicate documents based on file hash
        
        Note: This function filters duplicate FILES, not chunks. For PDFs with multiple pages,
        PyPDFLoader creates one document per page, but we want to keep ALL pages from the same file.
        Only actual duplicate files (different files with identical content) should be filtered.
        
        Args:
            documents: List of documents to filter
            
        Returns:
            List of unique documents
        """
        unique_documents = []
        duplicate_count = 0
        seen_file_hashes_in_batch = set()
        
        # Track the first document-page encountered for each file hash to avoid duplicate files
        first_doc_per_file_hash = {}
        
        for doc in documents:
            file_hash = doc.metadata.get('file_hash', '')
            
            # Skip documents without file hash
            if not file_hash:
                unique_documents.append(doc)
                continue
            
            # Check if duplicate of existing file in vector DB
            if file_hash in self.existing_file_hashes:
                duplicate_count += 1
                self.logger.info(f"Skipping duplicate document (existing in vector DB): {doc.metadata.get('file_path', 'unknown')}")
                continue
            
            # Check if this is the first time we're seeing this file hash in the current batch
            if file_hash not in seen_file_hashes_in_batch:
                # First time seeing this file hash - keep the document
                unique_documents.append(doc)
                seen_file_hashes_in_batch.add(file_hash)
                first_doc_per_file_hash[file_hash] = doc
                
                # Log that we're seeing a new file
                self.logger.debug(f"Processing new file: {doc.metadata.get('file_path', 'unknown')} "
                                f"(file hash: {file_hash[:8]}...)")
            else:
                # We've already seen this file hash in the current batch
                # This means we have multiple pages from the same file
                # We want to keep ALL pages from the same file, so we check if this is the same file
                first_doc = first_doc_per_file_hash[file_hash]
                # Compare source file paths to ensure they're from the same file
                if (doc.metadata.get('source') == first_doc.metadata.get('source')):
                    # Same file - keep this page
                    unique_documents.append(doc)
                    self.logger.debug(f"Keeping additional page from file: {doc.metadata.get('file_path', 'unknown')} "
                                    f"(page: {doc.metadata.get('page', 'unknown')})")
                else:
                    # Different files with same hash - this is a true duplicate
                    duplicate_count += 1
                    self.logger.info(f"Skipping duplicate document (same batch): {doc.metadata.get('file_path', 'unknown')}")
        
        if duplicate_count > 0:
            self.logger.info(f"Filtered out {duplicate_count} duplicate documents")
        
        return unique_documents