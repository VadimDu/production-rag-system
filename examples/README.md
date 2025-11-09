# Production RAG System Examples

This directory contains example scripts demonstrating various features of the Production RAG System.

## Table of Contents

- [Production RAG System Examples](#production-rag-system-examples)
  - [Table of Contents](#table-of-contents)
  - [Basic Usage](#basic-usage)
  - [Conversational AI](#conversational-ai)
  - [Document Processing](#document-processing)
  - [Conversation Persistence](#conversation-persistence)
  - [Advanced Retrieval](#advanced-retrieval)
  - [Common Requirements](#common-requirements)
  - [Setup Instructions](#setup-instructions)
  - [Running Examples](#running-examples)
  - [Customization](#customization)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
    - [Getting Help](#getting-help)
  - [Contributing](#contributing)

## Basic Usage

**File**: `basic_usage.py`

Demonstrates the basic functionality of the Production RAG System:
- Creating a RAG system with default settings
- Checking system health
- Processing a simple query
- Viewing performance metrics

**Usage**:
```bash
python basic_usage.py
```

**Requirements**:
- An existing vector database with documents
- Configured LLM API

## Conversational AI

**File**: `conversational_ai.py`

Demonstrates conversational AI capabilities:
- Processing conversational queries with memory
- Using different memory types (buffer and summary)
- Getting conversation history
- Saving and loading conversations

**Usage**:
```bash
python conversational_ai.py
```

**Requirements**:
- An existing vector database with documents
- Configured LLM API
- Disk space for saving conversations

## Document Processing

**File**: `document_processing.py`

Demonstrates document processing capabilities:
- Building vector databases from documents
- Loading and chunking documents
- Handling duplicate documents
- Updating vector databases

**Usage**:
```bash
python document_processing.py
```

**Requirements**:
- Documents in supported formats (.txt, .md, .pdf, .docx)
- A `documents` directory with your documents
- Disk space for the vector database

## Conversation Persistence

**File**: `conversation_persistence.py`

Demonstrates conversation persistence features:
- Saving conversations to disk
- Loading conversations from disk
- Managing multiple conversations
- Continuing conversations across sessions

**Usage**:
```bash
python conversation_persistence.py
```

**Requirements**:
- An existing vector database with documents
- Configured LLM API
- Disk space for saving conversations

## Advanced Retrieval

**File**: `advanced_retrieval.py`

Demonstrates advanced retrieval features:
- Using different retriever types (vector semantic, BM25, ensemble)
- Comparing retrieval results
- Customizing retrieval parameters
- Performance comparison

**Usage**:
```bash
python advanced_retrieval.py
```

**Requirements**:
- An existing vector database with documents
- Configured LLM API
- Sufficient documents for BM25 retrieval

## Common Requirements

All examples require:

1. **Python 3.8+**
2. **Production RAG System** installed
3. **Dependencies** installed (see requirements.txt)
4. **Configured LLM API** (OpenAI or compatible)

## Setup Instructions

1. Install the Production RAG System:
   ```bash
   cd production_rag_system
   pip install -r requirements.txt
   ```

2. Configure your environment:
   ```bash
   # Copy the template to create your .env file
   cp examples/.env.template .env
   
   # Edit .env with your actual values
   nano .env  # or use your preferred editor
   ```

3. For document processing examples:
   ```bash
   mkdir -p examples/documents
   # Add your documents to this directory
   ```

4. For conversation examples:
   ```bash
   mkdir -p examples/conversations
   ```

5. For vector database examples:
   ```bash
   mkdir -p examples/chroma_db
   ```

6. Configure your LLM API (in .env file):
   - Set `LLM_API_KEY` to your API key
   - Set `LLM_BASE_URL` to your API endpoint
   - Set `LLM_MODEL_NAME` to your preferred model

## Running Examples

Each example can be run independently:

```bash
# Run a specific example
python examples/basic_usage.py

# Or run from the examples directory
cd examples
python basic_usage.py
```

## Customization

You can customize the examples by modifying the settings:

```python
from production_rag_system import Settings

settings = Settings(
    persist_dir="./my_chroma_db",
    embedding_model_name="all-MiniLM-L6-v2",
    llm_model_name="gpt-3.5-turbo",
    chunk_size_threshold=1000,
    chunk_overlap_ratio=0.1,
    default_k=5,
    log_level="INFO"
)
```

## Troubleshooting

### Common Issues

1. **"No documents available"**
   - Make sure you have documents in the documents directory
   - Run the document_processing example first

2. **"System not ready"**
   - Check your LLM API configuration
   - Ensure the vector database exists

3. **"Failed to save/load conversation"**
   - Check directory permissions
   - Ensure the conversations directory exists

### Getting Help

If you encounter issues:

1. Check the logs in the log file
2. Run the basic_usage.py example first
3. Check the system health status
4. Review the error messages

## Contributing

To add new examples:

1. Create a new Python file in this directory
2. Follow the existing code style and structure
3. Include comprehensive error handling
4. Add documentation to this README
5. Update the main README.md if needed