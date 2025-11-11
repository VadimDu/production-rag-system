# Production RAG System - Overview

A modular, production-ready Retrieval-Augmented Generation (RAG) system with comprehensive error handling, input validation, and monitoring capabilities.

## Features

- **Modular Architecture**: Clean separation of concerns with 9 specialized modules
- **Multiple Retrieval Types**: Support for vector semantic, BM25, and ensemble retrievers
- **Conversation Memory**: Buffer and summary memory types for conversational AI
- **Performance Monitoring**: Built-in metrics tracking and system health monitoring
- **Input Validation**: Comprehensive validation and sanitization of inputs
- **Error Handling**: Robust error handling with retry logic and detailed logging
- **Conversation Persistence**: Save and restore conversations across sessions
- **Duplicate Detection**: Automatic detection and filtering of duplicate documents
- **Web Interface**: Optional Streamlit-based web application for easy interaction

## Installation types

### Standard Installation

```bash
pip install git+https://github.com/VadimDu/production-rag-system.git
```

### From Source

```bash
git clone https://github.com/VadimDu/production-rag-system.git
cd production-rag-system
pip install -e .
```

## ⚠️ Important: LLM Backend Required

**This RAG system requires a Language Model (LLM) backend to function.** Without a connected LLM, the system cannot process queries or generate responses.

### What You Need

Choose one of the following options:

1. **Local LLM Server** (Recommended for privacy)
   - **LM Studio**: Start a local server at `http://localhost:1234/v1` 
   - **Ollama**: Run local models like `ollama serve`
   - **Oobabooga**: Text Generation WebUI with OpenAI-compatible API

2. **Cloud API Services**
   - **OpenAI**: OpenAI API with API key
   - **Anthropic**: Claude API with API key
   - **Custom**: Any OpenAI-compatible API endpoint

### Quick LLM Setup (LM Studio)

Here's a complete step-by-step guide to get your local LLM running with LM Studio:

#### Step 1: Install LM Studio
1. Download and install [LM Studio](https://lmstudio.ai/) for your operating system
2. Launch LM Studio after installation

#### Step 2: Download a Language Model
1. Click on the "Discover" tab in LM Studio
2. Search for and download a recommended model such as (depending on your memory/compute):
   - **Qwen2.5-7B-Instruct**
   - **Llama-3.1-8B-Instruct** 
   - **Mistral-7B-Instruct**
   - ** **
3. Wait for the download to complete

#### Step 3: Load the Model
1. Go to the "Chat" tab
2. Select your downloaded model from the dropdown
3. Click "Load" to load the model into memory (this may take a few minutes)
4. Test the model by sending a simple message like "Hello"

#### Step 4: Start the Local Server
1. Click on the "Local Server" tab (or go to Menu → Preferences → Local Server)
2. Set the following configuration:
   - **Server URL**: `http://localhost:1234/v1` (default)
   - **Model**: Select the model you loaded
   - **Context Length**: 4096 (or higher if your model supports it)
3. Click "Start Server"
4. You should see "Server is running on http://localhost:1234/v1"

#### Step 5: Configure the RAG System
The system will automatically connect to `http://localhost:1234/v1` with these default settings:
- **Model Name**: `qwen/qwen3-next-80b` (can be changed in settings)
- **API Key**: `not-needed` (local servers don't require API keys)
- **Base URL**: `http://localhost:1234/v1`

To customize settings, either:
- Create a `.env` file with `RAG_LLM_BASE_URL=http://localhost:1234/v1`
- Or pass custom settings to the RAG system:
  ```python
  from production_rag_system import Settings, create_production_rag_system
  
  settings = Settings(
      llm_base_url="http://localhost:1234/v1",
      llm_model_name="qwen2.5-7b-instruct"  # or your loaded model name
  )
  rag = create_production_rag_system(settings)
  ```

#### Step 6: Verify Connection
To test if your LLM backend is working:
```python
from production_rag_system import create_production_rag_system

# Create system (it will connect to LM Studio automatically)
rag = create_production_rag_system()

# Check system health
health = rag.get_system_health()
print(f"LLM Connection: {'✅ Working' if health['llm_connection'] else '❌ Failed'}")
```

**Troubleshooting:**
- If connection fails, ensure LM Studio's local server is running
- Check that the server URL matches exactly (including `http://` and `/v1`)
- Try restarting both LM Studio and your RAG system
- Some models may have different optimal context lengths - try adjusting in LM Studio

## Quick Start

### Basic Usage

```python
from production_rag_system import create_production_rag_system, QueryRequest

# Create RAG system
rag = create_production_rag_system()

# Check system health
health = rag.get_system_health()
print(f"System ready: {health['overall']}")

# Build vector database from documents
rag.build_vector_db_from_docs(["./path/to/documents"])

# Process a query
query = QueryRequest(
    question="What is artificial intelligence?",
    retriever_type="vec_semantic"
)
result = rag.process_query(query)
print(f"Answer: {result['result']}")
```

### Conversational Queries

```python
# Process conversational query
query = QueryRequest(question="Tell me more about that.", retriever_type="vec_semantic")
result = rag.process_query(query, use_conversation=True, memory_type="buffer")

# Get conversation history
history = rag.get_conversation_history(retriever_type="vec_semantic", memory_type="buffer")
print(f"Conversation History:\n{history}")
```

### Web Interface

Launch the web interface:

```bash
rag-web
```

Or run directly:

```bash
streamlit run production_rag_system/web_app/app.py
```

## Configuration

The system uses Pydantic for configuration management. You can customize settings by:

1. Creating a `.env` file in the root directory
2. Passing a Settings object directly
3. Specifying a custom config file path

### Example Settings

```python
from production_rag_system import Settings, create_production_rag_system

settings = Settings(
    persist_dir="./my_chroma_db",
    embedding_model_name="all-MiniLM-L6-v2",
    llm_model_name="gpt-3.5-turbo",
    chunk_size_threshold=1000,
    chunk_overlap_ratio=0.1,
    default_k=5,
    log_level="INFO"
)

rag = create_production_rag_system(settings)
```

### Environment Variables

Create a `.env` file with your configuration:

```bash
# Database settings
RAG_PERSIST_DIR=./chroma_db

# Model settings
RAG_EMBEDDING_MODEL=mixedbread-ai/mxbai-embed-large-v1
RAG_LLM_MODEL=qwen/qwen3-next-80b
RAG_LLM_BASE_URL=http://localhost:1234/v1

# Performance settings
RAG_DEFAULT_K=5
RAG_EMBEDDING_BATCH_SIZE=32

# Logging settings
RAG_LOG_LEVEL=INFO
RAG_LOG_FILE=./rag_system.log
```
There is a a complete example of `.env` template file in the `examples/` directory.

## Architecture

The system is organized into 9 specialized modules:

- **core/**: Main RAG system class and device configuration
- **config/**: Configuration management with Pydantic settings
- **exceptions/**: Custom exceptions and error handling decorators
- **validation/**: Input validation models and utilities
- **logging/**: Structured logging configuration
- **monitoring/**: Performance metrics and system monitoring
- **processing/**: Document loading, chunking, and preprocessing
- **retrieval/**: Document retrieval and conversation memory management
- **web_app/**: Streamlit-based web interface

## Supported Document Formats

- Text files (.txt)
- Markdown files (.md)
- PDF documents (.pdf)
- Word documents (.docx)

## LLM Providers

The system supports various LLM providers:

- **LM Studio**: Local LLM server (default)
- **OpenAI**: OpenAI API
- **Anthropic**: Anthropic Claude API
- **Custom**: Any OpenAI-compatible API endpoint

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Basic query processing
- `conversational_ai.py`: Conversational AI with memory
- `document_processing.py`: Building vector databases
- `conversation_persistence.py`: Saving and loading conversations
- `advanced_retrieval.py`: Using different retriever types

## Performance Monitoring

The system includes built-in performance monitoring:

```python
# Get system health
health = rag.get_system_health()
print(f"Query count: {health['metrics']['query_count']}")
print(f"Average query time: {health['metrics']['average_query_time']:.2f}s")
print(f"Cache hit rate: {health['metrics']['cache_hit_rate']:.2%}")

# Reset metrics
rag.reset_metrics()
```

## Error Handling

The system provides comprehensive error handling:

```python
from production_rag_system import (
    RAGSystemError,
    ConfigurationError,
    DocumentProcessingError,
    ValidationError,
    LLMError,
    VectorDBError
)

try:
    result = rag.process_query(query)
except ValidationError as e:
    print(f"Invalid query: {e}")
except LLMError as e:
    print(f"LLM error: {e}")
except RAGSystemError as e:
    print(f"System error: {e}")
```

## Advanced Usage

### Custom Retrievers

```python
# Create different retriever types
retrievers = rag.create_retrievers(k=10)

# Use specific retriever
vec_retriever = retrievers["vec_semantic"]
bm25_retriever = retrievers["bm25"]
ensemble_retriever = retrievers["ensemble"]
```

### Memory Management

```python
# Save conversation
rag.save_conversation_history(
    file_path="./conversations/chat1.json",
    retriever_type="vec_semantic",
    memory_type="buffer"
)

# Load conversation
rag.load_conversation_history(
    file_path="./conversations/chat1.json",
    retriever_type="vec_semantic",
    memory_type="buffer"
)

# Clear conversation
rag.clear_conversation_history(
    retriever_type="vec_semantic",
    memory_type="buffer"
)
```

### Document Processing

```python
# Load documents
documents = rag.load_documents("./documents")

# Chunk documents
chunks = rag.chunk_docs(documents)

# Update vector database with new documents
rag.update_vector_db(["./new_documents"], skip_duplicates=True)
```

## Requirements

- Python 3.8+
- LangChain
- ChromaDB
- Pydantic
- HuggingFace Transformers
- OpenAI-compatible LLM API (or local LLM server)

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit a pull request.

## Support

For issues and questions:
- GitHub Issues: https://github.com/VadimDu/production-rag-system/issues
- Documentation: https://github.com/VadimDu/production-rag-system#readme

## Changelog

### Version 1.0.0
- Initial release
- Core RAG functionality
- Performance monitoring
- Error handling and validation
- Conversation memory
- Document processing
- Multiple retriever types
- Web interface