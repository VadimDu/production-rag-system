# Production RAG System - Streamlit Web App

A web interface for the Production RAG System that allows you to easily query your documents using natural language, manage conversations, and monitor system health.

## Features

- 🤖 **Natural Language Querying**: Ask questions about your documents and get intelligent answers
- 💬 **Conversation Management**: Maintain conversation history with memory options
- 📚 **Document Management**: Upload and manage your knowledge base
- ⚙️ **System Monitoring**: View system health, metrics, and configuration
- 🎨 **User-Friendly Interface**: Clean, responsive web interface

## Prerequisites

Make sure you have the Production RAG System installed and working (follow the instructions from the main repository README)

## Quick Start

### Option 1: Using the launcher script (recommended)
```bash
python run_app.py
```

### Option 2: Running directly from the parent directory
```bash
cd production_rag_system
streamlit run web_app/app.py
```

### Option 3: Running with explicit Python path (if import conflicts occur)
```bash
cd production_rag_system/web_app
PYTHONPATH=.. streamlit run app.py
```

### Option 4: Running from the analysis directory
```bash
streamlit run production_rag_system/web_app/app.py
```

The web app will open in your browser at `http://localhost:8501`

## Usage Guide

### 1. Initialize the System
- Click the "🚀 Initialize System" button in the sidebar
- The system will load your existing vector database or wait for documents to be added
- Check the system health status in the sidebar

### 2. Add Documents (if needed)
- Go to the "📚 Documents" tab
- Upload files using the file uploader (supports .txt, .md, .pdf, .docx)
- Or add documents from a directory using the directory path input
- Click "Process" to add documents to your knowledge base

### 3. Query Your Documents
- Go to the "💬 Query" tab
- Enter your question in the text input
- Choose your retrieval options:
  - **Retriever Type**: vec_semantic (default), bm25, or ensemble
  - **Conversation Memory**: Enable to maintain context across questions
  - **Memory Type**: buffer (for full history) or summary (for condensed history)
- Click "🔍 Submit Query" to get your answer

### 4. Manage Conversations
- Go to the "💭 Conversations" tab
- View your conversation history
- Clear conversation history if needed
- Save conversations to disk for later use

### 5. Monitor System
- Go to the "⚙️ Settings" tab
- View current system configuration
- Monitor performance metrics
- Export system information

## Configuration

The web app provides two ways to configure your RAG system:

### 1. Web Interface Configuration (Recommended)
Go to the "🔧 Configuration" tab to customize:
- **Database Settings**: ChromaDB persist directory, chunk size, overlap
- **Model Settings**: Embedding model, LLM model, temperature
- **Performance Settings**: Default K, max tokens, batch sizes
- **LLM Settings**: Provider, API endpoints, authentication
- **Advanced Settings**: Logging, document processing parameters

### 2. Traditional Configuration
You can also use traditional methods:
- Environment variables (see `.env.template` in the main project)
- Direct configuration file changes

### Configuration Workflow:
1. Open the "🔧 Configuration" tab (available before system initialization)
2. Adjust settings as needed
3. Click "🔧 Apply Configuration"
4. Click "🚀 Initialize System" to use your custom settings
5. Your configuration is preserved for the session

## Troubleshooting

### Import Error: "attempted relative import beyond top-level package"
This occurs due to naming conflicts with the logging module. Try these solutions:

1. **Run from the parent directory:**
   ```bash
   cd production_rag_system
   streamlit run web_app/app.py
   ```

2. **Use the launcher script:**
   ```bash
   python production_rag_system/web_app/run_app.py
   ```

3. **Set PYTHONPATH explicitly:**
   ```bash
   cd production_rag_system/web_app
   PYTHONPATH=.. streamlit run app.py
   ```

### System Initialization Issues
- Make sure your LLM server is running (e.g., LM Studio at `http://localhost:1234/v1`)
- Check that your vector database exists or upload documents to create one
- Verify your configuration settings in the main system

### Document Processing Issues
- Ensure documents are in supported formats (.txt, .md, .pdf, .docx)
- Check file permissions for document directories
- Monitor the console for error messages during processing

### Query Issues
- Make sure documents are loaded in the vector database
- Try different retriever types if results are not satisfactory
- Adjust the K parameter (number of documents to retrieve) if needed

## Development

To extend or modify the web app:

1. Edit `app.py` for main functionality changes
2. Modify the CSS styles in the `st.markdown()` block for visual changes
3. Add new features as separate functions and integrate them in the `main()` function

## File Structure

```
web_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Support

For issues related to:
- **Web App Interface**: Check this README and the app code
- **Core RAG System**: Refer to the main Production RAG System documentation
- **LLM Configuration**: Ensure your local LLM server is properly configured

## License

This project is licensed under the MIT License.