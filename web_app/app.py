"""
Streamlit Web App for Production RAG System

This is a web interface for the Production RAG System that allows users to:
- Query their documents using natural language
- Manage conversations with memory
- View system health and metrics
- Manage documents and settings
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time
import json

# Add the parent directory to the path so we can import the module
#sys.path.append(str(Path(__file__).parent.parent))
production_rag_system_path = Path.home()/"biotax/analysis"
sys.path.append(str(production_rag_system_path))

from production_rag_system import create_production_rag_system, QueryRequest, Settings


# Cache expensive operations
@st.cache_resource(show_spinner="Initializing RAG System...")
def get_cached_rag_system(settings=None):
    """Create and cache the RAG system instance with custom settings"""
    return create_production_rag_system(settings)

# Cache data that changes occasionally only every few minutes (ttl)
@st.cache_data(ttl=300, show_spinner="Updating system health...")
def get_cached_system_health(_rag_system):
    """Get and cache system health for 5 minutes"""
    return _rag_system.get_system_health()

# Cache data that changes occasionally only every few minutes (ttl)
@st.cache_data(ttl=60, show_spinner="Loading documents...")
def get_cached_documents(_rag_system):
    """Cache document list for 1 minute"""
    if _rag_system.vectordb:
        return _rag_system.list_existing_documents()
    return {"ids": [], "metadatas": [], "documents": []}


# Configure Streamlit page
st.set_page_config(
    page_title="Production RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .health-indicator {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .health-good {
        background-color: #d4edda;
        color: #155724;
    }
    .health-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    .health-error {
        background-color: #f8d7da;
        color: #721c24;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .ai-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-document {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'system_health' not in st.session_state:
        st.session_state.system_health = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_retriever_type' not in st.session_state:
        st.session_state.current_retriever_type = "vec_semantic"
    if 'current_memory_type' not in st.session_state:
        st.session_state.current_memory_type = "buffer"
    if 'use_conversation' not in st.session_state:
        st.session_state.use_conversation = False
    if 'custom_settings' not in st.session_state:
        st.session_state.custom_settings = None
    if 'show_advanced_config' not in st.session_state:
        st.session_state.show_advanced_config = False


def initialize_rag_system():
    """Initialize the RAG system with error handling and caching"""
    try:
        # Get cached RAG system with custom settings if available
        rag = get_cached_rag_system(st.session_state.custom_settings)
        st.session_state.rag_system = rag
        
        # Get cached system health
        health = get_cached_system_health(rag)
        st.session_state.system_health = health
        
        if health["overall"]:
            st.success("✅ RAG System initialized successfully!")
            if st.session_state.custom_settings:
                st.info("🔧 Using custom configuration")
        else:
            st.warning("⚠️ RAG System initialized with warnings")
            
        return True
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG System: {str(e)}")
        return False


def display_system_health():
    """Display system health status in sidebar with caching"""
    if st.session_state.rag_system:
        # Get fresh health data (cached)
        health = get_cached_system_health(st.session_state.rag_system)
        st.session_state.system_health = health
        
        st.subheader("🔍 System Health")
        
        # Overall status
        if health["overall"]:
            st.markdown('<div class="health-indicator health-good">✅ System Healthy</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="health-indicator health-error">❌ System Issues</div>',
                       unsafe_allow_html=True)
        
        # Component status
        st.write("**Components:**")
        st.write(f"- Vector DB: {'✅' if health['vector_db'] else '❌'}")
        st.write(f"- Embedding Model: {'✅' if health['embedding_model'] else '❌'}")
        st.write(f"- LLM Connection: {'✅' if health['llm_connection'] else '❌'}")
        
        # Document count
        st.write(f"**Document Chunks:** {health['document_count']}")

        # Performance metrics
        if "metrics" in health:
            metrics = health["metrics"]
            st.write("**Performance:**")
            st.write(f"- Queries: {metrics['query_count']}")
            st.write(f"- Avg Time: {metrics['average_query_time']:.2f}s")
            st.write(f"- Cache Hit: {metrics['cache_hit_rate']:.2%}")
        
        # Add refresh button for health
        if st.button("🔄 Refresh Health", key="refresh_health"):
            # Clear the health cache
            get_cached_system_health.clear()
            st.rerun()


def query_interface():
    """Main query interface"""
    st.header("💬 Query Your Documents")
    
    if not st.session_state.rag_system or not st.session_state.system_health["vector_db"]:
        st.warning("⚠️ Please initialize the system and ensure documents are loaded")
        return
    
    # Example questions
    example_questions = [
        "What is the main topic of these documents?",
        "Summarize the key findings",
        "What are the important conclusions?",
        "Explain the methodology used",
        "What data sources were referenced?"
    ]
    
    # Query input with example questions
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Enter your question:", placeholder="What would you like to know?")
    with col2:
        if st.button("📋 Example", help="Show example questions"):
            st.session_state.show_examples = not st.session_state.get('show_examples', False)
    
    # Show example questions if toggled
    if st.session_state.get('show_examples', False):
        st.write("**Example Questions:**")
        for i, example_q in enumerate(example_questions, 1):
            if st.button(f"{i}. {example_q}", key=f"example_{i}"):
                query = example_q
                st.rerun()
    
    # Query options
    col1, col2, col3 = st.columns(3)
    with col1:
        retriever_type = st.selectbox(
            "Retriever Type:",
            ["vec_semantic", "bm25", "ensemble"],
            index=0,
            key="query_retriever_type"
        )
        st.session_state.current_retriever_type = retriever_type
    
    with col2:
        use_conversation = st.checkbox("Use Conversation Memory")
        st.session_state.use_conversation = use_conversation
    
    with col3:
        if use_conversation:
            memory_type = st.selectbox(
                "Memory Type:",
                ["buffer", "summary"],
                index=0,
                key="query_memory_type"
            )
            st.session_state.current_memory_type = memory_type
    
    # Query button
    if st.button("🔍 Submit Query") and query:
        with st.spinner("Processing your query..."):
            try:
                # Create query request
                query_request = QueryRequest(
                    question=query,
                    retriever_type=retriever_type
                )
                
                # Process query
                start_time = time.time()
                result = st.session_state.rag_system.process_query(
                    query_request,
                    use_conversation=use_conversation,
                    memory_type=memory_type if use_conversation else "buffer"
                )
                end_time = time.time()
                
                # Display answer
                st.markdown('<div class="ai-message chat-message">', unsafe_allow_html=True)
                st.write("**Answer:**")
                # Handle both conversation and non-conversation results
                answer = result.get('result', result.get('answer', 'No answer available'))
                st.write(answer)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display sources
                if 'source_documents' in result and result['source_documents']:
                    st.write("**Sources:**")
                    for i, doc in enumerate(result['source_documents'][:3], 1):
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'N/A')
                        st.markdown(
                            f'<div class="source-document">{i}. {source} (Page: {page})</div>',
                            unsafe_allow_html=True
                        )
                
                # Show query time
                st.write(f"⏱️ Query processed in {end_time - start_time:.2f} seconds")
                
                # Add to conversation history
                answer = result.get('result', result.get('answer', 'No answer available'))
                st.session_state.conversation_history.append({
                    "question": query,
                    "answer": answer,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")


def conversation_management():
    """Conversation management interface"""
    st.header("💭 Conversation Management")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ Please initialize the system first")
        return
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.subheader("Current Session History")
        for i, turn in enumerate(st.session_state.conversation_history, 1):
            with st.expander(f"Turn {i} - {turn['timestamp']}"):
                st.write(f"**Question:** {turn['question']}")
                st.write(f"**Answer:** {turn['answer']}")
    
    # Conversation controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Conversation History"):
            st.session_state.rag_system.clear_conversation_history(
                retriever_type=st.session_state.current_retriever_type,
                memory_type=st.session_state.current_memory_type
            )
            st.session_state.conversation_history = []
            st.success("✅ Conversation history cleared")
            st.rerun()
    
    with col2:
        # Save conversation
        save_path = st.text_input("Save conversation to:", value="./conversation.json")
        if st.button("💾 Save Conversation"):
            try:
                success = st.session_state.rag_system.save_conversation_history(
                    file_path=save_path,
                    retriever_type=st.session_state.current_retriever_type,
                    memory_type=st.session_state.current_memory_type
                )
                if success:
                    st.success(f"✅ Conversation saved to {save_path}")
                else:
                    st.error("❌ Failed to save conversation")
            except Exception as e:
                st.error(f"❌ Error saving conversation: {str(e)}")
        
        st.markdown("---")
        
        # Load conversation section
        st.subheader("📂 Load Conversation")
        
        load_path = st.text_input("Load conversation from:", value="./conversation.json")
        col1, col2 = st.columns(2)
        with col1:
            load_retriever_type = st.selectbox(
                "Retriever Type:",
                ["vec_semantic", "bm25", "ensemble"],
                index=0,
                key="load_retriever_type"
            )
        with col2:
            load_memory_type = st.selectbox(
                "Memory Type:",
                ["buffer", "summary"],
                index=0,
                key="load_memory_type"
            )
        
        if st.button("📂 Load Conversation"):
            try:
                # Clear current conversation history first
                st.session_state.rag_system.clear_conversation_history(
                    retriever_type=load_retriever_type,
                    memory_type=load_memory_type
                )
                
                # Load the conversation
                success = st.session_state.rag_system.load_conversation_history(
                    file_path=load_path,
                    retriever_type=load_retriever_type,
                    memory_type=load_memory_type
                )
                
                if success:
                    st.success(f"✅ Conversation loaded from {load_path}")
                    # Show the loaded conversation history
                    loaded_history = st.session_state.rag_system.get_conversation_history(
                        retriever_type=load_retriever_type,
                        memory_type=load_memory_type
                    )
                    with st.expander("View Loaded Conversation"):
                        st.write(loaded_history)
                else:
                    st.error("❌ Failed to load conversation")
            except Exception as e:
                st.error(f"❌ Error loading conversation: {str(e)}")


def document_management():
    """Document management interface"""
    st.header("📚 Document Management")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ Please initialize the system first")
        return
    
    # Document upload section
    st.subheader("📤 Add Documents")
    
    # Show VectorDB path info if custom settings are applied
    if st.session_state.custom_settings:
        persist_dir = st.session_state.custom_settings.persist_dir
        st.info(f"📁 Using custom VectorDB path: `{persist_dir}`")
        if st.session_state.rag_system:
            current_persist_dir = st.session_state.rag_system.settings.persist_dir
            if persist_dir != current_persist_dir:
                st.warning("⚠️ System was initialized with a different VectorDB path. Reinitialize to use your custom path.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=['txt', 'md', 'pdf', 'docx'],
        accept_multiple_files=True,
        help="Upload documents to add to the knowledge base"
    )
    
    if uploaded_files:
        if st.button("📥 Process Uploaded Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Create temporary directory for uploaded files
                    import tempfile
                    import shutil
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save uploaded files
                        file_paths = []
                        for uploaded_file in uploaded_files:
                            # Create a more structured temp path
                            temp_subdir = Path(temp_dir) / "uploaded_docs"
                            temp_subdir.mkdir(exist_ok=True)
                            temp_path = temp_subdir / uploaded_file.name
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            file_paths.append(temp_path)
                        
                        # Also add the temp directory itself as a path for the document loader
                        file_paths.append(temp_subdir)
                        
                        # Process documents
                        if st.session_state.rag_system.vectordb is None:
                            # Create new vector database
                            st.session_state.rag_system.build_vector_db_from_docs(file_paths)
                            st.success("✅ Vector database created with uploaded documents")
                        else:
                            # Update existing vector database
                            st.session_state.rag_system.update_vector_db(file_paths)
                            st.success("✅ Documents added to existing vector database")
                        
                        # Clear caches and update system health
                        get_cached_documents.clear()
                        get_cached_system_health.clear()
                        st.session_state.system_health = get_cached_system_health(st.session_state.rag_system)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
    
    # Document directory input
    st.subheader("📁 Add Documents from Directory")
    doc_dir = st.text_input(
        "Enter directory path:",
        placeholder="/path/to/your/documents"
    )
    
    if doc_dir and st.button("📂 Process Directory"):
        if os.path.exists(doc_dir) and os.path.isdir(doc_dir):
            with st.spinner("Processing directory..."):
                try:
                    if st.session_state.rag_system.vectordb is None:
                        # Create new vector database
                        st.session_state.rag_system.build_vector_db_from_docs([doc_dir])
                        st.success("✅ Vector database created with documents from directory")
                        time.sleep(2)  # Give user time to see the message
                    else:
                        # Update existing vector database
                        st.session_state.rag_system.update_vector_db([doc_dir])
                        st.success("✅ Documents from directory added to vector database")
                        time.sleep(2)  # Give user time to see the message
                    
                    # Clear caches and update system health
                    get_cached_documents.clear()
                    get_cached_system_health.clear()
                    st.session_state.system_health = get_cached_system_health(st.session_state.rag_system)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing directory: {str(e)}")
        else:
            st.error("❌ Invalid directory path")
    
    # View existing documents
    st.subheader("📋 Existing Documents")
    if st.session_state.rag_system.vectordb:
        try:
            # Get cached documents
            docs = get_cached_documents(st.session_state.rag_system)
            if docs['ids']:
                st.write(f"**Total document chunks:** {len(docs['ids'])}")
                
                # Show unique documents
                unique_docs = {}
                for i, metadata in enumerate(docs.get('metadatas', [])):
                    if metadata:
                        source = metadata.get('source', 'Unknown')
                        if source not in unique_docs:
                            unique_docs[source] = {
                                'chunks': 0,
                                'file_hash': metadata.get('file_hash', 'N/A'),
                                'file_size': metadata.get('file_size', 'N/A'),
                                'file_modified': metadata.get('file_modified', 'N/A')
                            }
                        unique_docs[source]['chunks'] += 1
                
                if unique_docs:
                    for source, info in unique_docs.items():
                        with st.expander(f"📄 {source}"):
                            st.write(f"**Chunks:** {info['chunks']}")
                            st.write(f"**File Hash:** {info['file_hash']}")
                            st.write(f"**File Size:** {info['file_size']} bytes")
                            st.write(f"**Last Modified:** {info['file_modified']}")
                else:
                    st.info("No document metadata available")
            else:
                st.info("No documents found in vector database")
            
            # Add refresh button for documents
            if st.button("🔄 Refresh Documents", key="refresh_docs"):
                # Clear the documents cache
                get_cached_documents.clear()
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error listing documents: {str(e)}")
    else:
        st.info("No vector database available")



def settings_page():
    """Unified settings and configuration page"""
    st.header("⚙️ System Settings & Configuration")
    
    # Configuration Section (available always)
    st.subheader("🔧 Configure System")
    st.write("Configure your RAG system settings. These will be used when you click 'Initialize System'.")
    
    # Show current custom settings if they exist
    if st.session_state.custom_settings:
        st.success("🔧 Custom configuration is applied")
        with st.expander("View Current Custom Settings"):
            st.json(st.session_state.custom_settings.to_dict())
    
    # Toggle advanced configuration
    st.session_state.show_advanced_config = st.checkbox(
        "Show Advanced Configuration",
        value=st.session_state.show_advanced_config
    )
    
    # Basic configuration
    st.write("**Basic Configuration:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Database settings
        st.write("**Database Settings:**")
        persist_dir = st.text_input(
            "ChromaDB Persist Directory:",
            value="",
            placeholder="Enter path to VectorDB directory (required)",
            help="Directory where ChromaDB will store vector data"
        )
        
        default_k = st.number_input(
            "Default K (documents *chunks* to retrieve):",
            min_value=1,
            max_value=50,
            value=5,
            help="Number of document chunks to retrieve for each query"
        )
    
    with col2:
        # Model settings
        st.write("**Model Settings:**")
        embedding_model = st.selectbox(
            "Embedding Model:",
            options=[
                "mixedbread-ai/mxbai-embed-large-v1",
                "all-MiniLM-L6-v2",
                "BAAI/bge-small-en-v1.5",
                "BAAI/bge-large-en-v1.5",
                "all-mpnet-base-v2"
            ],
            index=0,
            help="Model used for creating document embeddings"
        )
        
        llm_temperature = st.slider(
            "LLM Temperature:",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Controls randomness in LLM responses (0=deterministic, 2=creative)"
        )
    
    if st.session_state.show_advanced_config:
        st.write("**Advanced Configuration:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Document Processing:**")
            chunk_size = st.number_input(
                "Chunk Size Threshold:",
                min_value=100,
                max_value=10000,
                value=1200,
                help="Maximum size of document chunks"
            )
            
            chunk_overlap = st.number_input(
                "Chunk Overlap Ratio:",
                min_value=0.0,
                max_value=0.5,
                value=0.1,
                step=0.05,
                help="Fraction of overlap between chunks"
            )
            
            embedding_batch_size = st.number_input(
                "Embedding Batch Size:",
                min_value=1,
                max_value=128,
                value=32,
                help="Number of documents to process at once"
            )
        
        with col2:
            st.write("**Performance & Logging:**")
            llm_max_tokens = st.number_input(
                "LLM Max Tokens:",
                min_value=1000,
                max_value=100000,
                value=32000,
                step=1000,
                help="Maximum tokens in LLM responses"
            )
            
            log_level = st.selectbox(
                "Log Level:",
                options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                index=1,
                help="Level of logging detail"
            )
            
            log_file = st.text_input(
                "Log File Path:",
                value="",
                placeholder="Enter path to log file (optional)",
                help="Path to log file"
            )
    
    # LLM Configuration
    st.write("**LLM Configuration:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        llm_provider = st.selectbox(
            "LLM Provider:",
            options=["lmstudio", "openai", "anthropic"],
            index=0,
            help="Provider for the language model"
        )
        
        llm_model = st.text_input(
            "LLM Model Name:",
            value="qwen/qwen3-next-80b",
            help="Name of the LLM model to use"
        )
    
    with col2:
        llm_base_url = st.text_input(
            "LLM Base URL:",
            value="http://localhost:1234/v1",
            help="API endpoint for LLM"
        )
        
        llm_api_key = st.text_input(
            "LLM API Key:",
            value="not-needed",
            type="password",
            help="API key for LLM (if required)"
        )

        request_timeout = st.number_input(
            "LLM Request Timeout (seconds):",
            min_value=10,
            max_value=600,
            value=30,
            help="Timeout for LLM API requests"
        )

    # Configuration buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔧 Apply Configuration", type="primary"):
            try:
                # Validate required fields
                if not persist_dir.strip():
                    st.error("❌ VectorDB Persist Directory is required!")
                    return
                
                # Create settings object
                custom_settings = Settings(
                    persist_dir=persist_dir,
                    embedding_model_name=embedding_model,
                    llm_model_name=llm_model,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    llm_temperature=llm_temperature,
                    llm_max_tokens=llm_max_tokens,
                    chunk_size_threshold=chunk_size if st.session_state.show_advanced_config else 1200,
                    chunk_overlap_ratio=chunk_overlap if st.session_state.show_advanced_config else 0.1,
                    embedding_batch_size=embedding_batch_size if st.session_state.show_advanced_config else 32,
                    default_k=default_k,
                    log_level=log_level if st.session_state.show_advanced_config else "INFO",
                    log_file=log_file if log_file.strip() else str(Path.home() / "biotax/analysis/rag_system.log")
                )
                
                # Store in session state
                st.session_state.custom_settings = custom_settings
                
                # Clear cached system to force reinitialization with new settings
                get_cached_rag_system.clear()
                get_cached_system_health.clear()
                
                st.success("✅ Configuration applied! Click 'Initialize System' in the sidebar to use these settings.")
                if st.session_state.rag_system:
                    st.warning("⚠️ System is already running. Reinitialize to apply new settings.")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error applying configuration: {str(e)}")
    
    with col2:
        if st.button("🔄 Reset to Default Settings"):
            st.session_state.custom_settings = None
            get_cached_rag_system.clear()
            get_cached_system_health.clear()
            st.success("✅ Reset to default settings. Click 'Initialize System' to use defaults.")
            st.rerun()
    
    with col3:
        if st.button("📋 Export Configuration"):
            if st.session_state.custom_settings:
                config_data = st.session_state.custom_settings.to_dict()
                st.download_button(
                    label="📥 Download Config",
                    data=json.dumps(config_data, indent=2),
                    file_name="rag_config.json",
                    mime="application/json"
                )
            else:
                st.warning("No custom configuration to export")
    
    # Divider
    st.markdown("---")
    
    # Current System Status (only if system is initialized)
    if st.session_state.rag_system:
        st.subheader("📊 Current System Status")
        
        # Current settings
        if st.session_state.system_health:
            settings_data = st.session_state.system_health["settings"]
            
            # Display settings in a more readable format
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Active Settings:**")
                st.write(f"- Persist Directory: `{settings_data.get('persist_dir', 'N/A')}`")
                st.write(f"- Chunk Size: {settings_data.get('chunk_size', 'N/A')}")
                st.write(f"- Chunk Overlap: {settings_data.get('chunk_overlap_ratio', 'N/A')}")
                
                st.write("**Model Settings:**")
                st.write(f"- Embedding Model: `{settings_data.get('embedding_model', 'N/A')}`")
                st.write(f"- LLM Model: `{settings_data.get('llm_model', 'N/A')}`")
                st.write(f"- LLM Temperature: {settings_data.get('llm_temperature', 'N/A')}")
            
            with col2:
                st.write("**Performance Settings:**")
                st.write(f"- Default K: {settings_data.get('default_k', 'N/A')}")
                st.write(f"- Max Tokens: {settings_data.get('llm_max_tokens', 'N/A')}")
                
                st.write("**Retrieval Settings:**")
                st.write(f"- Default Retriever: {settings_data.get('default_retriever_type', 'N/A')}")
                st.write(f"- Default Memory: {settings_data.get('default_memory_type', 'N/A')}")
        
        # System performance metrics
        st.subheader("📈 Performance Metrics")
        if st.session_state.system_health and "metrics" in st.session_state.system_health:
            metrics = st.session_state.system_health["metrics"]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Queries", metrics['query_count'])
            
            with col2:
                st.metric("Avg Query Time", f"{metrics['average_query_time']:.2f}s")
            
            with col3:
                st.metric("Cache Hit Rate", f"{metrics['cache_hit_rate']:.2%}")
            
            with col4:
                st.metric("Error Count", metrics['error_count'])
            
            # Reset metrics button
            if st.button("🔄 Reset Metrics"):
                st.session_state.rag_system.reset_metrics()
                # Clear health cache to get fresh metrics
                get_cached_system_health.clear()
                st.session_state.system_health = get_cached_system_health(st.session_state.rag_system)
                st.success("✅ Metrics reset successfully")
                st.rerun()
        
        # Export system info
        if st.button("📋 Export System Info"):
            try:
                # Export system information
                system_info = {
                    "system_health": st.session_state.system_health,
                    "conversation_history": st.session_state.conversation_history,
                    "current_settings": {
                        "retriever_type": st.session_state.current_retriever_type,
                        "memory_type": st.session_state.current_memory_type,
                        "use_conversation": st.session_state.use_conversation
                    }
                }
                
                st.download_button(
                    label="📥 Download System Info",
                    data=json.dumps(system_info, indent=2),
                    file_name="rag_system_info.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"❌ Error exporting system info: {str(e)}")
    else:
        st.info("💡 Configure your settings above, then click 'Initialize System' in the sidebar to start using the RAG system.")


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.markdown('<h1 class="main-header">🤖 Production RAG System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize system in sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        if st.button("🚀 Initialize System"):
            initialize_rag_system()
        
        # Only auto-initialize if we have applied custom settings and no system yet
        if st.session_state.rag_system is None and st.session_state.custom_settings:
            initialize_rag_system()
        
        # Display system health
        if st.session_state.rag_system:
            display_system_health()
    
    # Main content area
    if st.session_state.rag_system:
        # Create tabs for different features
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Query", "💭 Conversations", "📚 Documents", "⚙️ Settings"])
        
        with tab1:
            query_interface()
        
        with tab2:
            conversation_management()
        
        with tab3:
            document_management()
        
        with tab4:
            settings_page()
    else:
        # Show settings tab even before initialization
        st.info("👋 Welcome to the Production RAG System!")
        st.warning("⚠️ **Action Required:** Please configure your VectorDB path below and then click 'Initialize System' in the sidebar.")
        
        # Show settings interface before initialization
        settings_page()


if __name__ == "__main__":
    main()