"""
Main RAG System Application
Streamlit web interface for the RAG system
"""

import streamlit as st
import asyncio
import os
from pathlib import Path
import time

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent / 'src'))

from src.retrieval.rag_pipeline import RAGPipeline
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logging

# Page configuration
st.set_page_config(
    page_title="RAG System with Gemini",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🤖 RAG System with Gemini API")
    st.markdown("*Retrieval-Augmented Generation powered by Google's Gemini*")
    
    # Initialize session state
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        # Model selection
        model_choice = st.selectbox(
            "Gemini Model",
            ["gemini-2.5-pro", "gemini-1.5-flash"],
            help="Choose your Gemini model"
        )
        
        # Initialize RAG Pipeline
        if st.button("🚀 Initialize RAG System"):
            if not api_key:
                st.error("Please enter your Gemini API key")
                return
            
            # Set API key in environment
            os.environ['GEMINI_API_KEY'] = api_key
            
            try:
                with st.spinner("Initializing RAG pipeline..."):
                    config = ConfigLoader.load_config()
                    config['gemini']['model_name'] = model_choice
                    st.session_state.rag_pipeline = RAGPipeline(config)
                st.success("✅ RAG System initialized successfully!")
            except Exception as e:
                st.error(f"❌ Initialization failed: {e}")
        
        # System stats
        if st.session_state.rag_pipeline:
            st.markdown("---")
            st.header("📊 System Status")
            
            try:
                stats = st.session_state.rag_pipeline.get_system_stats()
                
                # Vector DB stats
                db_stats = stats.get('vector_db_stats', {})
                st.metric("Documents", db_stats.get('document_count', 0))
                
                # Gemini usage
                usage = stats.get('gemini_usage', {})
                st.metric("Daily Usage", f"{usage.get('daily_usage', 0)}/{usage.get('daily_limit', 100)}")
                st.metric("Remaining RPM", usage.get('remaining_rpm', 0))
                
            except Exception as e:
                st.error(f"Error getting stats: {e}")
    
    # Main content area
    if not st.session_state.rag_pipeline:
        st.info("👈 Please configure and initialize the RAG system using the sidebar")
        
        # Show configuration example
        st.markdown("## 🔧 Setup Instructions")
        st.markdown("""
        1. **Get your Gemini API key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. **Enter your API key** in the sidebar
        3. **Choose your model**:
           - `gemini-2.5-pro`: Higher quality, 5 RPM limit
           - `gemini-1.5-flash`: Faster, 15 RPM limit
        4. **Click Initialize** to start the RAG system
        """)
        
        # Show system architecture
        st.markdown("## 🏗️ System Architecture")
        st.markdown("""
        - **Gemini API**: Embeddings & text generation
        - **ChromaDB**: Vector storage & similarity search  
        - **Document Processing**: PDF, DOCX, Markdown, TXT support
        - **Smart Chunking**: Semantic and adaptive strategies
        - **Rate Limiting**: Intelligent API quota management
        - **Caching**: Response and embedding caching
        """)
        
        return
    
    # Document Management Tab
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Management", "💬 Chat", "📊 Analytics", "🗃️ Vector DB Viewer"])
    
    with tab1:
        st.header("📄 Document Management")
        
        # File upload
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload documents",
                accept_multiple_files=True,
                type=['pdf', 'txt', 'md', 'docx'],
                help="Upload PDF, TXT, Markdown, or DOCX files"
            )
        
        with col2:
            chunking_method = st.selectbox(
                "Chunking Method",
                ["semantic", "paragraph", "fixed", "adaptive"],
                help="Choose how to split documents into chunks"
            )
        
        if uploaded_files and st.button("📤 Process Documents"):
            process_uploaded_files(uploaded_files, chunking_method)
        
        # Directory ingestion
        st.markdown("---")
        st.subheader("📁 Directory Ingestion")
        
        directory_path = st.text_input(
            "Directory Path",
            help="Enter path to directory containing documents"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            recursive = st.checkbox("Include subdirectories", value=True)
        with col2:
            dir_chunking = st.selectbox(
                "Chunking Method",
                ["semantic", "paragraph", "fixed", "adaptive"],
                key="dir_chunking"
            )
        
        if directory_path and st.button("📂 Process Directory"):
            process_directory(directory_path, recursive, dir_chunking)
        
        # Add text directly
        st.markdown("---")
        st.subheader("✍️ Add Text Directly")
        
        direct_text = st.text_area(
            "Enter text directly",
            height=150,
            help="Enter text content to add to the knowledge base"
        )
        
        if direct_text and st.button("➕ Add Text"):
            add_direct_text(direct_text)
    
    with tab2:
        st.header("💬 Chat with your Documents")
        
        # Prompt type selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            question = st.text_input(
                "Ask a question",
                placeholder="What would you like to know?",
                key="question_input"
            )
        
        with col2:
            prompt_type = st.selectbox(
                "Response Style",
                ["basic", "advanced", "conversational", "technical", "comparative", "summary"],
                help="Choose the response style"
            )
        
        if question and st.button("🔍 Ask"):
            answer_question(question, prompt_type)
        
        # Chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("💭 Chat History")
            
            for i, (q, a, timestamp) in enumerate(reversed(st.session_state.chat_history[-10:])):
                with st.expander(f"Q: {q[:50]}... ({timestamp})"):
                    st.markdown(f"**Question:** {q}")
                    st.markdown(f"**Answer:** {a}")
            
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
    
    with tab3:
        st.header("📊 System Analytics")
        
        if st.button("🔄 Refresh Stats"):
            show_analytics()
        
        show_analytics()

    with tab4:
        st.header("🗃️ Vector DB Viewer")
        show_vector_db_viewer()

def process_uploaded_files(uploaded_files, chunking_method):
    """Process uploaded files"""
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    file_paths = []
    
    # Save uploaded files temporarily
    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        file_paths.append(str(file_path))
    
    # Process files
    with st.spinner(f"Processing {len(file_paths)} files..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                st.session_state.rag_pipeline.ingest_documents(file_paths, chunking_method)
            )
            
            if result['status'] == 'success':
                st.success(f"✅ Successfully processed {result['chunks_processed']} chunks from {len(result['successful_files'])} files")
                
                if result['failed_files']:
                    st.warning(f"⚠️ Failed to process {len(result['failed_files'])} files")
                    for failed in result['failed_files']:
                        st.error(f"- {failed['file']}: {failed['error']}")
            else:
                st.error(f"❌ Processing failed: {result['message']}")
                
        except Exception as e:
            st.error(f"❌ Processing error: {e}")
        finally:
            loop.close()
    
    # Cleanup temp files
    for file_path in file_paths:
        try:
            os.unlink(file_path)
        except:
            pass

def process_directory(directory_path, recursive, chunking_method):
    """Process directory of documents"""
    if not Path(directory_path).exists():
        st.error("❌ Directory not found")
        return
    
    with st.spinner("Processing directory..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                st.session_state.rag_pipeline.ingest_directory(
                    directory_path, recursive, chunking_method
                )
            )
            
            if result['status'] == 'success':
                st.success(f"✅ Successfully processed {result['chunks_processed']} chunks")
            else:
                st.error(f"❌ Processing failed: {result['message']}")
                
        except Exception as e:
            st.error(f"❌ Processing error: {e}")
        finally:
            loop.close()

def add_direct_text(text):
    """Add text directly to the knowledge base"""
    with st.spinner("Adding text to knowledge base..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            success = loop.run_until_complete(
                st.session_state.rag_pipeline.add_text(
                    text, 
                    metadata={"source": "direct_input", "timestamp": str(time.time())}
                )
            )
            
            if success:
                st.success("✅ Text added successfully")
            else:
                st.error("❌ Failed to add text")
                
        except Exception as e:
            st.error(f"❌ Error adding text: {e}")
        finally:
            loop.close()

def answer_question(question, prompt_type):
    """Answer a question using the RAG system"""
    with st.spinner("Thinking..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Get chat history for conversational mode
            chat_history = ""
            if prompt_type == "conversational" and st.session_state.chat_history:
                recent_history = st.session_state.chat_history[-3:]  # Last 3 exchanges
                history_parts = []
                for q, a, _ in recent_history:
                    history_parts.append(f"Human: {q}")
                    history_parts.append(f"Assistant: {a}")
                chat_history = "\n".join(history_parts)
            
            result = loop.run_until_complete(
                st.session_state.rag_pipeline.query(
                    question, 
                    prompt_type,
                    chat_history=chat_history
                )
            )
            
            # Display answer
            st.markdown("### 🤖 Answer")
            st.markdown(result.answer)
            
            # Display sources
            if result.sources:
                st.markdown("### 📚 Sources")
                for i, (source, metadata, similarity) in enumerate(zip(result.sources, result.metadata, result.similarities)):
                    with st.expander(f"Source {i+1} - {metadata.get('file_name', 'Unknown')} (Similarity: {similarity:.2f})"):
                        st.text(source[:500] + "..." if len(source) > 500 else source)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processing Time", f"{result.processing_time:.2f}s")
            with col2:
                st.metric("Sources Found", len(result.sources))
            with col3:
                st.metric("Tokens Used", result.tokens_used)
            
            # Add to chat history
            timestamp = time.strftime("%H:%M:%S")
            st.session_state.chat_history.append((question, result.answer, timestamp))
            
        except Exception as e:
            st.error(f"❌ Query failed: {e}")
        finally:
            loop.close()

def show_analytics():
    """Show system analytics"""
    try:
        stats = st.session_state.rag_pipeline.get_system_stats()
        
        # Vector DB Statistics
        st.subheader("🗃️ Vector Database")
        db_stats = stats.get('vector_db_stats', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", db_stats.get('document_count', 0))
        with col2:
            st.metric("Collection", db_stats.get('collection_name', 'N/A'))
        with col3:
            st.metric("Status", "✅ Active" if db_stats.get('document_count', 0) > 0 else "⏳ Empty")
        
        # Gemini Usage
        st.subheader("🤖 Gemini API Usage")
        usage = stats.get('gemini_usage', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Daily Usage", f"{usage.get('daily_usage', 0)}/{usage.get('daily_limit', 100)}")
        with col2:
            st.metric("RPM Usage", f"{usage.get('rpm_usage', 0)}/{usage.get('rpm_limit', 5)}")
        with col3:
            st.metric("Remaining Daily", usage.get('remaining_daily', 0))
        with col4:
            st.metric("Remaining RPM", usage.get('remaining_rpm', 0))
        
        # Processing Configuration
        st.subheader("⚙️ Processing Configuration")
        processing = stats.get('processing_stats', {})
        retrieval = stats.get('retrieval_config', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.json({
                "Chunk Size": processing.get('chunk_size', 1000),
                "Chunk Overlap": processing.get('chunk_overlap', 200),
                "Supported Formats": processing.get('supported_formats', [])
            })
        with col2:
            st.json({
                "Top K Results": retrieval.get('top_k', 5),
                "Similarity Threshold": retrieval.get('similarity_threshold', 0.7),
                "Re-rank Results": retrieval.get('rerank_results', True)
            })
        
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

def show_vector_db_viewer():
    """Interactive viewer for ChromaDB contents"""
    try:
        # Lazy import to avoid circulars
        from src.vectordb.chroma_client import ChromaDBClient

        client = ChromaDBClient()
        collection = client.ensure_collection()

        # Header metrics
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Collection", collection.name)
        with colB:
            st.metric("Total Items", collection.count())
        with colC:
            st.metric("Persist Dir", client.get_persist_directory())

        st.markdown("---")

        # Controls
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            page_size = st.selectbox("Page Size", [10, 25, 50, 100], index=2)
        with col2:
            page = st.number_input("Page", min_value=1, value=1, step=1)
        with col3:
            metadata_key = st.text_input("Filter: metadata key (optional)")
            metadata_value = st.text_input("Filter: metadata value (optional)")

        where = None
        if metadata_key and metadata_value:
            # Simple equality filter
            where = {metadata_key: metadata_value}

        offset = (page - 1) * page_size

        # Fetch items
        data = client.get_documents(limit=page_size, offset=offset, where=where)

        ids = data.get("ids", []) or []
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []

        if not ids:
            st.info("No items found for this page/filter.")
            return

        # Table-like display
        for idx, (vid, doc, meta) in enumerate(zip(ids, docs, metas)):
            with st.expander(f"ID: {vid}"):
                colx, coly = st.columns([3,2])
                with colx:
                    st.markdown("**Document (first 500 chars):**")
                    preview = (doc or "")
                    st.text(preview[:500] + ("..." if len(preview) > 500 else ""))
                with coly:
                    st.markdown("**Metadata:**")
                    st.json(meta or {})
                # Delete button per item
                if st.button("🗑️ Delete", key=f"del_{vid}"):
                    try:
                        collection.delete(ids=[vid])
                        st.success(f"Deleted {vid}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

        # Bulk operations
        st.markdown("---")
        with st.expander("Bulk Operations"):
            ids_text = st.text_area("Enter IDs to delete (one per line)")
            if st.button("Delete IDs"):
                del_ids = [i.strip() for i in ids_text.splitlines() if i.strip()]
                if del_ids:
                    try:
                        collection.delete(ids=del_ids)
                        st.success(f"Deleted {len(del_ids)} items")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Bulk delete failed: {e}")
                else:
                    st.info("No IDs provided.")
    except Exception as e:
        st.error(f"Error loading vector DB: {e}")

if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level="INFO")
    
    main()