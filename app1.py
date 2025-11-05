import streamlit as st
import uuid
import os
from datetime import datetime
import plotly.graph_objects as go
import config
#from rag_engine import RAGEngine
from rag_engine1 import RAGEngine
from database import MemoryDatabase
import utils
from dotenv import load_dotenv
load_dotenv()
# Page config
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT)
# Initialize session state
if "rag_engine1" not in st.session_state:
    st.session_state.rag_engine1 = RAGEngine()
    st.session_state.db = MemoryDatabase()
    st.session_state.current_thread = None
    st.session_state.show_sources = True
    st.session_state.show_suggestions = True
    #st.session_state.thread_id=str(uuid.uuid4())
    st.session_state.k_value = 4
# Sidebar
with st.sidebar:
    st.title(" RAG Control Panel")
    
    # Thread management
    st.subheader("📂 Conversations")
    
    if st.button("➕ New Thread", use_container_width=True):
        thread_id = str(uuid.uuid4())
        st.session_state.db.create_thread(thread_id, f"Chat {datetime.now().strftime('%m/%d %H:%M')}")
        st.session_state.current_thread = thread_id
        st.rerun()
      
    # List threads
    threads = st.session_state.db.get_all_threads()
    for thread in threads:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                f" {thread['name']} ({thread['document_count']} docs)",
                key=f"thread_{thread['thread_id']}",
                use_container_width=True
            ):
                st.session_state.current_thread = thread['thread_id']
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{thread['thread_id']}"):
                st.session_state.db.delete_thread(thread['thread_id'])
                if st.session_state.current_thread == thread['thread_id']:
                    st.session_state.current_thread = None
                st.rerun()
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    st.session_state.show_sources = st.checkbox("Show Sources", value=True)
    st.session_state.show_suggestions = st.checkbox("Query Suggestions", value=True)
    #show_stats = st.checkbox("Show Statistics", value=True)
    
    #Token tracking toggle
    track_tokens = st.checkbox("Track Token Usage", value=False)
    
    st.divider()
    
    # Current thread info
    if st.session_state.current_thread:
        st.subheader("📊 Thread Info")
        docs = st.session_state.db.get_thread_documents(st.session_state.current_thread)
        st.metric("Documents", len(docs))
        
        if track_tokens:
            stats = st.session_state.db.get_thread_stats(st.session_state.current_thread)
            st.metric("Total Tokens", f"{stats['total_input_tokens'] + stats['total_output_tokens']:,}")
            st.metric("Est. Cost", f"${stats['total_cost']:.4f}")  # Uncomment for cost tracking
# Main area
# if no thread exists, show thread create info
if not st.session_state.current_thread:
    st.title(config.PAGE_TITLE)
    st.info("👈 Create or select a thread to start chatting")
    st.markdown("Upload multiple files or add a web link to chat with your data 🔍")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📄 Multi-Document Support")
        st.write("Upload PDFs, DOCX, TXT, PPTX or web URLs")
    with col2:
        st.markdown("### 💾 Persistent Memory")
        st.write("Conversations saved across sessions")
    with col3:
        st.markdown("### 🔄 Checkpointing")
        st.write("Resume or rollback conversations")
    
if  st.session_state.current_thread:
    # Document upload
    st.subheader("📤 Upload Documents")
    
    tab1, tab2 = st.tabs(["📁 File Upload", "🔗 URL"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "docx", "pptx", "py","csv"],
            accept_multiple_files=True,
            key="file_uploader"
        )
               
        if uploaded_files:
            if st.button("Process Files", type="primary"):
                with st.spinner("Processing documents...",show_time=True):
                    for file in uploaded_files:
                        # Save temp file
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        # Load and process
                        file_type = utils.get_file_extension(file.name)
                        docs = st.session_state.rag_engine1.load_documents(temp_path, file_type)
                        if not docs:
                            st.error(f"Couldn't extract text from {file.name}. Is it scanned PDF or unsupported format?")
                            continue
                        chunk_count = st.session_state.rag_engine1.process_documents(
                            docs, st.session_state.current_thread
                        )
                        
                        # Save to DB
                        st.session_state.db.add_document(
                            st.session_state.current_thread,
                            file.name, file_type, chunk_count
                        )
                        
                        # Cleanup
                        os.remove(temp_path)
                        
                        st.success(f"✅ {file.name}: {chunk_count} chunks")
    
    with tab2:
        url = st.text_input("Enter URL")
        if url and st.button("Load URL", type="primary"):
            with st.spinner("Loading URL..."):
                try:
                    docs = st.session_state.rag_engine1.load_documents(url, "url")
                    chunk_count = st.session_state.rag_engine1.process_documents(
                        docs, st.session_state.current_thread
                    )
                    st.session_state.vectorstore = st.session_state.rag_engine1.vector_stores

                    st.session_state.db.add_document(
                        st.session_state.current_thread,
                        url, "url", chunk_count
                    )
                    st.success(f" Loaded: {chunk_count} chunks")
                except Exception as e:
                    st.error(f" Error: {str(e)}")
    
    st.divider()
    
    # Chat interface
    st.subheader("💬 Chat")
    
    # Display chat history
    messages = st.session_state.db.get_thread_messages(st.session_state.current_thread)
    
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant" and st.session_state.show_sources and msg.get("sources"):
                with st.expander("📚 Sources"):
                    for i, j in enumerate(msg["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(j.get("content", "")[:200] + "...")
                        st.caption(utils.format_source_metadata(j.get("metadata", {})))
    
    # Query input
    query = st.chat_input("Ask a question about your documents",max_chars=150)
    
    if query:
        # Display user message
        with st.chat_message("user",avatar='human'):
            st.markdown(query)
        
        # Save user message
        st.session_state.db.add_message(
            st.session_state.current_thread, "user", query
        )
        
        # Generate response with streaming
        with st.chat_message("assistant",avatar=None):
            response_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            full_response = ""
            sources = []
            tokens = {"input": 0, "output": 0}
            
            for event in st.session_state.rag_engine1.query(
                query, 
                st.session_state.current_thread,
                [{"role": m["role"], "content": m["content"]} for m in messages]
            ):
                if event["type"] == "answer":
                    full_response += event["content"]
                    response_placeholder.markdown(full_response)
                
                elif event["type"] == "complete":
                    full_response = event["answer"]
                    sources = event["sources"]
                    tokens["input"] = event["input_tokens"]
                    tokens["output"] = event["output_tokens"]
            
            response_placeholder.markdown(full_response)
            
            # Display sources
            if st.session_state.show_sources and sources:
                with sources_placeholder.expander("📚 Sources", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(src.get("content", "")[:200] + "...")
                        st.caption(utils.format_source_metadata(src.get("metadata", {})))
         
        # Save assistant message
        st.session_state.db.add_message(
            st.session_state.current_thread, 
            "assistant", 
            full_response,
            sources,
            tokens["output"]
        )
        
        # Log tokens
        cost = utils.calculate_cost(tokens["input"], tokens["output"])
        st.session_state.db.log_token_usage(
            st.session_state.current_thread,
            tokens["input"], tokens["output"], cost
        )
        
        # Query suggestions
        if st.session_state.show_suggestions:
            suggestions = utils.generate_query_suggestions(query, [m["content"] for m in messages])
            if suggestions:
                st.markdown("**💡 Suggested follow-ups:**")
                cols = st.columns(len(suggestions))
                for col, suggestion in zip(cols, suggestions):
                    if col.button(suggestion, key=f"sug_{hash(suggestion)}"):
                        st.rerun()
    
    #Thread actions
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_name = st.text_input("Rename thread", key="rename_input")
        if new_name and st.button("✏️ Rename"):
            st.session_state.db.rename_thread(st.session_state.current_thread, new_name)
            st.success("✅ Renamed!")
            st.rerun()
    
    with col2:
        if st.button("📜 View Checkpoints"):
            checkpoints = st.session_state.rag_engine1.get_checkpoint_history(
                st.session_state.current_thread
            )
            if checkpoints:
                for cp in checkpoints[:10]:  # Show last 5
                    st.caption(f"Checkpoint {cp['step']}: {cp['checkpoint_id'][:8]}...")
            else:
                st.info("No checkpoints yet")
    
    with col3:
        if st.button("🔄 Export Chat"):
            chat_export = "\n\n".join([
                f"{m['role'].upper()}: {m['content']}" 
                for m in messages
            ])
            st.download_button(
                "💾 Download",
                chat_export,
                file_name=f"chat_{st.session_state.current_thread[:8]}.txt",
                mime="text/plain"
            )
            
    
