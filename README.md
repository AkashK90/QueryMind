# 🤖 Advanced RAG with LangGraph

Production-ready Retrieval-Augmented Generation system built with LangGraph, featuring persistent memory, checkpointing, and multi-document support.

## ✨ Features

### Core Capabilities
- 🔄 **LangGraph Workflow**: Stateful RAG pipeline with conditional logic
- 💾 **Persistent Memory**: SQLite-based conversation storage across sessions
- 📍 **Checkpointing**: Resume/rollback conversations to any state
- 📡 **Streaming**: Real-time token-by-token response streaming
- 🧵 **Thread Management**: Isolated conversations with independent vector stores

### Advanced Features
- 📚 **Multi-Document Chat**: Query across multiple uploaded documents
- 🌳 **Conversation Branching**: Checkpoint-based conversation trees
- 🎯 **Source Highlighting**: Visual source attribution in responses
- 💡 **Query Suggestions**: AI-powered follow-up question recommendations
- 📊 **Token Tracking**: Monitor usage and estimate costs (optional)

### Supported Formats
- PDF, DOCX, TXT, PPTX, Python files
- Web URLs (article scraping)
- Multiple files per conversation

##  Architecture

### LangGraph Workflow
```
User Query → Retrieve → Generate → [Refine] → Stream → Save
              ↓          ↓          ↓         ↓      ↓
        Vector DB    LLM API    History   User   SQLite
```

### Components
1. **Retrieve Node**: FAISS similarity search (top-K=4)
2. **Generate Node**: LLM answer generation with context
3. **Refine Node**: History-aware answer refinement (conditional)
4. **Checkpointer**: SQLite state persistence per thread
5. **Memory DB**: Conversation history and metadata storage

##  Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd rag_langgraph

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
# OPENAI_API_KEY=your_openai_key  # Uncomment for OpenAI
```

Modify `config.py` for customization:
- LLM provider (Groq/OpenAI)
- Model selection
- Chunk size/overlap
- Retrieval parameters

### Run Application

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

## 📖 Usage Guide

### 1. Create Thread
Click "➕ New Thread" in sidebar to start a conversation.

### 2. Upload Documents
- **File Upload**: PDF, DOCX, TXT, PPTX, Python
- **URL**: Paste article/webpage URL

System splits documents into chunks (1000 chars, 200 overlap) and embeds with HuggingFace `all-MiniLM-L6-v2`.

### 3. Ask Questions
Type questions in chat input. System:
- Retrieves top 4 relevant chunks
- Generates contextual answer
- Streams response in real-time
- Displays source citations

### 4. Advanced Features

**Query Suggestions**: Click suggested follow-ups

**View Checkpoints**: See conversation state history

**Rollback**: Restore previous conversation state

**Export Chat**: Download conversation as text file

**Multi-Doc**: Upload multiple files to same thread

### 5. Thread Management
- Switch threads from sidebar
- Rename threads with custom names
- Delete threads (removes all data)

## 🗄️ Database Schema

### Tables
- **threads**: Conversation metadata
- **messages**: Chat history with sources
- **documents**: Uploaded file tracking
- **token_usage**: API consumption logs
- **checkpoints** (LangGraph): State snapshots

## 🔧 Customization

### Switch to OpenAI

1. Uncomment in `requirements.txt`:
```python
langchain-openai==0.2.5
```

2. Update `config.py`:
```python
LLM_PROVIDER = "openai"
OPENAI_MODEL = "gpt-4-turbo-preview"
```

3. Uncomment in `rag_engine.py`:
```python
from langchain_openai import ChatOpenAI
# ... in __init__:
self.llm = ChatOpenAI(...)
```

### Enable Cost Tracking

Uncomment in `app.py`:
```python
track_tokens = st.checkbox("Track Token Usage", value=False)
# ... later:
cost = utils.calculate_cost(tokens["input"], tokens["output"])
st.session_state.db.log_token_usage(...)
st.metric("Est. Cost", f"${stats['total_cost']:.4f}")
```

### Adjust Retrieval

In `config.py`:
```python
RETRIEVAL_K = 6  # Retrieve more chunks
SIMILARITY_THRESHOLD = 0.8  # Higher threshold
```

### Custom Prompts

Edit prompts in `rag_engine.py`:
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Your custom system prompt..."),
    ("user", "Your custom user template...")
])
```

## 🔍 Workflow Diagram

See `langgraph_workflow.mermaid` for visual workflow representation.

## 📊 Token Management

### Tracking (Optional)
- Input tokens: Context + Query
- Output tokens: Generated answer
- Cost estimation per provider

### Optimization Tips
- Reduce `CHUNK_SIZE` for fewer tokens
- Lower `RETRIEVAL_K` to retrieve less context
- Use smaller models (e.g., Mixtral-7b)

## 🐛 Troubleshooting

### "No relevant documents found"
- Ensure documents are uploaded
- Check vector store initialization
- Verify document processing completed

### Streaming not working
- Check LLM API key validity
- Ensure proper async handling
- Verify network connectivity

### Database locked
- Close other connections to SQLite
- Check file permissions
- Restart application

## 📝 Code Structure

```
rag_langgraph/
├── app.py                      # Streamlit UI
├── rag_engine.py              # LangGraph workflow
├── database.py                # SQLite memory management
├── utils.py                   # Helper functions
├── config.py                  # Configuration
├── requirements.txt           # Dependencies
├── langgraph_workflow.mermaid # Workflow diagram
├── .env                       # API keys (create this)
├── rag_memory.db             # Auto-generated database
└── checkpoints.db            # Auto-generated checkpoints
```

## 🚧 Future Enhancements

- [ ] Multi-modal support (images, audio)
- [ ] Advanced retrieval (HyDE, query expansion)
- [ ] Custom embedding models
- [ ] API endpoint (FastAPI)
- [ ] User authentication
- [ ] Cloud deployment configs

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please open issues/PRs.

## 📧 Contact

For questions or support, open an issue on GitHub.