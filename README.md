# 🤖 Advanced RAG with LangGraph

Production-ready Retrieval-Augmented Generation system built with LangGraph, featuring persistent memory, checkpointing, and multi-document support.

## ✨ Features

### Core Capabilities
-  **LangGraph Workflow**: Stateful RAG pipeline with conditional logic
-  **Persistent Memory**: SQLite-based conversation storage across sessions
-  **Checkpointing**: Resume/rollback conversations to any state
-  **Streaming**: Real-time token-by-token response streaming
-  **Thread Management**: Isolated conversations with independent vector stores

### Advanced Features
-  **Multi-Document Chat**: Query across multiple uploaded documents
-  **Conversation Branching**: Checkpoint-based conversation trees
-  **Source Highlighting**: Visual source attribution in responses
-  **Query Suggestions**: AI-powered follow-up question recommendations
-  **Token Tracking**: Monitor usage and estimate costs (optional)

### Supported Formats
- PDF, DOCX, TXT, PPTX, Python files
- Web URLs (article scraping)
- Multiple files per conversation

