import os
from dotenv import load_dotenv
load_dotenv()

os.environ["USER_AGENT"] = "RAG-Chatbot/1.0" 
# LLM Configuration
LLM_PROVIDER = "groq"
GROQ_MODEL = "openai/gpt-oss-120b" 
LLM_TEMPERATURE = 0.5
MAX_TOKENS = 1024
# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 730
# Text Splitting
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
# Retrieval
RETRIEVAL_K = 4
SIMILARITY_THRESHOLD = 0.75
# Database
DB_PATH = "rag_memory.db"
CHECKPOINT_PATH = "checkpoints.db"
# Token Pricing (USD per 1M tokens)
GROQ_PRICING = {"input": 0.27, "output": 0.27}  
# UI Configuration
PAGE_TITLE = " Chat with your Docs"
PAGE_ICON = "💬"
LAYOUT = "wide"