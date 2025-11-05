
from typing import TypedDict, List, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI  # Uncomment for OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader,UnstructuredPDFLoader,
    WebBaseLoader,CSVLoader,PDFPlumberLoader,DirectoryLoader,PlaywrightURLLoader,PythonLoader,PyMuPDFLoader,SeleniumURLLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
import operator
import config
import utils
import requests
import sqlite3
import fitz
import os
from dotenv import load_dotenv
load_dotenv()
class RAGState(TypedDict):
    """State for RAG workflow"""
    query: str
    documents: List[Document]
    context: str
    answer: str
    sources: List[dict]
    thread_id: str
    history: Annotated[List[dict], operator.add]
    input_tokens: int
    output_tokens: int
    
class RAGEngine:
    """Advanced RAG Engine with LangGraph"""
    
    def __init__(self):
        # Initialize LLM
        if config.LLM_PROVIDER == "groq":
            self.llm = ChatGroq(
                model=config.GROQ_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                groq_api_key=config.GROQ_API_KEY,
                streaming=True
            )
        # else:  # OpenAI
        #     self.llm = ChatOpenAI(
        #         model=config.OPENAI_MODEL,
        #         temperature=config.LLM_TEMPERATURE,
        #         max_tokens=config.MAX_TOKENS,
        #         openai_api_key=config.OPENAI_API_KEY
        #     )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Vector stores per thread
        self.vector_stores = {}
        
        # Checkpointer for persistence
       # self.checkpointer = SqliteSaver.from_conn_string(config.CHECKPOINT_PATH)  #  don't override
        conn = sqlite3.connect(config.CHECKPOINT_PATH, check_same_thread=False)
        # for making big size data because datastoring constraint coming
        conn.execute("PRAGMA max_page_count = 2147483646")  # ~2TB limit
        conn.execute("PRAGMA page_size = 32768")  # Increase page size
        self.checkpointer = SqliteSaver(conn)        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(RAGState)  
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("generate", self._generate_answer)
        workflow.add_node("refine", self._refine_answer)
        
        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_conditional_edges(
            "generate",
            self._should_refine,
            {
                "refine": "refine",
                "end": END
            }
        )
        workflow.add_edge("refine", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    '''def load_documents(self, file_path: str, file_type: str) -> List[Document]:
        """Load documents based on file type"""
        loaders = {
            "pdf":  PyPDFLoader,
            "txt":  TextLoader,
            "py":   PythonLoader,
            "docx": Docx2txtLoader,
            "pptx": UnstructuredPowerPointLoader
        }
        
        if file_type == "url":
            loader = WebBaseLoader(file_path)
            
        # if file_type == "pdf":
        #     loader= PyPDFLoader(file_path)
        else:
            loader_class = loaders.get(file_type, TextLoader)
            loader = loader_class(file_path)
        
        return loader.load()
    '''
    
    def load_documents(self,file_path: str, file_type: str) -> List[Document]:
        try:
        # Auto-detect extension if not provided
            # if not file_type:
            #     if file_path.startswith("http://") or file_path.startswith("https://"):
            #         file_type = "url"
            #     else:
            #         file_type = os.path.splitext(file_path)[1].lower().lstrip(".") or "txt"

        
        # loaders = {
        #         "pdf": UnstructuredPDFLoader, #PDFPlumberLoader,                 # Extracts text from PDF using pdfplumber (more accurate than PyPDF)
        #         "txt": TextLoader,                       # Loads plain text files (.txt)
        #         "py": PythonLoader,                      # Loads Python source code files (.py)
        #         "docx": Docx2txtLoader,                  # Extracts text from Word documents (.docx)
        #         "pptx": UnstructuredPowerPointLoader,    # Extracts text from PowerPoint slides (.pptx)
        #         "csv": CSVLoader                         # Loads CSV files as tabular documents
        #     }
               
# ---------------------- URL Loader ----------------------
            if file_type == "url":
                print("Loading URL using WebBaseLoader...")
                loader = WebBaseLoader([file_path])  # SeleniumURLLoader  for javascript
                return loader.load()

        # ---------------------- PDF Loader (PyMuPDF) ----------------------
            elif file_type == "pdf":
                #print("Loading PDF using PyMuPDF...")
                print("Loading PDF using pymuPDFLoader...")
               # loader = UnstructuredPDFLoader(file_path, mode="single",extract_images=False)
                loader = PyMuPDFLoader(file_path,extract_images=False) #, mode="single",extract_images=False)
                return loader.load()
            
            # if file_type == "pdf":
            #     print("Trying PyMuPDFLoader first...")
            #     try:
            #         loader=PyMuPDFLoader(file_path)
            #         return loader.load()
            #     except Exception:
            #             print("PyMuPDF couldn't extract. Falling back to UnstructuredPDFLoader...")
            #             loader= UnstructuredPDFLoader(file_path, strategy="ocr_only",extract_images=False )  # when image then true
            #             return loader.load()
        # ---------------------- TXT Loader ----------------------
            elif file_type == "txt":
                print("Loading TXT using TextLoader...")
                loader = TextLoader(file_path,autodetect_encoding=True)
                return loader.load()
            # ---------------------- PY Loader ----------------------
            elif file_type == "py":
                print("Loading Python file...")
                loader = PythonLoader(file_path)
                return loader.load()
        # ---------------------- DOCX Loader ----------------------
            elif file_type == "docx":
                print("Loading DOCX using Docx2txtLoader...")
                loader = Docx2txtLoader(file_path)
                return loader.load()
         #-----------------csv loader -------------------------------   
            elif file_type == "csv":
                print("Loading csv using CSVLoader...")
                loader = CSVLoader(file_path)
                return loader.load()
            # ---------------------- PPTX Loader ----------------------
            elif file_type == "pptx":
                print("Loading PPTX using UnstructuredPowerPointLoader...")
                loader = UnstructuredPowerPointLoader(file_path)
                return loader.load()

        # ---------------------- Default Loader ----------------------
            else:
                print("Unknown file type — using TextLoader as fallback...")
                loader = TextLoader(file_path)
                return loader.load()
        except Exception as e:
        # Don't raise here — return empty list and log so your app can continue gracefully.
            print(f"Error loading document {file_path} (type={file_type}): {e}")
        return []
       
    def process_documents(self, documents: List[Document], thread_id: str):
        """Process and store documents in vector store"""
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        # add this validation:
        if not splits:
            raise ValueError("No text chunks generated from documents")
        for chunk in splits:
            chunk.metadata["thread_id"]=thread_id
        # Create or update vector store for thread
        if thread_id in self.vector_stores:
            # Add to existing store
            print(f"[FAISS] Creating update vector store: {thread_id}")
            self.vector_stores[thread_id].add_documents(splits)
        else:
            # Create new store
             print(f"[FAISS] Creating new vector store: {thread_id}")
             self.vector_stores[thread_id] = FAISS.from_documents(
                splits, self.embeddings
            )
        self.vector_stores[thread_id].save_local(f"db/{thread_id}")
        print(f"[FAISS] Stored {len(splits)} chunks for thread {thread_id}")
        print(f"[FAISS] Total embeddings in store: {self.vector_stores[thread_id].index.ntotal}")
        return len(splits)
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents"""
        thread_id = state["thread_id"]
        query = state["query"]
        
        if thread_id not in self.vector_stores:
            state["documents"] = []
            state["context"] = ""
            state['sources']=[]
            return state
        
        # Retrieve documents
        
        retriever = self.vector_stores[thread_id].as_retriever(
            search_kwargs={"k": config.RETRIEVAL_K,
                           'fetch_k':10,
                           'lambda_mult':0.5},
            search_type='similarity'#'mmr'
            #search_type="similarity_score_threshold"
        )
        # docs = retriever.get_relevant_documents(query)
        docs = retriever.invoke(query)
        #docs= retriever.get_relevant_documents(query)
        # Filter by similarity threshold (optional)
        # docs = [d for d in docs if d.metadata.get('score', 1.0) > config.SIMILARITY_THRESHOLD]
        docs = [d for d in docs if d.metadata.get("thread_id") == thread_id]
        state["documents"] = docs
        state["context"] = "\n\n".join([d.page_content for d in docs])
        state["sources"] = [
            {"content": d.page_content[:200], "metadata": d.metadata}
            for d in docs
        ]
        
        return state
    # print("Embedding count:", self.vector_stores[thread_id].index.ntotal)
    def _generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using LLM"""
        thread_id = state.get("thread_id", "unknown")
        if not state["context"]:
            state["answer"] = "I couldn't find relevant information in the documents to answer your question."
            state["input_tokens"] = 0
            state["output_tokens"] = 0
            print(f"[GENERATE] Answer generated for thread {thread_id}")
            print(f"[TOKENS] Input: {state.get('input_tokens', 0)}, Output: {state.get('output_tokens', 0)}")
            return state
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Answer the question based on the provided context.
            If the context doesn't contain enough information, say so clearly.
            Always cite sources by mentioning specific details from the context."""),
            ("user", """Context:{context}
                        Question: {question}
                        Answer:""")
        ])
        parser= StrOutputParser()
        # Generate answer
        chain = prompt | self.llm | parser
        response = chain.invoke({
            "context": state["context"],
            "question": state["query"]
        })
        
        state["answer"] = response
        
        if hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('usage', {})
            state["input_tokens"] = usage.get('prompt_tokens', 0)
            state["output_tokens"] = usage.get('completion_tokens', 0)
        else:
        # Track tokens (approximate)
            state["input_tokens"] = utils.count_tokens(state["context"] + state["query"])
            state["output_tokens"] = utils.count_tokens(state['answer'])
        
        return state
    
    def _refine_answer(self, state: RAGState) -> RAGState:
        """Refine answer with additional context from history"""
        history_context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in state.get("history", [])[-3:]  # Last 3 messages
        ])
        
        if not history_context:
            return state
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Refine the answer considering conversation history. Keep it concise."),
            ("user", """Previous conversation:
            {history}
            Current answer: {answer}
            Refined answer:""")
            ])
        parser= StrOutputParser()
        chain = prompt | self.llm | parser
        # Stream the response
        response = ""
        for chunk in chain.stream({
        #
            "history": history_context,
            "answer": state["answer"]
        }):
            response += chunk
        state["answer"] = response
        state["output_tokens"] += utils.count_tokens(response)
        
        return state
    
    def _should_refine(self, state: RAGState) -> str:
        """Decide if answer should be refined"""
        # Refine if there's conversation history
        if len(state.get("history", [])) > 0:
            return "refine"
        return "end"
    
    def query(self, query: str, thread_id: str, history: List[dict] = None):
        """Query the RAG system with streaming"""
        initial_state = {
            "query": query,
            "documents": [],
            "context": "",
            "answer": "",
            "sources": [],
            "thread_id": thread_id,
            "history": history or [],
            "input_tokens": 0,
            "output_tokens": 0
        }
        
        config_dict = {"configurable": {"thread_id": thread_id}}
        
        # Stream the workflow
        for event in self.workflow.stream(initial_state, config=config_dict):
            if "generate" in event:
                # Stream answer token by token
                state = event["generate"]
                if "answer" in state:
                    yield {"type": "answer", "content": state["answer"]}
            elif "refine" in event:
                state = event["refine"]
                if "answer" in state:
                    yield {"type": "answer", "content": state["answer"]}
        
        # Final state
        # final_state = self.workflow.get_state(config_dict)
        # if final_state:
        #     yield {
        #         "type": "complete",
        #         "answer": final_state.values.get("answer", ""),
        #         "sources": final_state.values.get("sources", []),
        #         "input_tokens": final_state.values.get("input_tokens", 0),
        #         "output_tokens": final_state.values.get("output_tokens", 0)
        #     }
            
        final_state = self.workflow.get_state(config_dict)

        state_values = getattr(final_state, "values", {}) if final_state else {}

        yield {
            "type": "complete",
            "answer": state_values.get("answer", ""),
            "sources": state_values.get("sources", []),
            "input_tokens": state_values.get("input_tokens", 0),
            "output_tokens": state_values.get("output_tokens", 0)
        }

    def get_checkpoint_history(self, thread_id: str) -> List[dict]:
        """Get checkpoint history for thread"""
        config_dict = {"configurable": {"thread_id": thread_id}}
        checkpoints = list(self.checkpointer.list(config_dict))
        return [
            {
                "checkpoint_id": str(cp.config["configurable"]["checkpoint_id"]),
                "timestamp": cp.metadata.get("timestamp", ""),
                "step": cp.metadata.get("step", 0)
            }
            for cp in checkpoints
        ]
    
    def rollback_to_checkpoint(self, thread_id: str, checkpoint_id: str):
        """Rollback to specific checkpoint"""
        config_dict = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id
            }
        }
        return self.workflow.get_state(config_dict)
