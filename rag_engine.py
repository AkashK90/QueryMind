from typing import TypedDict, List, Annotated, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI  # Uncomment for OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader,UnstructuredPDFLoader,
    WebBaseLoader,CSVLoader,PDFPlumberLoader,DirectoryLoader,PlaywrightURLLoader,PythonLoader
)
import fitz # for pymupdf (use when pdf has more layout or table)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()
import uuid
import operator
import config
import utils
import requests
import sqlite3


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
                groq_api_key=config.GROQ_API_KEY
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
        self.checkpointer = SqliteSaver.from_conn_string(config.CHECKPOINT_PATH)
        conn = sqlite3.connect(config.CHECKPOINT_PATH, check_same_thread=False)
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
    
    def load_documents(self, file_path: str, file_type: str) -> List[Document]:
        """Load documents based on file type"""
        loaders = {
            "pdf":  UnstructuredPDFLoader, #	PyMuPDF (fitz)	PyPDF see the comparision
            "txt":  TextLoader,
            "py":   PythonLoader,
            "docx": Docx2txtLoader,
            "pptx": UnstructuredPowerPointLoader
        }
        
        if file_type == "url":
            loader = WebBaseLoader(file_path)          
        else:
            loader_class = loaders.get(file_type, TextLoader) 
            loader = loader_class(file_path)
        
        return loader.load()
    
    '''def _load_pdf_with_pymupdf(self, file_path: str) -> List[Document]: # special for pdfs
        """Extract PDF text using PyMuPDF instead of PyPDFLoader"""
        docs = []
        pdf = fitz.open(file_path)

        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text")

            if text.strip():  # avoid storing empty pages
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_num
                        }
                    )
                )

        pdf.close()
        return docs'''
    
    def process_documents(self, documents: List[Document], thread_id: str):
        """Process and store documents in vector store"""
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        # ADD THIS VALIDATION:
        if not splits:
            raise ValueError("No text chunks generated from documents")
        # Create or update vector store for thread
        if thread_id in self.vector_stores:
            # Add to existing store
            self.vector_stores[thread_id].add_documents(splits)
        else:
            # Create new store
            self.vector_stores[thread_id] = FAISS.from_documents(
                splits, self.embeddings
            )
        
        return len(splits)
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents"""
        thread_id = state["thread_id"]
        query = state["query"]
        
        if thread_id not in self.vector_stores:
            state["documents"] = []
            state["context"] = ""
            return state
        
        # Retrieve documents
        retriever = self.vector_stores[thread_id].as_retriever(
            search_kwargs={"k": config.RETRIEVAL_K}
        )
        # docs = retriever.get_relevant_documents(query)
        docs = retriever.invoke(query)
        
        # Filter by similarity threshold (optional)
        # docs = [d for d in docs if d.metadata.get('score', 1.0) > config.SIMILARITY_THRESHOLD]
        
        state["documents"] = docs
        state["context"] = "\n\n".join([d.page_content for d in docs])
        state["sources"] = [
            {"content": d.page_content[:200], "metadata": d.metadata}
            for d in docs
        ]
        
        return state
    
    def _generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using LLM"""
        if not state["context"]:
            state["answer"] = "I couldn't find relevant information in the documents to answer your question."
            state["input_tokens"] = 0
            state["output_tokens"] = 0
            return state
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Answer the question based on the provided context.
            If the context doesn't contain enough information, say so clearly and try to know information on internet.
            Always cite sources by mentioning specific details from the context."""),
            ("user", """Context:{context}
                        Question: {question}
                    Answer:""")
        ])
        parser=StrOutputParser()
        # Generate answer
        chain = prompt | self.llm | parser
        response = chain.invoke({
            "context": state["context"],
            "question": state["query"]
        })
        
        # state["answer"] = response.content
        state["answer"] = getattr(response, "content", response)
        # state["answer"] = response.content if hasattr(response, "content") else response
        
        # Track tokens (approximate)
        state["input_tokens"] = utils.count_tokens(state["context"] + state["query"])
        state["output_tokens"] = utils.count_tokens(state["answer"])
        
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
        
        chain = prompt | self.llm 
        response = chain.invoke({
            "history": history_context,
            "answer": state["answer"]
        })
        
        state["answer"] = response.content
        state["output_tokens"] += utils.count_tokens(response.content)
        
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
        for event in self.workflow.stream(initial_state, config_dict):
            if "generate" in event:
                
                #  ONLY stream the new token, not full answer
                if "delta" in state:
                    yield {"type": "token", "content": state["delta"]}
                # Stream answer token by token
                state = event["generate"]
                if "answer" in state:
                    yield {"type": "answer", "content": state["answer"]}
            elif "refine" in event:
                state = event["refine"]
                if "answer" in state:
                    yield {"type": "answer", "content": state["answer"]}
        
        # Final state
        final_state = self.workflow.get_state(config_dict)
        if final_state:
            yield {
                "type": "complete",
                "answer": final_state.values.get("answer", ""),
                "sources": final_state.values.get("sources", []),
                "input_tokens": final_state.values.get("input_tokens", 0),
                "output_tokens": final_state.values.get("output_tokens", 0)
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