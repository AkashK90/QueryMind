import re
from typing import List, Dict
import tiktoken
import config

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    #Count tokens in text using tiktoken
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback: approximate 4 chars per token
        # return len(text) //4
        return int(len(text.split()) * 1.3)

def calculate_cost(input_tokens: int, output_tokens: int, 
                   provider: str = config.LLM_PROVIDER) -> float:
    "Calculate API cost based on token usage"
    pricing = config.GROQ_PRICING if provider == "groq" else {}
    # pricing = config.OPENAI_PRICING if provider == "openai" else {}
    
    input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0)
    output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)
    return input_cost + output_cost

def highlight_sources(text: str, sources: List[str]) -> str:
    "Add HTML highlighting for source references"
    for i, source in enumerate(sources, 1):
        pattern = re.escape(source[:50])  # Match first 50 chars
        text = re.sub(pattern, f'<mark>📄 Source {i}</mark>', text, flags=re.IGNORECASE)
    return text

def format_source_metadata(metadata: Dict) -> str:
    "Format source metadata for display"
    parts = []
    if 'source' in metadata:
        parts.append(f"📄 {metadata['source']}")
    if 'page' in metadata:
        parts.append(f"Page {metadata['page']}")
    if 'chunk' in metadata:
        parts.append(f"Chunk {metadata['chunk']}")
    return " | ".join(parts) if parts else "Unknown Source"

def generate_query_suggestions(query: str, history: List[str]) -> List[str]:
    "Generate follow-up query suggestions"
    suggestions = []
    
    # Question-based suggestions
    if "what" in query.lower():
        suggestions.append("Can you elaborate on that?")
        suggestions.append("What are the key points?")
    elif "how" in query.lower():
        suggestions.append("Can you provide step-by-step details?")
        suggestions.append("Are there any alternatives?")
    elif "why" in query.lower():
        suggestions.append("What are the underlying reasons?")
        suggestions.append("Are there any examples?")
    elif "when" in query.lower():
        suggestions.append("When this  reasons happened ?")
        suggestions.append("find relevant avalable date or month in any examples?")
    
    # Generic suggestions
    suggestions.extend([
        "Summarize the main findings",
        "Compare different approaches mentioned",
        "What are the limitations?",
        "Provide more specific examples",
        "give result by relevant maching"
    ])
    
    return suggestions[:3]  # Return top 3

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    "Extract keywords from text"
    # Simple keyword extraction (can be enhanced with NLP)
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    word_freq = {}
    for word in words:
        if word not in ['what', 'this', 'that', 'with', 'from', 'have']:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]

def chunk_text_with_metadata(text: str, chunk_size: int = config.CHUNK_SIZE,
                              overlap: int = config.CHUNK_OVERLAP,
                              source: str = "") -> List[Dict]:
    "Split text into chunks with metadata"
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')
            break_point = max(last_period, last_newline)
            if break_point > chunk_size * 0.5:  # At least 50% into chunk
                end = start + break_point + 1
                chunk_text = text[start:end]
        
        chunks.append({
            "text": chunk_text.strip(),
            "metadata": {
                "source": source,
                "chunk": chunk_id,
                "start": start,
                "end": end
            }
        })
        
        start = end - overlap
        chunk_id += 1
    
    return chunks

def format_chat_message(role: str, content: str, sources: List = None) -> Dict:
    "Format message for chat display"
    msg = {"role": role, "content": content}
    if sources:
        msg["sources"] = sources
    return msg

def sanitize_filename(filename: str) -> str:
    "Sanitize filename for safe storage"
    return re.sub(r'[^\w\s.-]', '', filename)[:100]

def get_file_extension(filename: str) -> str:
    """Get file extension"""
    return filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''