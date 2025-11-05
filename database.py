import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import config

class MemoryDatabase:
    """SQLite database for persistent memory and thread management"""
    
    def __init__(self, db_path: str = config.DB_PATH):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Threads table
        c.execute('''CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TIMESTAMP,
            last_updated TIMESTAMP,
            document_count INTEGER DEFAULT 0
        )''')
        
        # Messages table
        c.execute('''CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            role TEXT,
            content TEXT,
            sources TEXT,
            token_count INTEGER,
            timestamp TIMESTAMP,
            FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
        )''')
        
        # Documents table
        c.execute('''CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            filename TEXT,
            file_type TEXT,
            chunk_count INTEGER,
            upload_time TIMESTAMP,
            FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
        )''')
        
        # Token usage table
        c.execute('''CREATE TABLE IF NOT EXISTS token_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_cost REAL,
            timestamp TIMESTAMP,
            FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
        )''')
        
        conn.commit()
        conn.close()
    
    def create_thread(self, thread_id: str, name: str = "New Conversation") -> str:
        """Create new conversation thread"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        now = datetime.now()
        c.execute('''INSERT OR REPLACE INTO threads 
                     (thread_id, name, created_at, last_updated)
                     VALUES (?, ?, ?, ?)''',
                  (thread_id, name, now, now))
        conn.commit()
        conn.close()
        return thread_id
    
    def get_all_threads(self) -> List[Dict]:
        """Get all conversation threads"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''SELECT thread_id, name, created_at, last_updated, document_count 
                     FROM threads ORDER BY last_updated DESC''')
        threads = [{"thread_id": r[0], "name": r[1], "created_at": r[2], 
                   "last_updated": r[3], "document_count": r[4]} 
                  for r in c.fetchall()]
        conn.close()
        return threads
    
    def add_message(self, thread_id: str, role: str, content: str, 
                    sources: Optional[List] = None, token_count: int = 0):
        """Add message to thread"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO messages 
                     (thread_id, role, content, sources, token_count, timestamp)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (thread_id, role, content, json.dumps(sources or []), 
                   token_count, datetime.now()))
        c.execute('''UPDATE threads SET last_updated = ? WHERE thread_id = ?''',
                  (datetime.now(), thread_id))
        conn.commit()
        conn.close()
    
    def get_thread_messages(self, thread_id: str) -> List[Dict]:
        """Get all messages in thread"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''SELECT role, content, sources, token_count, timestamp 
                     FROM messages WHERE thread_id = ? ORDER BY timestamp''',
                  (thread_id,))
        messages = [{"role": r[0], "content": r[1], "sources": json.loads(r[2]), 
                    "token_count": r[3], "timestamp": r[4]} 
                   for r in c.fetchall()]
        conn.close()
        return messages
    
    def add_document(self, thread_id: str, filename: str, 
                     file_type: str, chunk_count: int):
        """Add document to thread"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO documents 
                     (thread_id, filename, file_type, chunk_count, upload_time)
                     VALUES (?, ?, ?, ?, ?)''',
                  (thread_id, filename, file_type, chunk_count, datetime.now()))
        c.execute('''UPDATE threads SET document_count = document_count + 1 
                     WHERE thread_id = ?''', (thread_id,))
        conn.commit()
        conn.close()
    
    def get_thread_documents(self, thread_id: str) -> List[Dict]:
        """Get all documents in thread"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''SELECT filename, file_type, chunk_count, upload_time 
                     FROM documents WHERE thread_id = ? ORDER BY upload_time''',
                  (thread_id,))
        docs = [{"filename": r[0], "file_type": r[1], 
                "chunk_count": r[2], "upload_time": r[3]} 
               for r in c.fetchall()]
        conn.close()
        return docs
    
    def log_token_usage(self, thread_id: str, input_tokens: int, 
                       output_tokens: int, cost: float):
        """Log token usage and cost"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT INTO token_usage 
                     (thread_id, input_tokens, output_tokens, total_cost, timestamp)
                     VALUES (?, ?, ?, ?, ?)''',
                  (thread_id, input_tokens, output_tokens, cost, datetime.now()))
        conn.commit()
        conn.close()
    
    def get_thread_stats(self, thread_id: str) -> Dict:
        """Get thread statistics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''SELECT SUM(input_tokens), SUM(output_tokens), SUM(total_cost)
                     FROM token_usage WHERE thread_id = ?''', (thread_id,))
        stats = c.fetchone()
        conn.close()
        return {
            "total_input_tokens": stats[0] or 0,
            "total_output_tokens": stats[1] or 0,
            "total_cost": stats[2] or 0.0
        }
    
    def delete_thread(self, thread_id: str):
        """Delete thread and all associated data"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM messages WHERE thread_id = ?', (thread_id,))
        c.execute('DELETE FROM documents WHERE thread_id = ?', (thread_id,))
        c.execute('DELETE FROM token_usage WHERE thread_id = ?', (thread_id,))
        c.execute('DELETE FROM threads WHERE thread_id = ?', (thread_id,))
        conn.commit()
        conn.close()
    
    def rename_thread(self, thread_id: str, new_name: str):
        """Rename thread"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('UPDATE threads SET name = ? WHERE thread_id = ?', 
                 (new_name, thread_id))
        conn.commit()
        conn.close()