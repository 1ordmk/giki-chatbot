"""
GIKI Chatbot Backend - Fixed Version
Run this on port 5005 to match frontend
"""

import os
import json
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_HOST = os.getenv('PINECONE_INDEX_HOST')
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'codellama:13b')


class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline"""

    def __init__(self):
        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
        
        # Initialize Pinecone
        self.index = None
        if PINECONE_API_KEY and PINECONE_INDEX_HOST:
            try:
                pc = Pinecone(api_key=PINECONE_API_KEY)
                self.index = pc.Index(
                    name="llama-text-embed-v2-index",
                    host=PINECONE_INDEX_HOST
                )
                logger.info("✓ Connected to Pinecone")
            except Exception as e:
                logger.error(f"✗ Pinecone connection failed: {e}")
        
        # Initialize LLM
        self.llm_client = self._init_llm()
        
        self.system_prompt = """You are a helpful assistant EXCLUSIVELY for Ghulam Ishaq Khan Institute (GIKI).

STRICT RULES:
- ONLY answer questions about GIKI - its admissions, academics, departments, societies, campus life, facilities, and history
- NEVER answer questions about: other universities, general knowledge, current events, politics, sports, entertainment, or any non-GIKI topics
- If asked about anything not related to GIKI, politely decline and redirect to GIKI topics
- Base ALL answers ONLY on the provided context
- If context is insufficient or irrelevant, say you don't have that information
- Be concise, friendly, and factual
- Use bullet points for lists when appropriate

Remember: You are a GIKI-specific assistant. Stay focused on GIKI only!"""

    def _init_llm(self):
        """Initialize LLM client"""
        try:
            # Try to import and use Ollama
            import ollama
            ollama.list()
            logger.info(f"✓ Using Ollama with model: {LLM_PROVIDER}")
            return ollama
        except Exception as e:
            logger.error(f"✗ LLM initialization failed: {e}")
            logger.warning("Install Ollama: https://ollama.ai/download")
            return None
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant context from Pinecone"""
        if not self.index:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode(query).tolist()
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return results['matches']
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def format_context(self, matches: List[Dict]) -> str:
        """Format retrieved contexts for LLM"""
        if not matches:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        
        for i, match in enumerate(matches, 1):
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            source = metadata.get('title', 'Unknown Source')
            
            context_parts.append(f"[Source {i}: {source}]\n{text}\n")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM"""
        if not self.llm_client:
            return "LLM not available. Please install Ollama and run: ollama pull codellama:13b"
        
        try:
            # Determine model name
            model_name = LLM_PROVIDER if ':' in LLM_PROVIDER else 'llama3.2'
            
            response = self.llm_client.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {query}"}
                ]
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error processing your request. Error: {str(e)}"
    
    def chat(self, query: str) -> Dict:
        """Main chat function"""
        try:
            # Retrieve relevant context
            matches = self.retrieve_context(query)
            
            # Check relevance threshold - if top match is below 0.5, it's not GIKI-related
            if not matches or (matches and matches[0]['score'] < 0.5):
                return {
                    'response': "I'm sorry, but I can only answer questions about GIKI (Ghulam Ishaq Khan Institute). I don't have information about that topic.\n\nI can help you with:\n• Admissions and entry requirements\n• Academic programs and departments\n• Campus life and student societies\n• Hostel facilities and fee structure\n• Contact information\n\nPlease ask me something about GIKI!",
                    'sources': []
                }
            
            # Format context
            context = self.format_context(matches)
            
            # Generate response
            response = self.generate_response(query, context)
            
            # Prepare sources
            sources = [
                {
                    'title': match['metadata'].get('title', 'Unknown'),
                    'url': match['metadata'].get('source_url', ''),
                    'relevance': round(match['score'], 3)
                }
                for match in matches[:3]
            ]
            
            return {
                'response': response,
                'sources': sources
            }
        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                'response': "I apologize, but I encountered an error. Please try again.",
                'sources': [],
                'error': str(e)
            }


class AuthManager:
    """Handle admin authentication"""
    
    def __init__(self):
        self.admins = {
            os.getenv('ADMIN_USERNAME', 'admin'): generate_password_hash(
                os.getenv('ADMIN_PASSWORD', 'admin123')
            )
        }
    
    def verify_credentials(self, username: str, password: str) -> bool:
        if username in self.admins:
            return check_password_hash(self.admins[username], password)
        return False
    
    def generate_token(self, username: str) -> str:
        payload = {
            'username': username,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[str]:
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            return payload['username']
        except:
            return None


class StudentDatabase:
    """Mock student database"""
    
    def __init__(self):
        self.students = self._load_mock_data()
    
    def _load_mock_data(self) -> Dict:
        mock_file = 'data/mock_students.json'
        if os.path.exists(mock_file):
            with open(mock_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_student_info(self, student_id: str) -> Optional[Dict]:
        return self.students.get(student_id)


# Initialize components
logger.info("Initializing RAG Pipeline...")
rag_pipeline = RAGPipeline()
auth_manager = AuthManager()
student_db = StudentDatabase()
logger.info("✓ Backend initialized successfully!")


# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'pinecone': 'connected' if rag_pipeline.index else 'disconnected',
        'llm': 'available' if rag_pipeline.llm_client else 'unavailable'
    }
    return jsonify(status)


@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Check for admin queries
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    is_admin = auth_manager.verify_token(token) if token else False
    
    if is_admin and ('student' in query.lower() or 'attendance' in query.lower()):
        student_id_match = re.search(r'\d{7}', query)
        
        if student_id_match:
            student_id = student_id_match.group()
            student_info = student_db.get_student_info(student_id)
            
            if student_info:
                response_text = f"""**Student Information for {student_id}:**

• **Name:** {student_info['name']}
• **Department:** {student_info['department']}
• **Semester:** {student_info['semester']}
• **GPA:** {student_info['gpa']}
• **Attendance:** {student_info['attendance']}%
• **Status:** {student_info['status']}"""
                
                return jsonify({
                    'response': response_text,
                    'sources': [],
                    'admin_query': True
                })
            else:
                return jsonify({
                    'response': f"Student ID {student_id} not found in database.",
                    'sources': []
                })
    
    # Regular RAG query
    result = rag_pipeline.chat(query)
    return jsonify(result)


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Admin login endpoint"""
    data = request.json
    username = data.get('username', '')
    password = data.get('password', '')
    
    if auth_manager.verify_credentials(username, password):
        token = auth_manager.generate_token(username)
        return jsonify({'token': token, 'username': username})
    
    return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/api/auth/verify', methods=['GET'])
def verify_token():
    """Verify token endpoint"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    username = auth_manager.verify_token(token)
    
    if username:
        return jsonify({'valid': True, 'username': username})
    
    return jsonify({'valid': False}), 401


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🤖 GIKI CHATBOT BACKEND")
    print("="*60)
    print(f"✓ Server starting on http://localhost:5005")
    print(f"✓ Pinecone: {'Connected' if rag_pipeline.index else 'Not Connected'}")
    print(f"✓ LLM: {'Available' if rag_pipeline.llm_client else 'Not Available'}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5005, debug=True)