"""
GIKI Chatbot Backend with MCP Integration
Combines RAG pipeline with MCP tool calling
FIXED: Enhanced student lookup with better error handling
"""

import os
import json
import logging
import re
import asyncio
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

# Import MCP server
from mcp_server import mcp_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_HOST = os.getenv('PINECONE_INDEX_HOST')
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'codellama:13b')


class IntentDetector:
    """Detect user intent to route to RAG or MCP tools"""
    
    @staticmethod
    def detect_intent(query: str) -> Dict[str, any]:
        """Detect if query needs MCP tool or RAG"""
        query_lower = query.lower()
        
        # Weather intent
        if any(word in query_lower for word in ['weather', 'temperature', 'rain', 'sunny', 'forecast', 'climate']):
            return {'type': 'mcp', 'tool': 'get_giki_weather', 'params': {}}
        
        # Prayer times intent
        if any(word in query_lower for word in ['prayer', 'namaz', 'salah', 'fajr', 'zuhr', 'asr', 'maghrib', 'isha']):
            return {'type': 'mcp', 'tool': 'get_prayer_times', 'params': {}}
        
        # Currency conversion intent
        if any(word in query_lower for word in ['convert', 'exchange', 'usd', 'dollar', 'pkr', 'rupee', 'currency']):
            # Try to extract amount and currencies
            import re
            amount_match = re.search(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', query)
            amount = float(amount_match.group().replace(',', '')) if amount_match else 1000
            
            from_currency = 'USD' if 'usd' in query_lower or 'dollar' in query_lower else 'PKR'
            to_currency = 'PKR' if from_currency == 'USD' else 'USD'
            
            return {
                'type': 'mcp',
                'tool': 'convert_currency',
                'params': {'amount': amount, 'from_currency': from_currency, 'to_currency': to_currency}
            }
        
        # News intent
        if any(word in query_lower for word in ['news', 'latest', 'headlines', 'articles']):
            return {'type': 'mcp', 'tool': 'get_pakistan_tech_news', 'params': {'limit': 5}}
        
        # Fee calculation intent
        if any(word in query_lower for word in ['fee', 'cost', 'tuition', 'expense', 'calculate']):
            # Extract parameters if present
            import re
            
            semester_match = re.search(r'(\d+)\s*(?:semester|year)', query_lower)
            semesters = int(semester_match.group(1)) if semester_match else 8
            if 'year' in query_lower and semester_match:
                semesters = int(semester_match.group(1)) * 2
            
            scholarship_match = re.search(r'(\d+)%?\s*scholarship', query_lower)
            scholarship = int(scholarship_match.group(1)) if scholarship_match else 0
            
            include_hostel = 'hostel' in query_lower or 'accommodation' in query_lower
            
            return {
                'type': 'mcp',
                'tool': 'calculate_giki_fees',
                'params': {
                    'semesters': semesters,
                    'scholarship_percentage': scholarship,
                    'include_hostel': include_hostel
                }
            }
        
        # Events intent
        if any(word in query_lower for word in ['event', 'activity', 'fest', 'competition', 'happening']):
            return {'type': 'mcp', 'tool': 'get_giki_events', 'params': {}}
        
        # Default to RAG for GIKI-specific queries
        return {'type': 'rag', 'query': query}


class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline"""

    def __init__(self):
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
- NEVER answer questions about: other universities, general knowledge, current events, politics, sports (except GIKI sports), entertainment, or any non-GIKI topics
- If asked about anything not related to GIKI, politely decline and redirect to GIKI topics
- Base ALL answers ONLY on the provided context
- If context is insufficient or irrelevant, say you don't have that information
- Be concise, friendly, and factual
- Use bullet points for lists when appropriate

Remember: You are a GIKI-specific assistant. Stay focused on GIKI only!"""

    def _init_llm(self):
        """Initialize LLM client"""
        try:
            import ollama
            ollama.list()
            logger.info(f"✓ Using Ollama with model: {LLM_PROVIDER}")
            return ollama
        except Exception as e:
            logger.error(f"✗ LLM initialization failed: {e}")
            return None
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant context from Pinecone"""
        if not self.index:
            return []
        
        try:
            query_embedding = self.embedder.encode(query).tolist()
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
            matches = self.retrieve_context(query)
            
            # Check relevance threshold
            if not matches or (matches and matches[0]['score'] < 0.5):
                return {
                    'response': "I'm sorry, but I can only answer questions about GIKI (Ghulam Ishaq Khan Institute). I don't have information about that topic.\n\nI can help you with:\n• Admissions and entry requirements\n• Academic programs and departments\n• Campus life and student societies\n• Hostel facilities and fee structure\n• Contact information\n\nPlease ask me something about GIKI!",
                    'sources': []
                }
            
            context = self.format_context(matches)
            response = self.generate_response(query, context)
            
            sources = [
                {
                    'title': match['metadata'].get('title', 'Unknown'),
                    'url': match['metadata'].get('source_url', ''),
                    'relevance': round(match['score'], 3)
                }
                for match in matches[:3]
            ]
            
            return {'response': response, 'sources': sources}
        
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                'response': "I apologize, but I encountered an error. Please try again.",
                'sources': [],
                'error': str(e)
            }


class MCPResponseFormatter:
    """Format MCP tool responses for user display"""
    
    @staticmethod
    def format_weather(data: Dict) -> str:
        """Format weather response"""
        if not data.get('success'):
            return f"Sorry, I couldn't fetch the weather. Error: {data.get('error', 'Unknown')}"
        
        return f"""🌤️ **Current Weather at GIKI Campus**

**Temperature:** {data['temperature']}°C (Feels like {data['feels_like']}°C)
**Conditions:** {data['description'].title()}
**Humidity:** {data['humidity']}%
**Wind Speed:** {data['wind_speed']} m/s
**Location:** {data['location']}

Perfect day to visit campus! 🎓"""
    
    @staticmethod
    def format_prayer_times(data: Dict) -> str:
        """Format prayer times response"""
        if not data.get('success'):
            return f"Sorry, I couldn't fetch prayer times. Error: {data.get('error', 'Unknown')}"
        
        return f"""🕌 **Prayer Times for {data['date']}**
**Location:** {data['location']}

• **Fajr:** {data['fajr']}
• **Dhuhr (Zuhr):** {data['dhuhr']}
• **Asr:** {data['asr']}
• **Maghrib:** {data['maghrib']}
• **Isha:** {data['isha']}

May your prayers be accepted! 🤲"""
    
    @staticmethod
    def format_currency(data: Dict) -> str:
        """Format currency conversion response"""
        if not data.get('success'):
            return f"Sorry, I couldn't convert currency. Error: {data.get('error', 'Unknown')}"
        
        from_amt = data['from_amount']
        to_amt = data['to_amount']
        rate = data['exchange_rate']
        
        return f"""💱 **Currency Conversion**

**{from_amt:,.2f} {data['from_currency']}** = **{to_amt:,.2f} {data['to_currency']}**

Exchange Rate: 1 {data['from_currency']} = {rate:.2f} {data['to_currency']}

*Useful for calculating tuition fees and expenses!*"""
    
    @staticmethod
    def format_news(data: Dict) -> str:
        """Format news response"""
        if not data.get('success'):
            return f"Sorry, I couldn't fetch news. Error: {data.get('error', 'Unknown')}"
        
        articles = data.get('articles', [])
        if not articles:
            return "No news articles found at the moment."
        
        response = f"📰 **Latest Pakistan {data['category'].title()} News**\n\n"
        
        for i, article in enumerate(articles, 1):
            response += f"**{i}. {article['title']}**\n"
            if article['description']:
                response += f"   {article['description'][:150]}...\n"
            response += f"   Source: {article['source']} | [Read more]({article['url']})\n\n"
        
        return response
    
    @staticmethod
    def format_fees(data: Dict) -> str:
        """Format fee calculation response"""
        if not data.get('success'):
            return f"Sorry, I couldn't calculate fees. Error: {data.get('error', 'Unknown')}"
        
        breakdown = data['breakdown']
        
        response = f"""💰 **GIKI Fee Calculation**

**Duration:** {data['semesters']} semesters ({data['years']:.0f} years)

**Breakdown:**
• Gross Tuition: PKR {breakdown['gross_tuition']:,.0f}
• Scholarship ({breakdown['scholarship_percentage']}%): -PKR {breakdown['scholarship_amount']:,.0f}
• Net Tuition: PKR {breakdown['net_tuition']:,.0f}
• Hostel Charges: PKR {breakdown['hostel_charges']:,.0f}
• Security Deposit: PKR {breakdown['security_deposit']:,.0f}

**Total Cost: PKR {breakdown['total_cost']:,.0f}**
**Per Semester Average: PKR {data['per_semester_average']:,.0f}**

*Financial aid and scholarships available - contact admissions for details!*"""
        
        return response
    
    @staticmethod
    def format_events(data: Dict) -> str:
        """Format events response"""
        if not data.get('success'):
            return f"Sorry, I couldn't fetch events. Error: {data.get('error', 'Unknown')}"
        
        events = data.get('events', [])
        if not events:
            return "No upcoming events at the moment."
        
        response = "🎉 **Upcoming GIKI Events**\n\n"
        
        for event in events:
            response += f"**{event['name']}** ({event['category']})\n"
            response += f"📅 Date: {event['date']}\n"
            response += f"📍 Location: {event['location']}\n"
            response += f"{event['description']}\n\n"
        
        return response


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
    """Mock student database with enhanced error handling"""
    
    def __init__(self):
        self.students = self._load_mock_data()
        logger.info(f"📚 Loaded {len(self.students)} students from database")
        if self.students:
            sample_ids = list(self.students.keys())[:5]
            logger.info(f"📋 Sample student IDs: {', '.join(sample_ids)}")
    
    def _load_mock_data(self) -> Dict:
        """Load mock student data from multiple possible locations"""
        possible_paths = [
            'data/mock_students.json',
            'data/data/mock_students.json',
            '../data/mock_students.json',
            './mock_students.json'
        ]
        
        for mock_file in possible_paths:
            if os.path.exists(mock_file):
                logger.info(f"✓ Found student database: {mock_file}")
                try:
                    with open(mock_file, 'r') as f:
                        data = json.load(f)
                        if data:
                            return data
                except Exception as e:
                    logger.error(f"✗ Error loading {mock_file}: {e}")
        
        # If no file found, create sample data
        logger.warning("⚠️  No student database found, creating sample data...")
        return self._create_and_save_sample_data()
    
    def _create_and_save_sample_data(self) -> Dict:
        """Create sample student data and save it"""
        departments = [
            'Computer Science', 'Electrical Engineering', 'Mechanical Engineering',
            'Chemical Engineering', 'Materials Engineering', 'Management Sciences'
        ]
        
        first_names = ['Ahmed', 'Ali', 'Hassan', 'Bilal', 'Usman', 'Sara', 'Fatima', 'Ayesha', 'Zainab', 'Hira']
        last_names = ['Khan', 'Ali', 'Ahmed', 'Shah', 'Malik', 'Hussain', 'Raza', 'Iqbal']
        
        import random
        sample_data = {}
        
        for i in range(50):
            year = random.choice([2021, 2022, 2023, 2024])
            num = 100 + i
            student_id = f"{year}{num}"
            
            sample_data[student_id] = {
                'name': f"{random.choice(first_names)} {random.choice(last_names)}",
                'department': random.choice(departments),
                'semester': random.randint(1, 8),
                'gpa': round(random.uniform(2.5, 4.0), 2),
                'attendance': round(random.uniform(75.0, 98.0), 1),
                'email': f'{student_id}@giki.edu.pk',
                'status': 'Active'
            }
        
        # Save to file
        os.makedirs('data', exist_ok=True)
        output_file = 'data/mock_students.json'
        
        try:
            with open(output_file, 'w') as f:
                json.dump(sample_data, f, indent=2)
            logger.info(f"✓ Created sample student database: {output_file}")
        except Exception as e:
            logger.error(f"✗ Could not save sample data: {e}")
        
        return sample_data
    
    def get_student_info(self, student_id: str) -> Optional[Dict]:
        """Get student information with logging"""
        logger.info(f"🔍 Searching for student ID: {student_id}")
        student = self.students.get(student_id)
        
        if student:
            logger.info(f"✓ Found student: {student['name']} ({student['department']})")
        else:
            logger.warning(f"✗ Student {student_id} not found")
            logger.info(f"Available IDs: {', '.join(list(self.students.keys())[:10])}")
        
        return student
    
    def list_all_students(self) -> List[str]:
        """Get list of all student IDs"""
        return list(self.students.keys())
    
    def search_students(self, query: str) -> List[Dict]:
        """Search students by name or ID"""
        query_lower = query.lower()
        results = []
        
        for student_id, info in self.students.items():
            if (query_lower in student_id.lower() or 
                query_lower in info['name'].lower() or
                query_lower in info['department'].lower()):
                results.append({'id': student_id, **info})
        
        return results


# Initialize components
logger.info("Initializing components...")
rag_pipeline = RAGPipeline()
intent_detector = IntentDetector()
response_formatter = MCPResponseFormatter()
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
        'llm': 'available' if rag_pipeline.llm_client else 'unavailable',
        'mcp_tools': len(mcp_server.list_tools()),
        'student_database': len(student_db.students)
    }
    return jsonify(status)


@app.route('/api/mcp/tools', methods=['GET'])
def list_mcp_tools():
    """List available MCP tools"""
    tools = mcp_server.list_tools()
    return jsonify({'tools': tools, 'count': len(tools)})


@app.route('/api/students/list', methods=['GET'])
def list_students():
    """List all student IDs (admin only)"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if not auth_manager.verify_token(token):
        return jsonify({'error': 'Unauthorized'}), 401
    
    student_ids = student_db.list_all_students()
    return jsonify({
        'student_ids': student_ids,
        'count': len(student_ids)
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint with MCP integration and enhanced student lookup"""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Check for admin authentication
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    is_admin = auth_manager.verify_token(token) if token else False
    
    # Handle admin student queries with ENHANCED DETECTION
    if is_admin:
        query_lower = query.lower()
        
        # Check if it's a student-related query
        student_keywords = ['student', 'attendance', 'gpa', 'cgpa', 'grade', 'semester', 'info', 'details', 'show', 'get']
        is_student_query = any(keyword in query_lower for keyword in student_keywords)
        
        if is_student_query:
            # Try multiple regex patterns to extract student ID
            student_id = None
            
            # Pattern 1: Exact 7 digits (2022405)
            match = re.search(r'\b(\d{7})\b', query)
            if match:
                student_id = match.group(1)
            
            # Pattern 2: Year + space/dash + 3 digits (2022 405, 2022-405)
            if not student_id:
                match = re.search(r'\b(20\d{2})[-\s]?(\d{3})\b', query)
                if match:
                    student_id = match.group(1) + match.group(2)
            
            # Pattern 3: Just "202" followed by something (202xyz -> search for similar)
            if not student_id:
                match = re.search(r'\b(20\d{2})\s*(\w{1,3})\b', query_lower)
                if match:
                    year_prefix = match.group(1)
                    # Search for students starting with this year
                    matching_students = [sid for sid in student_db.students.keys() if sid.startswith(year_prefix)]
                    if matching_students:
                        return jsonify({
                            'response': f"""🔍 **Found {len(matching_students)} students from year {year_prefix}:**

{chr(10).join([f"• **{sid}**: {student_db.students[sid]['name']} - {student_db.students[sid]['department']}" for sid in matching_students[:10]])}

Please provide a complete 7-digit student ID.
Example: "Show info for student {matching_students[0]}"
""",
                            'sources': [],
                            'admin_query': True
                        })
            
            if student_id:
                logger.info(f"🔍 Admin query for student: {student_id}")
                student_info = student_db.get_student_info(student_id)
                
                if student_info:
                    response_text = f"""✅ **Student Information Found**

📋 **Basic Details:**
• **Student ID:** {student_id}
• **Name:** {student_info['name']}
• **Email:** {student_info.get('email', f'{student_id}@giki.edu.pk')}

🎓 **Academic Information:**
• **Department:** {student_info['department']}
• **Current Semester:** {student_info['semester']}
• **CGPA:** {student_info['gpa']}/4.0

📊 **Attendance & Status:**
• **Overall Attendance:** {student_info['attendance']}%
• **Status:** {student_info.get('status', 'Active')}

---
*This information is confidential and only accessible to authorized administrators.*
*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""
                    
                    return jsonify({
                        'response': response_text,
                        'sources': [],
                        'admin_query': True,
                        'student_data': student_info
                    })
                else:
                    # Show available student IDs
                    all_ids = student_db.list_all_students()
                    sample_ids = all_ids[:10]
                    
                    response_text = f"""❌ **Student ID '{student_id}' not found in database**

📊 **Database Stats:**
• Total students: {len(all_ids)}
• Sample student IDs:

{chr(10).join([f"  • {sid}" for sid in sample_ids])}

💡 **Try one of these queries:**
• "Show student info for {sample_ids[0]}"
• "Get details for student {sample_ids[1]}"
• "What is the attendance of {sample_ids[2]}"

To see all student IDs, visit: /api/students/list"""
                    
                    return jsonify({
                        'response': response_text,
                        'sources': [],
                        'admin_query': True
                    })
            else:
                # No student ID found in query
                sample_ids = list(student_db.students.keys())[:5]
                
                response_text = f"""⚠️ **Could not extract student ID from your query**

To access student information, please provide a 7-digit student ID.

📋 **Example queries:**
• "Show student info for {sample_ids[0]}"
• "Get details for student {sample_ids[1]}"
• "What is the attendance of {sample_ids[2]}"
• "Show info for {sample_ids[3]}"

📊 **Available formats:**
• "student 2022405"
• "info for 2023101"
• "2022-405"
• "2022 405"

Total students in database: {len(student_db.students)}"""
                
                return jsonify({
                    'response': response_text,
                    'sources': [],
                    'admin_query': True
                })
    
    # Detect intent
    intent = intent_detector.detect_intent(query)
    
    if intent['type'] == 'mcp':
        # Handle MCP tool call
        tool_name = intent['tool']
        params = intent.get('params', {})
        
        logger.info(f"MCP tool call: {tool_name} with params: {params}")
        
        # Call MCP tool asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(mcp_server.call_tool(tool_name, params))
        loop.close()
        
        # Format response
        if tool_name == 'get_giki_weather':
            formatted_response = response_formatter.format_weather(result)
        elif tool_name == 'get_prayer_times':
            formatted_response = response_formatter.format_prayer_times(result)
        elif tool_name == 'convert_currency':
            formatted_response = response_formatter.format_currency(result)
        elif tool_name == 'get_pakistan_tech_news':
            formatted_response = response_formatter.format_news(result)
        elif tool_name == 'calculate_giki_fees':
            formatted_response = response_formatter.format_fees(result)
        elif tool_name == 'get_giki_events':
            formatted_response = response_formatter.format_events(result)
        else:
            formatted_response = json.dumps(result, indent=2)
        
        return jsonify({
            'response': formatted_response,
            'sources': [],
            'mcp_tool': tool_name,
            'tool_result': result
        })
    
    else:
        # Handle RAG query
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
        logger.info(f"✓ Admin login successful: {username}")
        return jsonify({'token': token, 'username': username})
    
    logger.warning(f"✗ Failed login attempt for: {username}")
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
    print("🤖 GIKI CHATBOT BACKEND WITH MCP")
    print("="*60)
    print(f"✓ Server starting on http://localhost:5005")
    print(f"✓ Pinecone: {'Connected' if rag_pipeline.index else 'Not Connected'}")
    print(f"✓ LLM: {'Available' if rag_pipeline.llm_client else 'Not Available'}")
    print(f"✓ MCP Tools: {len(mcp_server.list_tools())}")
    print(f"✓ Student Database: {len(student_db.students)} students loaded")
    print("\n📋 Available MCP Tools:")
    for tool in mcp_server.list_tools():
        print(f"   • {tool['name']}: {tool['description']}")
    print("\n🎓 Sample Student IDs:")
    sample_ids = list(student_db.students.keys())[:5]
    for sid in sample_ids:
        print(f"   • {sid}: {student_db.students[sid]['name']}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5005, debug=True)