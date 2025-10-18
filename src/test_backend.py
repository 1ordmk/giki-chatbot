from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:8000", "http://127.0.0.1:8000"]}})

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')

# Mock database
DEPARTMENTS = {
    "FEE": {
        "name": "Faculty of Electronic Engineering",
        "programs": ["Electronics", "Computer Engineering", "Telecommunication Engineering", "AI & Data Science"],
        "description": "The Faculty of Electronic Engineering offers cutting-edge programs in electronics, computing, and telecommunications."
    },
    "FME": {
        "name": "Faculty of Mechanical Engineering",
        "programs": ["Mechanical Engineering", "Materials Engineering", "Industrial Engineering"],
        "description": "The Faculty of Mechanical Engineering provides comprehensive education in mechanical systems and industrial processes."
    },
    "FCSE": {
        "name": "Faculty of Computer Science and Engineering",
        "programs": ["Computer Science", "Software Engineering"],
        "description": "The Faculty of Computer Science and Engineering focuses on modern computing technologies and software development."
    }
}

SOCIETIES = {
    "IEEE": {
        "name": "IEEE GIKI Student Branch",
        "description": "Professional development and technical activities",
        "type": "Professional"
    },
    "ACM": {
        "name": "ACM GIKI Chapter",
        "description": "Computing and programming competitions",
        "type": "Technical"
    },
    "GIKS": {
        "name": "GIKI Sports Society",
        "description": "Sports events and activities",
        "type": "Sports"
    },
    "GIKIST": {
        "name": "GIKI Society of Performing Arts",
        "description": "Cultural events and performances",
        "type": "Cultural"
    }
}

ADMISSION_INFO = {
    "process": {
        "steps": [
            "Register online at admissions.giki.edu.pk",
            "Complete the application form",
            "Pay the application fee",
            "Take the GIKI Entry Test",
            "Wait for merit list announcement"
        ],
        "requirements": [
            "FSc Pre-Engineering or equivalent with minimum 60% marks",
            "A-Level with Mathematics, Physics, and Chemistry/Computer Science",
            "SAT Subject Test scores (for overseas candidates)",
            "Valid email address and phone number"
        ],
        "deadlines": {
            "registration_starts": "February 1st",
            "registration_ends": "April 15th",
            "test_date": "May (exact date announced yearly)"
        }
    }
}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/api/login', methods=['POST'])
def login():
    """Admin login endpoint"""
    auth = request.json
    if not auth or not auth.get('username') or not auth.get('password'):
        return jsonify({'message': 'Could not verify'}), 401

    if auth.get('username') == ADMIN_USERNAME and auth.get('password') == ADMIN_PASSWORD:
        return jsonify({'message': 'Login successful'})

    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    data = request.json
    query = data.get('query', '').lower()
    
    if not query:
        return jsonify({'error': 'Query is required'}), 400

    # Process admission-related queries
    if 'how' in query and 'apply' in query:
        return jsonify({
            'response': "Here's how to apply to GIKI:\n\n" + 
                       "\n".join(f"- {step}" for step in ADMISSION_INFO['process']['steps']) +
                       "\n\nRequirements:\n" +
                       "\n".join(f"- {req}" for req in ADMISSION_INFO['process']['requirements']),
            'sources': ['admissions.giki.edu.pk']
        })

    # Process department-related queries
    if any(word in query for word in ['department', 'faculty', 'program']):
        departments_info = []
        for code, dept in DEPARTMENTS.items():
            departments_info.append(f"{code} - {dept['name']}:\n" +
                                 f"Programs: {', '.join(dept['programs'])}\n" +
                                 f"{dept['description']}")
        return jsonify({
            'response': "Here are GIKI's departments and programs:\n\n" + "\n\n".join(departments_info),
            'sources': ['giki.edu.pk/academics']
        })

    # Process society-related queries
    if any(word in query for word in ['society', 'societies', 'club']):
        societies_info = []
        for code, society in SOCIETIES.items():
            societies_info.append(f"{society['name']} ({code}):\n" +
                               f"Type: {society['type']}\n" +
                               f"{society['description']}")
        return jsonify({
            'response': "Here are the major societies at GIKI:\n\n" + "\n\n".join(societies_info),
            'sources': ['giki.edu.pk/student-life']
        })

    # Default response for unhandled queries
    return jsonify({
        'response': "I can help you with information about:\n" +
                   "- Admission process and requirements\n" +
                   "- Departments and academic programs\n" +
                   "- Student societies and campus life\n\n" +
                   "Please ask specific questions about these topics!",
        'sources': []
    })

if __name__ == '__main__':
    print("Starting server on http://127.0.0.1:5005")
    app.run(host='127.0.0.1', port=5005, debug=False, use_reloader=False)