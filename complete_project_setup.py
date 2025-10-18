"""
Complete GIKI Chatbot Project Setup Script
Run this to set up all missing components and generate sample data
"""

import json
import os
from pathlib import Path

def create_directory_structure():
    """Create all necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'data/analysis',
        'static',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_env_file():
    """Create .env template file"""
    env_content = """# GIKI Chatbot Environment Variables

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=gcp-starter

# LLM Provider (choose one: ollama, groq, gemini)
LLM_PROVIDER=ollama

# Groq API (if using Groq)
GROQ_API_KEY=your-groq-api-key-here

# Google Gemini API (if using Gemini)
GOOGLE_API_KEY=your-google-api-key-here

# Flask Secret Key (change this!)
SECRET_KEY=your-secret-key-change-this-to-something-random

# Admin Credentials (CHANGE THESE!)
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("✓ Created .env file")

def generate_sample_giki_data():
    """Generate comprehensive sample GIKI data for testing"""
    
    giki_data = [
        {
            "url": "https://giki.edu.pk/admissions/undergraduate",
            "title": "Undergraduate Admissions - GIKI",
            "category": "admissions",
            "content": """Ghulam Ishaq Khan Institute of Engineering Sciences and Technology (GIKI) offers undergraduate programs in various disciplines. 

Admission Requirements:
- Minimum 60% marks in FSc Pre-Engineering or equivalent
- Valid SAT score (minimum 1000) or NTS score
- Passing the GIKI entrance test

Application Process:
1. Register online at admissions.giki.edu.pk
2. Submit required documents (transcripts, CNIC, photographs)
3. Pay application fee
4. Appear for entrance test
5. Merit list announcement

Important Dates:
- Application deadline: Usually in July
- Entrance test: August
- Classes begin: September

Fee Structure:
- Tuition fee: PKR 250,000 per semester
- Hostel charges: PKR 50,000 per semester
- Security deposit: PKR 25,000 (one-time)

Financial Aid:
GIKI offers need-based and merit-based scholarships covering up to 100% tuition fee."""
        },
        {
            "url": "https://giki.edu.pk/academics/departments",
            "title": "Academic Departments - GIKI",
            "category": "academics",
            "content": """GIKI offers programs through six faculties:

1. Faculty of Computer Science and Engineering
   - Bachelor of Science in Computer Science (BSCS)
   - Bachelor of Science in Software Engineering (BSSE)
   - Master of Science in Computer Science (MSCS)

2. Faculty of Electrical Engineering
   - Bachelor of Science in Electrical Engineering (BSEE)
   - Specializations: Power Systems, Electronics, Telecommunications

3. Faculty of Mechanical Engineering
   - Bachelor of Science in Mechanical Engineering (BSME)
   - Focus areas: Thermal Sciences, Design, Manufacturing

4. Faculty of Materials and Chemical Engineering
   - Bachelor of Science in Chemical Engineering
   - Bachelor of Science in Materials Engineering

5. Faculty of Management Sciences
   - Bachelor of Business Administration (BBA)
   - Master of Business Administration (MBA)

6. Faculty of Basic and Applied Sciences
   - Offers foundational courses in Mathematics, Physics, Chemistry

All programs are HEC recognized and internationally accredited by PEC (for engineering)."""
        },
        {
            "url": "https://giki.edu.pk/campus-life/societies",
            "title": "Student Societies - GIKI",
            "category": "student_life",
            "content": """GIKI has a vibrant campus life with over 30 student societies:

Technical Societies:
- ACM (Association for Computing Machinery)
- IEEE Student Branch
- ASME (Mechanical Engineering)
- Robotics Society
- Google Developer Student Club (GDSC)

Cultural Societies:
- Dramatics Society
- Music Society
- Fine Arts Society
- Literary Society
- Photography Club

Sports Clubs:
- Cricket Club
- Football Club
- Basketball Club
- Table Tennis Club
- Badminton Club

Social Impact:
- Community Service Society
- Red Crescent Society
- Environmental Society

Events organized:
- GIKI Annual Dinner (GAD)
- Tech Fest
- Cultural Week
- Sports Gala
- Hackathons and coding competitions

All students are encouraged to join societies for holistic development."""
        },
        {
            "url": "https://giki.edu.pk/campus-life/hostels",
            "title": "Hostel Facilities - GIKI",
            "category": "student_life",
            "content": """GIKI provides separate hostel facilities for male and female students.

Hostel Blocks:
- Male Hostels: 8 blocks (A, B, C, D, E, F, G, H)
- Female Hostel: Dedicated block with enhanced security

Room Types:
- Single occupancy rooms for final year students
- Double occupancy rooms for junior students
- Triple rooms in some blocks

Facilities:
- 24/7 electricity and internet
- Common rooms with TV and games
- Study areas and reading rooms
- Laundry facilities
- Mess and cafeteria
- Medical dispensary nearby
- Sports facilities (gym, courts)

Mess Menu:
- Breakfast: 7:30 AM - 9:30 AM
- Lunch: 1:00 PM - 3:00 PM
- Dinner: 7:30 PM - 9:30 PM

Hostel Rules:
- Curfew timings enforced
- No opposite gender allowed in hostel premises
- Visitors must register at entrance
- Keep rooms clean and organized

Security:
- 24/7 security guards
- CCTV surveillance
- Biometric entry system"""
        },
        {
            "url": "https://giki.edu.pk/about/history",
            "title": "History of GIKI",
            "category": "about",
            "content": """Ghulam Ishaq Khan Institute of Engineering Sciences and Technology was established in 1993.

Foundation:
- Founded by former President of Pakistan, Ghulam Ishaq Khan
- Established with vision to create world-class technical education
- Located in Topi, Swabi, Khyber Pakhtunkhwa

Milestones:
- 1993: First batch admitted
- 1997: First convocation held
- 2000: Achieved PEC accreditation
- 2005: Established PhD programs
- 2010: Expanded to 6 faculties
- 2020: Celebrated 27 years of excellence

Campus:
- Spread over 400+ acres
- Modern architecture with state-of-the-art facilities
- Located near Islamabad-Peshawar motorway
- 2 hours from Islamabad, 1 hour from Peshawar

Recognition:
- Ranked among top 5 engineering universities in Pakistan
- HEC recognized Category W university
- International collaborations with MIT, Stanford, Cambridge

Notable Alumni:
- Leading positions in tech companies (Google, Microsoft, Amazon)
- Entrepreneurs and startup founders
- Academia and research institutions
- Government and public sector"""
        },
        {
            "url": "https://giki.edu.pk/research",
            "title": "Research at GIKI",
            "category": "academics",
            "content": """GIKI emphasizes research and innovation across all disciplines.

Research Centers:
- Center for Advanced Research in Engineering (CARE)
- Renewable Energy Research Center
- Materials Science Laboratory
- Software Engineering Lab
- AI and Machine Learning Lab

Focus Areas:
- Artificial Intelligence and Machine Learning
- Renewable Energy Systems
- Materials Science
- Robotics and Automation
- Network Security and Blockchain
- Sustainable Development

Publications:
- Over 500 research papers in international journals
- Multiple patents filed
- Regular publications in IEEE, Springer, Elsevier

Funding:
- HEC funded projects
- Industry collaborations
- International grants
- Public-private partnerships

Student Research:
- Final Year Projects (FYP)
- Undergraduate Research Program
- MS and PhD thesis research
- Research assistantships available

Conferences:
- Annual research conference
- Participation in international conferences
- Hosting workshops and seminars"""
        },
        {
            "url": "https://giki.edu.pk/placements",
            "title": "Career and Placements - GIKI",
            "category": "academics",
            "content": """GIKI has excellent placement record with top companies recruiting from campus.

Career Development Center:
- Resume building workshops
- Interview preparation sessions
- Mock interviews
- Career counseling
- Industry mentorship program

Top Recruiters:
Technology:
- Google, Microsoft, Amazon
- IBM, Oracle, SAP
- Systems Limited, NetSol
- Arbisoft, 10Pearls

Engineering:
- Siemens, ABB
- Pakistan State Oil (PSO)
- K-Electric
- PTCL, Huawei

Consulting & Finance:
- McKinsey, BCG
- Deloitte, EY, PwC
- Banks (HBL, MCB, UBL)

Average Package:
- Fresh graduates: PKR 80,000 - 150,000/month
- Top performers: PKR 200,000+/month
- International offers: $60,000 - 100,000/year

Internships:
- Summer internship programs
- Industry-sponsored projects
- Remote internships with international companies

Alumni Network:
- Strong alumni network globally
- Alumni mentorship program
- Networking events and reunions"""
        },
        {
            "url": "https://giki.edu.pk/admissions/graduate",
            "title": "Graduate Admissions - GIKI",
            "category": "admissions",
            "content": """GIKI offers MS and PhD programs in various disciplines.

MS Programs:
- MS Computer Science
- MS Electrical Engineering
- MS Mechanical Engineering
- MS Materials Engineering
- MS Management Sciences

PhD Programs:
Available in all engineering disciplines and management sciences.

Admission Requirements (MS):
- 16 years of education (BS/BE)
- Minimum CGPA 2.5/4.0
- Valid GRE (International) or GAT (General) score
- Two recommendation letters
- Statement of purpose

PhD Requirements:
- MS/MPhil degree in relevant field
- Minimum CGPA 3.0/4.0
- Research proposal
- Interview with faculty

Duration:
- MS: 2 years (coursework + thesis)
- PhD: 3-5 years (coursework + dissertation)

Financial Support:
- All PhD students receive stipend
- Research assistantships for MS students
- Teaching assistantships available
- Tuition fee waivers for qualified students

Application Process:
1. Apply online
2. Submit transcripts and test scores
3. Interview (if shortlisted)
4. Admission decision

Deadlines:
- Fall semester: June 30
- Spring semester: November 30"""
        },
        {
            "url": "https://giki.edu.pk/contact",
            "title": "Contact GIKI",
            "category": "about",
            "content": """Contact Information for GIKI:

Main Campus:
Ghulam Ishaq Khan Institute of Engineering Sciences and Technology
Topi, Swabi, Khyber Pakhtunkhwa, Pakistan

Phone Numbers:
- Main Switchboard: +92-938-281026-34
- Admissions Office: +92-938-281030
- Faculty Office: +92-938-281050

Email Addresses:
- General Inquiries: info@giki.edu.pk
- Admissions: admissions@giki.edu.pk
- Faculty: faculty@giki.edu.pk
- Career Services: placement@giki.edu.pk

Website: www.giki.edu.pk

Social Media:
- Facebook: /GIKIOfficial
- Twitter: @GIKIOfficial
- LinkedIn: Ghulam Ishaq Khan Institute
- Instagram: @giki_official

Visiting:
- Located on Swabi-Mardan Road
- 2 hours from Islamabad via M1 Motorway
- Nearest airport: Islamabad International Airport
- Taxi/ride service available from major cities

Campus Tours:
- Available for prospective students
- Book in advance through admissions office
- Virtual tours available on website

Office Hours:
- Monday to Friday: 9:00 AM - 5:00 PM
- Saturday: 9:00 AM - 1:00 PM
- Sunday: Closed"""
        },
        {
            "url": "https://giki.edu.pk/facilities",
            "title": "Campus Facilities - GIKI",
            "category": "student_life",
            "content": """GIKI campus provides world-class facilities for students.

Academic Facilities:
- Modern classrooms with multimedia
- Computer labs with latest hardware
- Engineering labs (circuits, machines, thermal)
- Libraries with 50,000+ books
- Digital library with international journals

Sports Facilities:
- Cricket ground
- Football ground
- Basketball courts
- Volleyball courts
- Tennis courts
- Badminton hall
- Gymnasium with modern equipment
- Swimming pool

Medical:
- On-campus dispensary
- 24/7 emergency medical care
- Visiting doctors
- Ambulance service
- Tie-up with nearby hospitals

Transport:
- University buses for Islamabad/Peshawar
- Local transport for Swabi/Mardan
- Pickup/drop service for events

Food & Dining:
- Main mess (3 meals daily)
- Multiple cafeterias
- Juice corner
- Pizza/fast food outlets
- Café with snacks

Other Facilities:
- Mosque with separate prayer areas
- Banking/ATM services
- Post office
- Stationery shop
- Printing/photocopying
- WiFi coverage across campus
- Auditorium (capacity 1000+)
- Seminar halls"""
        }
    ]
    
    # Save to file
    output_path = Path('data/raw/giki_targeted.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(giki_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Generated sample GIKI data: {len(giki_data)} documents")
    return giki_data

def generate_mock_students():
    """Generate mock student database"""
    departments = ['Computer Science', 'Electrical Engineering', 'Mechanical Engineering',
                   'Chemical Engineering', 'Materials Engineering', 'Management Sciences']
    first_names = ['Ahmed', 'Ali', 'Hassan', 'Usman', 'Bilal', 'Hamza',
                   'Fatima', 'Ayesha', 'Sara', 'Zainab', 'Maryam', 'Hira']
    last_names = ['Khan', 'Ali', 'Ahmed', 'Shah', 'Malik', 'Hussain']
    
    students = {}
    
    for i in range(50):
        import random
        year = random.choice([2021, 2022, 2023, 2024])
        student_id = f"{year}{random.randint(100, 999)}"
        
        students[student_id] = {
            'name': f"{random.choice(first_names)} {random.choice(last_names)}",
            'department': random.choice(departments),
            'semester': random.randint(1, 8),
            'gpa': round(random.uniform(2.5, 4.0), 2),
            'attendance': round(random.uniform(75.0, 98.0), 1),
            'email': f"{student_id}@giki.edu.pk",
            'status': 'Active'
        }
    
    output_path = Path('data/mock_students.json')
    with open(output_path, 'w') as f:
        json.dump(students, f, indent=2)
    
    print(f"✓ Generated {len(students)} mock students")

def create_installation_script():
    """Create installation script"""
    install_script = """#!/bin/bash
# GIKI Chatbot Installation Script

echo "Installing GIKI Chatbot dependencies..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: python preprocessing.py (to process data and create embeddings)"
echo "3. Run: python backend.py (to start the backend server)"
echo "4. Open index.html in a browser or serve it with a local server"
"""
    
    with open('install.sh', 'w') as f:
        f.write(install_script)
    
    os.chmod('install.sh', 0o755)
    print("✓ Created install.sh script")
    
    # Windows version
    install_bat = """@echo off
REM GIKI Chatbot Installation Script for Windows

echo Installing GIKI Chatbot dependencies...

REM Create virtual environment
python -m venv venv
call venv\\Scripts\\activate

REM Upgrade pip
pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt

REM Download spaCy model
python -m spacy download en_core_web_sm

echo Installation complete!
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Run: python preprocessing.py
echo 3. Run: python backend.py
echo 4. Open index.html in a browser
"""
    
    with open('install.bat', 'w') as f:
        f.write(install_bat)
    
    print("✓ Created install.bat script")

def create_readme():
    """Create comprehensive README"""
    readme = """# GIKI Smart Chatbot 🤖

An intelligent, RAG-powered chatbot for Ghulam Ishaq Khan Institute (GIKI) with admin features and MCP support.

## Features ✨

- 🔍 **Semantic Search**: Uses embeddings and Pinecone for accurate information retrieval
- 💬 **Natural Conversations**: Powered by LLaMA/Mistral/Gemini
- 🔐 **Admin Portal**: Secure login for student data access
- 📊 **Analytics**: PCA and UMAP visualization of embeddings
- 🌐 **Modern UI**: Clean, responsive web interface
- ☁️ **Cloud Ready**: Deploy on AWS free tier

## Project Structure 📁

```
giki-chatbot/
├── data/
│   ├── raw/                 # Scraped data
│   ├── processed/           # Cleaned chunks with embeddings
│   └── analysis/            # PCA/UMAP results
├── scrapper.py             # Web scraping with Selenium
├── preprocessing.py        # Data cleaning and embedding generation
├── backend.py             # Flask API with RAG pipeline
├── index.html            # Frontend chat interface
├── utils.py              # Helper functions
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this!)
└── README.md            # This file
```

## Quick Start 🚀

### 1. Installation

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

**Windows:**
```cmd
install.bat
```

**Or manually:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configuration

Edit `.env` file with your API keys:

```env
# Get free Pinecone API key from: https://www.pinecone.io/
PINECONE_API_KEY=your-pinecone-api-key

# Choose LLM provider
LLM_PROVIDER=ollama  # or groq, gemini

# For Ollama (local, free)
# Install from: https://ollama.ai
# Then run: ollama pull llama3.2

# For Groq (free API)
# Get key from: https://console.groq.com
GROQ_API_KEY=your-groq-api-key

# For Google Gemini (free tier)
# Get key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your-google-api-key
```

### 3. Data Pipeline

```bash
# Step 1: Scrape GIKI website (optional - sample data included)
python scrapper.py

# Step 2: Process data and generate embeddings
python preprocessing.py

# This will:
# - Clean and chunk text
# - Generate embeddings
# - Create PCA/UMAP visualizations
# - Upload to Pinecone
```

### 4. Run the Chatbot

```bash
# Start backend server
python backend.py

# Backend will run on http://localhost:5000
```

Open `index.html` in your browser or serve it:

```bash
# Using Python
python -m http.server 8000

# Then open http://localhost:8000
```

## API Endpoints 🔌

### Chat Endpoint
```bash
POST /api/chat
Content-Type: application/json

{
  "query": "What programs does GIKI offer?",
  "history": []
}
```

### Admin Login
```bash
POST /api/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}
```

### Student Info (Admin only)
```bash
POST /api/chat
Authorization: Bearer <token>

{
  "query": "Get info for student 2022405"
}
```

## Admin Features 🔐

Default credentials (⚠️ Change these!):
- Username: `admin`
- Password: `admin123`

Admin can query:
- Student information
- Attendance records
- Grades and performance

## LLM Options 🤖

### Option 1: Ollama (Recommended - Free & Local)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.2

# Set in .env
LLM_PROVIDER=ollama
```

### Option 2: Groq (Free API - Fast)
```bash
# Get free API key from https://console.groq.com
# Set in .env
LLM_PROVIDER=groq
GROQ_API_KEY=your-key
```

### Option 3: Google Gemini (Free Tier)
```bash
# Get API key from https://makersuite.google.com/app/apikey
# Set in .env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-key
```

## Deployment ☁️

### AWS EC2 (Free Tier)

```bash
# 1. Launch t2.micro EC2 instance
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Clone and setup
git clone your-repo
cd giki-chatbot
./install.sh

# 4. Run with gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend:app
```

### Frontend Deployment

**Option 1: AWS S3 Static Hosting**
- Upload `index.html` to S3 bucket
- Enable static website hosting
- Update API_BASE_URL in index.html

**Option 2: Vercel/Netlify**
- Deploy index.html
- Add environment variable for API URL

## Data Visualization 📊

After running `preprocessing.py`, visualizations are saved in `data/analysis/`:

- `embeddings_umap2d.npy` - 2D UMAP projection
- `embeddings_umap3d.npy` - 3D UMAP projection
- `embeddings_pca50.npy` - 50-component PCA

View with:
```python
import numpy as np
import matplotlib.pyplot as plt

umap_2d = np.load('data/analysis/embeddings_umap2d.npy')
plt.scatter(umap_2d[:, 0], umap_2d[:, 1])
plt.show()
```

## Troubleshooting 🔧

**Pinecone Connection Error:**
- Check your API key is correct
- Verify environment is set to 'gcp-starter'

**LLM Not Responding:**
- For Ollama: Check it's running with `ollama list`
- For APIs: Verify API keys are valid

**Embedding Dimension Mismatch:**
- Ensure all embeddings use same model
- Delete and recreate Pinecone index if needed

**Port Already in Use:**
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9
```

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License 📄

MIT License - feel free to use for academic projects!

## Acknowledgments 🙏

- GIKI for inspiration
- Anthropic Claude for assistance
- Open source community

## Contact 📧

For questions or support, please open an issue on GitHub.

---

**⭐ Star this repo if you find it helpful!**
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    
    print("✓ Created README.md")

def create_test_queries():
    """Create test queries file"""
    test_queries = {
        "general": [
            "What programs does GIKI offer?",
            "How do I apply to GIKI?",
            "What is the fee structure?",
            "Tell me about campus facilities",
            "What societies are available?"
        ],
        "admissions": [
            "What is the admission deadline?",
            "What is the minimum GPA requirement?",
            "Do I need to give an entrance test?",
            "Is financial aid available?",
            "How do I apply for graduate programs?"
        ],
        "academic": [
            "Which department should I choose?",
            "What is the duration of BS programs?",
            "Are online courses available?",
            "What is the grading system?",
            "Can I do a double major?"
        ],
        "admin": [
            "Get student info for 2022405",
            "What is the attendance of student 2023101?",
            "Show me details for student 2021234"
        ]
    }
    
    with open('test_queries.json', 'w') as f:
        json.dump(test_queries, f, indent=2)
    
    print("✓ Created test_queries.json")

def main():
    """Run complete setup"""
    print("=" * 60)
    print("GIKI CHATBOT - COMPLETE PROJECT SETUP")
    print("=" * 60)
    print()
    
    print("Setting up project structure...")
    create_directory_structure()
    print()
    
    print("Creating configuration files...")
    create_env_file()
    print()
    
    print("Generating sample data...")
    generate_sample_giki_data()
    generate_mock_students()
    print()
    
    print("Creating installation scripts...")
    create_installation_script()
    print()
    
    print("Creating documentation...")
    create_readme()
    create_test_queries()
    print()
    
    print("=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run: ./install.sh (Linux/Mac) or install.bat (Windows)")
    print("2. Edit .env file with your API keys")
    print("3. Run: python preprocessing.py")
    print("4. Run: python backend.py")
    print("5. Open index.html in your browser")
    print()
    print("See README.md for detailed instructions!")

if __name__ == "__main__":
    main()