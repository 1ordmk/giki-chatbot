# GIKI Smart Chatbot 🤖

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
source venv/bin/activate  # Windows: venv\Scripts\activate

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
