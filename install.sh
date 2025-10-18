#!/bin/bash
# GIKI Chatbot Installation Script

echo "Installing GIKI Chatbot dependencies..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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
