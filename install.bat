@echo off
REM GIKI Chatbot Installation Script for Windows

echo Installing GIKI Chatbot dependencies...

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate

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
