@echo off
echo ========================================
echo Echo AI - Quick Start Guide
echo ========================================
echo.
echo STEP 1: Add Your API Keys
echo ========================================
echo.
echo 1. Open: .env file (in this folder)
echo 2. Replace these lines with your actual keys:
echo.
echo    GROQ_API_KEY=gsk_your_groq_key_here
echo    PINECONE_API_KEY=pcsk_your_pinecone_key_here
echo.
echo 3. Save the file
echo.
pause
echo.
echo STEP 2: Install Dependencies
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.9+ from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

echo Python found! Installing packages...
echo.

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Installing dependencies (this may take 2-3 minutes)...
pip install -r requirements.txt --quiet

echo.
echo STEP 3: Starting Echo AI
echo ========================================
echo.
echo Opening browser at: http://localhost:8501
echo.
echo To stop the server, press Ctrl+C
echo.
echo ========================================
echo QUICK TEST:
echo ========================================
echo 1. Upload sample_faq.md (in this folder)
echo 2. Click "Process Documents"
echo 3. Ask: "What is your refund policy?"
echo 4. Refresh page - data persists!
echo ========================================
echo.

streamlit run app.py
