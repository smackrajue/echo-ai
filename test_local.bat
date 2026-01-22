@echo off
echo ========================================
echo Echo AI - Local Testing
echo ========================================
echo.

REM Check if .env exists
if not exist .env (
    echo ERROR: .env file not found!
    echo.
    echo Please create .env file with your API keys:
    echo   GROQ_API_KEY=gsk_your_key
    echo   PINECONE_API_KEY=your_key
    echo   PINECONE_ENVIRONMENT=us-east-1
    echo.
    pause
    exit /b 1
)

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.9+ from https://www.python.org
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
if not exist venv (
    python -m venv venv
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Starting Echo AI...
echo ========================================
echo.
echo The app will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py
