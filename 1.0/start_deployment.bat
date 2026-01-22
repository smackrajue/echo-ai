@echo off
echo ========================================
echo Echo AI - Quick Deployment Helper
echo ========================================
echo.

echo Step 1: Opening API Key Registration Pages...
echo.
echo Opening Groq Console...
start https://console.groq.com
timeout /t 2 /nobreak >nul

echo Opening Pinecone...
start https://www.pinecone.io
timeout /t 2 /nobreak >nul

echo Opening HuggingFace Spaces...
start https://huggingface.co/spaces
echo.

echo ========================================
echo NEXT STEPS:
echo ========================================
echo.
echo 1. Sign up for Groq (https://console.groq.com)
echo    - Create API key
echo    - Copy key (starts with gsk_...)
echo.
echo 2. Sign up for Pinecone (https://www.pinecone.io)
echo    - Create API key
echo    - Note environment (e.g., us-east-1)
echo.
echo 3. Edit .env file and paste your API keys
echo.
echo 4. Run: test_local.bat (to test locally)
echo.
echo 5. Deploy to HuggingFace Spaces
echo    - Create new Space: echo-ai
echo    - Upload: app.py, requirements.txt, packages.txt
echo    - Add secrets in Settings
echo.
echo ========================================
echo See DEPLOYMENT_STEPS.md for full guide
echo ========================================
pause
