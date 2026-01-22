# 🚀 Echo AI - Deployment Checklist

## ✅ Step 1: Get API Keys (You need to do this)

### Groq API Key (2 minutes)
1. Open: **https://console.groq.com**
2. Sign up with GitHub or Google
3. Click "API Keys" → "Create API Key"
4. Copy the key (starts with `gsk_...`)
5. Paste it in `.env` file (line 5)

### Pinecone API Key (3 minutes)
1. Open: **https://www.pinecone.io**
2. Sign up (free account)
3. Verify your email
4. In dashboard, click "API Keys"
5. Copy your API key
6. Note your environment (usually `us-east-1`)
7. Paste both in `.env` file (lines 8-9)

---

## ✅ Step 2: Test Locally (I can help with this)

Once you have API keys in `.env`, run:
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ✅ Step 3: Deploy to HuggingFace Spaces (You need to do this)

### Create Account
1. Go to: **https://huggingface.co**
2. Sign up (free)
3. Verify email

### Create Space
1. Click "Spaces" → "Create new Space"
2. Settings:
   - **Name**: `echo-ai`
   - **SDK**: Streamlit
   - **Hardware**: CPU basic (free)
   - **Visibility**: Public
3. Click "Create Space"

### Upload Files
Click "Files" → "Upload files" → Upload these:
- ✅ `app.py`
- ✅ `requirements.txt`
- ✅ `packages.txt`
- ✅ `demo.txt` (Required for automatic demo data)
- ✅ `README_SPACES.md` (rename to `README.md`)

### Add Secrets
1. Click "Settings" tab
2. Scroll to "Repository secrets"
3. Add three secrets:

**Secret 1:**
- Name: `GROQ_API_KEY`
- Value: (paste your Groq key)

**Secret 2:**
- Name: `PINECONE_API_KEY`
- Value: (paste your Pinecone key)

**Secret 3:**
- Name: `PINECONE_ENVIRONMENT`
- Value: `us-east-1`

### Wait for Build
- Takes 5-10 minutes
- Watch "Logs" tab
- When done, click "App" tab to test

---

## ✅ Step 4: Test Your Deployment

1. Upload a PDF or enter a URL
2. Click "Process Documents"
3. Ask a question
4. **Refresh the page** 🔄
5. Ask the same question
6. ✅ Should still work! (Pinecone persistence)

---

## 🎉 You're Live!

Your URL will be:
```
https://huggingface.co/spaces/YOUR_USERNAME/echo-ai
```

Share this with clients!

---

## ❓ Need Help?

If you get stuck, let me know at which step and I'll help troubleshoot!
