# 🔊 Echo AI - Production RAG Chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot with **persistent vector storage** using Pinecone.

## 💰 Cost: $0/month

- **Frontend**: Streamlit
- **Vector DB**: **Pinecone** (persistent, cloud-native)
- **Embeddings**: HuggingFace (sentence-transformers)
- **LLM**: Groq API (free tier)
- **Hosting**: HuggingFace Spaces / Streamlit Cloud (free)

## ✨ Key Features

- 📄 **Multi-format ingestion**: Upload PDFs or scrape websites
- 💬 **Conversational AI**: Context-aware responses with memory
- 📚 **Source citations**: Shows which documents answers came from
- 🎯 **Multi-client support**: Separate namespaces per client
- **📌 Persistent storage**: Data survives page refreshes (Pinecone)
- 🚀 **Production-ready**: No data loss, robust architecture

## 🎯 Why Pinecone?

| Feature | ChromaDB (Old) | Pinecone (New) |
|---------|----------------|----------------|
| **Persistence** | ❌ Ephemeral on free hosting | ✅ Cloud-native, always persists |
| **Data Loss** | ❌ Resets on app restart | ✅ Never loses data |
| **Professional** | ⚠️ Demo-quality | ✅ Production-grade |
| **Scalability** | ⚠️ Local file-based | ✅ Serverless, auto-scales |
| **Free Tier** | ✅ Unlimited | ✅ 100K vectors, 1 index |

**Result**: Echo AI feels like a **real product**, not a fragile demo.

---

## 🚀 Quick Start (Local)

### 1. Clone & Setup

```bash
# Navigate to project
cd "e:\My Projects\AI Chatbot"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get API Keys

#### Groq API (Required)
1. Visit: https://console.groq.com
2. Sign up (free)
3. Create API key
4. Copy key (starts with `gsk_...`)

**Free tier**: 30 req/min, 6K req/day

#### Pinecone API (Required)
1. Visit: https://www.pinecone.io
2. Sign up (free)
3. Create API key
4. Note your environment (e.g., `us-east-1`)

**Free tier**: 1 index, 100K vectors, unlimited queries

### 3. Configure Environment

```bash
# Copy example file
cp .env.example .env

# Edit .env and add your keys
GROQ_API_KEY=gsk_your_actual_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1
```

### 4. Run Locally

```bash
streamlit run app.py
```

Opens at: http://localhost:8501

---

## 📦 Deploy to HuggingFace Spaces (FREE)

### Option A: Web Interface

1. Go to: https://huggingface.co/spaces
2. Click "Create new Space"
3. Configure:
   - **Name**: `echo-ai`
   - **SDK**: Streamlit
   - **Hardware**: CPU basic (free)
4. Upload files:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `README_SPACES.md` (rename to `README.md`)
5. Settings → Repository secrets:
   - `GROQ_API_KEY`: Your Groq key
   - `PINECONE_API_KEY`: Your Pinecone key
   - `PINECONE_ENVIRONMENT`: `us-east-1`
6. Wait 5-10 minutes for build

### Option B: Git Push

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/echo-ai
cd echo-ai

# Copy files
cp "e:\My Projects\AI Chatbot\app.py" .
cp "e:\My Projects\AI Chatbot\requirements.txt" .
cp "e:\My Projects\AI Chatbot\packages.txt" .
cp "e:\My Projects\AI Chatbot\README_SPACES.md" README.md

# Commit and push
git add .
git commit -m "Deploy Echo AI"
git push
```

---

## 🎯 Usage

### 1. Upload Documents
- Sidebar → Upload PDF files
- OR enter website URLs (one per line)
- Click "🚀 Process Documents"

### 2. Chat
- Type questions in chat input
- Echo AI answers using ONLY your documents
- Click "📖 Sources" to see references

### 3. Multi-Client Setup
- Change "Client/Company Name"
- Upload different documents
- Each client gets separate Pinecone namespace

**Data persists forever** - no need to re-upload!

---

## 🔧 Customization

### Change LLM Model

```python
# In app.py, line ~185
llm = ChatGroq(
    model_name="llama-3.1-70b-versatile",  # Larger model
    # or "mixtral-8x7b-32768"  # Longer context
)
```

### Adjust Chunk Size

```python
# In app.py, line ~125
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Increase for more context
    chunk_overlap=300,
)
```

### Modify Branding

```python
# In app.py, line ~191
prompt_template = f"""You are Echo AI, a helpful assistant for {client_name}."""
```

---

## 🌐 WordPress Integration

See `wordpress_integration.html` for:
- Full-page iframe embed
- Popup chat widget
- Inline section embed

Update `STREAMLIT_URL` to your deployed Echo AI URL.

---

## 📊 Free Tier Limits

### Groq API
- ✅ 30 requests/minute
- ✅ 6,000 requests/day
- ⚠️ Rate limit errors if exceeded

### Pinecone
- ✅ 1 index (sufficient for all clients via namespaces)
- ✅ 100,000 vectors (~200 PDF pages)
- ✅ Unlimited queries
- ✅ **Persistent storage** (data never deleted)

### HuggingFace Spaces
- ✅ Unlimited usage
- ⚠️ Sleeps after inactivity (~30s wake)
- ✅ **Works with Pinecone** (no local storage needed)

---

## 🔒 Security

- ✅ API keys in secrets (not in code)
- ✅ `.env` excluded from git
- ✅ Pinecone data isolated by namespace
- ⚠️ Free hosting = public apps
- 💡 For sensitive data, use private Space ($9/mo)

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: pinecone"
```bash
pip install -r requirements.txt --upgrade
```

### "Pinecone initialization error"
- Check API key is correct
- Verify environment matches (e.g., `us-east-1`)
- Check Pinecone dashboard for index status

### "Index not found"
- App auto-creates index on first run
- Wait 30 seconds for index creation
- Refresh page

### Slow performance
- Free tier has cold starts (~30s)
- Pinecone queries are fast (<100ms)
- Upgrade hosting for instant wake

---

## 📈 Upgrade Path

### When to upgrade:

**Pinecone Paid ($70/month)**
- Need >100K vectors (>200 pages)
- Need multiple indexes
- Need dedicated resources

**Streamlit Cloud Pro ($20/month)**
- Need private apps
- Need custom domain
- Need instant wake (no cold starts)

**Groq Paid Tier**
- Need higher rate limits
- Need guaranteed uptime

---

## 💡 Pinecone vs ChromaDB

| Scenario | ChromaDB | Pinecone |
|----------|----------|----------|
| **Local dev** | ✅ Great | ✅ Great |
| **Free cloud hosting** | ❌ Data resets | ✅ Persistent |
| **Client demos** | ❌ Breaks on refresh | ✅ Professional |
| **Production** | ⚠️ Need paid hosting | ✅ Ready now |

**Verdict**: Pinecone makes Echo AI production-ready on free tier.

---

## 🤝 Contributing

Contributions welcome!

---

## 📄 License

MIT License - free for commercial use

---

## 🙏 Credits

- **LangChain**: RAG framework
- **Streamlit**: UI framework
- **Groq**: Free LLM API
- **Pinecone**: Vector database
- **HuggingFace**: Embeddings & hosting

---

**Built with ❤️ for production AI demos**

🔊 **Echo AI** - Your knowledge, amplified.
