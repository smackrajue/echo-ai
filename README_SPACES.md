---
title: Echo AI
emoji: 🔊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
license: mit
---

# 🔊 Echo AI - Production RAG Chatbot

Persistent vector storage with Pinecone | Cost: $0/month

## Features
- 📄 Upload PDFs or scrape websites
- 💬 Conversational AI with memory
- 📚 Source citations
- 🎯 Multi-client support (namespaces)
- **📌 Persistent storage** - Data survives refreshes!

## Setup
1. Add secrets in Settings → Repository secrets:
   - `GROQ_API_KEY`: From https://console.groq.com
   - `PINECONE_API_KEY`: From https://www.pinecone.io
   - `PINECONE_ENVIRONMENT`: e.g., `us-east-1`
2. Upload documents using sidebar
3. Start chatting!

## Why Pinecone?
✅ **Persistent** - Data never resets  
✅ **Professional** - Production-ready  
✅ **Free tier** - 100K vectors, unlimited queries

## Tech Stack
- **LLM**: Groq (free)
- **Vector DB**: Pinecone (persistent)
- **Embeddings**: HuggingFace
- **Framework**: LangChain + Streamlit

🔊 **Echo AI** - Your knowledge, amplified.
