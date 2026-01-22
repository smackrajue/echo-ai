# 🔊 Echo AI - Product Documentation

## 1. Introduction
Echo AI is a production-ready Retrieval-Augmented Generation (RAG) chatbot designed to provide intelligent, context-aware answers based on private documentation. It bridges the gap between static knowledge bases and interactive AI by allowing users to upload PDFs or crawl websites and chat with that data in real-time.

## 2. Key Features
*   **Persistent Knowledge Base**: Powered by Pinecone vector database, ensuring your data survives application restarts.
*   **Multi-Source Ingestion**: Support for local PDF file uploads and remote website URL scraping.
*   **Automated Demo Mode**: Pre-loaded with professional documentation to allow immediate testing without configuration.
*   **Source Citations**: Every answer includes clickable or expandable sources to verify the information.
*   **Conversational Memory**: Remembers previous questions and context within a session for natural dialogue.
*   **Smart Fallback**: If an answer isn't found in the knowledge base, it provides a direct link to contact professional support.

## 3. Architecture Overview
Echo AI utilizes a modern AI stack for speed and cost-effectiveness:
- **Large Language Model (LLM)**: [Groq](https://groq.com/) - Utilizing Llama 3.1 8B for lightning-fast inference.
- **Vector Database**: [Pinecone](https://www.pinecone.io/) - Serverless vector storage for high-performance retrieval.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` - Efficient text vectorization.
- **Orchestration**: [LangChain](https://www.langchain.com/) - Managing the RAG pipeline, memory, and prompts.
- **Frontend**: [Streamlit](https://streamlit.io/) - A responsive, interactive web interface.

## 4. User Guide

### 4.1 Demo Mode
By default, Echo AI starts in **Demo Mode**.
1.  On the first run, it imports URLs from `demo.txt`.
2.  The sidebar shows the **Status: Importing...** with a progress counter.
3.  Once ready, the status changes to **Ready**.
4.  Users can click on **Suggested Topics** to see the AI in action immediately.

### 4.2 Custom Upload Mode
To use Echo AI with your own data:
1.  Switch the **Mode** in the sidebar to "Custom Upload".
2.  Enter your **Client/Company Name** (this creates a unique namespace in the database).
3.  Upload PDF files or paste a list of URLs (one per line).
4.  Click **Process Documents**.
5.  Wait for the success message: "Processed X document chunks!".

### 4.3 Support Integration
Echo AI is designed to minimize support load. However, when it cannot find an answer:
- It responds with a professional "I couldn't find an answer..." message.
- It displays a **📧 Contact Support** button that pre-fills an email with the user's question.

### 4.4 Pro Customization (v2.2)
Echo AI includes a **Pro Settings** panel for advanced business integration:
- **Intelligent CTA Injection**: Configure promotional links; the AI will occasionally (randomly) inject these as stylized CTA boxes in responses.
- **Ticketing Flow**: If no answer is found, users see a "Create as a Ticket" option, serving as a placeholder for enterprise helpdesk integration.
- **Monetization Placeholders**: Built-in support for WordPress Coupons and Analytics dashboard concepts.
- **Suggest Topic Expansion**: Curated startup topics are limited to 6, with a link to "More topics" leading to developer support.

---

## 5. Technical Configuration

### 5.1 Prerequisites
- Python 3.9+
- Groq API Key
- Pinecone API Key & Environment

### 5.2 Environment Variables
Create a `.env` file or set secrets in your hosting environment:
| Key | Description |
| :--- | :--- |
| `GROQ_API_KEY` | Your API key from Groq Console |
| `PINECONE_API_KEY` | Your API key from Pinecone Dashboard |
| `PINECONE_ENVIRONMENT` | The region of your Pinecone index (e.g., `us-east-1`) |

### 5.3 Deployment
Echo AI is optimized for deployment on **Hugging Face Spaces** or **Streamlit Cloud**.
- Use the included `Dockerfile` for containerized environments.
- Ensure `requirements.txt` is installed.

## 6. Maintenance
- **Clearing History**: Use the "Clear Chat History" button in the sidebar to reset the current session's memory.
- **Resetting Demo**: Use the "Reset to Demo Data" button to clear local session state and re-sync demo information if needed.

---
*Created by Smackcoders | Version 2.1*
