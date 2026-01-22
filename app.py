"""
Echo AI - Version 2.2
Persistent vector storage with Pinecone | Cost: $0/month
"""

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
import os
from pathlib import Path
import tempfile
import time
from dotenv import load_dotenv
import random

import datetime
from urllib.parse import quote

# Load environment variables
load_dotenv()

# Set User Agent to avoid warnings
os.environ["USER_AGENT"] = "EchoAI/2.2"

# ==================== CONFIGURATION ====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = "echo-ai"
DEMO_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.txt")
RESET_INTERVAL_HOURS = 24
PRO_SUPPORT_URL = "mailto:rajue@smackcoders.com?subject=EchoAI%20Support"

# Page config
st.set_page_config(
    page_title="Echo AI - Support Assistant",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def get_embeddings():
    """Load embeddings model (cached to avoid reloading)"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def init_pinecone():
    """Initialize Pinecone client and create index if needed"""
    if not PINECONE_API_KEY:
        return None
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=PINECONE_ENVIRONMENT
                )
            )
            time.sleep(1)
        
        return pc
    except Exception as e:
        st.error(f"Pinecone initialization error: {str(e)}")
        return None

def process_documents(uploaded_files, urls, client_name):
    """Process uploaded files and URLs into vector store"""
    documents = []
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                    doc.metadata['client'] = client_name
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    if urls:
        url_list = [url.strip() for url in urls.split('\n') if url.strip()]
        for url in url_list:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                if docs:
                    for doc in docs:
                        doc.metadata['source'] = url
                        doc.metadata['client'] = client_name
                    documents.extend(docs)
            except Exception as e:
                pass # Silently proceed in batch mode
    
    if not documents:
        return None, 0
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            return None, 0

        pc = init_pinecone()
        if not pc:
            return None, 0
        
        embeddings = get_embeddings()
        namespace = client_name.replace(' ', '_').lower()
        
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=INDEX_NAME,
            namespace=namespace
        )
        
        return vectorstore, len(chunks)
    except Exception as e:
        st.error(f"Error during document indexing: {str(e)}")
        return None, 0

def check_and_reset_demo():
    """Check if demo needs loading or reset"""
    now = datetime.datetime.now()
    
    if "last_reset_time" not in st.session_state:
        st.session_state.last_reset_time = now
        st.session_state.demo_loaded = False
        st.session_state.demo_progress = 0
        st.session_state.demo_total = 0
        if os.path.exists(DEMO_FILE_PATH):
            with open(DEMO_FILE_PATH, "r") as f:
                urls = [u.strip() for u in f.read().split('\n') if u.strip()]
                st.session_state.demo_total = len(urls)
    
    time_diff = now - st.session_state.last_reset_time
    if time_diff.total_seconds() > (RESET_INTERVAL_HOURS * 3600):
        st.session_state.last_reset_time = now
        st.session_state.demo_loaded = False
        st.session_state.messages = []
        st.session_state.qa_chain = None
        st.session_state.demo_progress = 0
        
    if st.session_state.get("demo_loaded", False) and st.session_state.qa_chain:
        return

    # Check Pinecone first
    pc = init_pinecone()
    if pc:
        try:
            index = pc.Index(INDEX_NAME)
            stats = index.describe_index_stats()
            count = stats.get('namespaces', {}).get('demo_user', {}).get('vector_count', 0)
            if count > 100:
                st.session_state.demo_loaded = True
                vectorstore = get_vectorstore("Demo User")
                if vectorstore:
                    st.session_state.qa_chain = create_qa_chain(vectorstore, "Smackcoders")
                    st.session_state.client_name = "Smackcoders"
                    return  
        except Exception:
            pass

    # Incremental loading from file
    if not st.session_state.get("demo_loaded", False):
        urls = []
        if os.path.exists(DEMO_FILE_PATH):
            with open(DEMO_FILE_PATH, "r") as f:
                urls = [u.strip() for u in f.read().split('\n') if u.strip()]
            
            total_urls = len(urls)
            st.session_state.demo_total = total_urls
        else:
            st.sidebar.error("⚠️ Demo source file not found")
            total_urls = 0
            
        if total_urls > 0:
            current_idx = st.session_state.get("demo_progress", 0)
            
            if current_idx < total_urls:
                batch_size = 3
                end_idx = min(current_idx + batch_size, total_urls)
                batch_urls = urls[current_idx:end_idx]
                batch_text = "\n".join(batch_urls)
                
                # Process
                process_documents(None, batch_text, "Demo User")
                st.session_state.demo_progress = end_idx
                
                if end_idx >= total_urls:
                    st.session_state.demo_loaded = True
                    vectorstore = get_vectorstore("Demo User")
                    st.session_state.qa_chain = create_qa_chain(vectorstore, "Smackcoders")
                    st.session_state.client_name = "Smackcoders"
                
                st.rerun()

def get_vectorstore(client_name):
    """Load existing vector store from Pinecone"""
    pc = init_pinecone()
    if not pc: return None
    
    try:
        embeddings = get_embeddings()
        namespace = client_name.replace(' ', '_').lower()
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        
        if namespace not in stats.get('namespaces', {}):
            return None
        
        return PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=namespace
        )
    except Exception as e:
        return None

def create_qa_chain(vectorstore, client_name):
    """Create conversational QA chain"""
    llm = ChatGroq(
        temperature=0.2,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        streaming=True
    )
    
    prompt_template = f"""You are Echo AI, a helpful AI assistant for {client_name}.

Use the following pieces of context to answer the question at the end. 

IMPORTANT RULES:
1. ONLY use information from the provided context below
2. If you cannot find the answer in the context, say "NO_ANSWER_FOUND"
3. Be friendly, professional, and concise
4. If you're not completely sure, say so
5. When helpful, mention which document/section the answer comes from

Context: {{context}}

Question: {{question}}

Helpful Answer:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

def handle_pro_click(feature_name):
    """Display Pro message and trigger mailto"""
    st.toast("🚀 Pro Custom Options: Contact developer for more options.", icon="⚠️")
    time.sleep(1)
    st.markdown(f'<meta http-equiv="refresh" content="0; url={PRO_SUPPORT_URL}">', unsafe_allow_html=True)

def main():
    if "messages" not in st.session_state: st.session_state.messages = []
    if "qa_chain" not in st.session_state: st.session_state.qa_chain = None
    if "client_name" not in st.session_state: st.session_state.client_name = "Smackcoders"
    if "mode" not in st.session_state: st.session_state.mode = "Demo"
    if "cta_links" not in st.session_state: st.session_state.cta_links = "🔥 Get Pro Version: https://www.smackcoders.com/echo-ai\n📦 Buy Bundle: https://www.smackcoders.com/bundle"
    if "upsell_links" not in st.session_state: st.session_state.upsell_links = "⚡ Upgrade to Enterprise: Contact Support"

    check_and_reset_demo()

    # CSS Injection: Persistent Scrollbar, Move Status, Anchor Links
    st.markdown("""
        <style>
            /* Make scrollbar wider and always visible */
            ::-webkit-scrollbar {
                width: 12px !important;
                height: 12px !important;
            }
            ::-webkit-scrollbar-track {
                background: #f1f1f1 !important;
            }
            ::-webkit-scrollbar-thumb {
                background: #888 !important;
                border-radius: 6px !important;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #555 !important;
            }
            html {
                scrollbar-width: thick !important;
                scrollbar-color: #888 #f1f1f1 !important;
            }
            
            [data-testid="stSidebar"] {
                padding-top: 1rem;
            }
            .block-container {
                padding-top: 2rem;
            }
            
            .bottom-anchor {
                text-align: right;
                font-size: 0.8rem;
                margin-top: 1rem;
            }
            
            /* CTA Styling */
            .cta-box {
                background: #f0f7ff;
                border-left: 4px solid #007bff;
                padding: 10px;
                margin: 10px 0;
                border-radius: 4px;
                font-size: 0.9rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        # Status Metric
        if st.session_state.qa_chain:
            st.metric("Status", "Ready", delta="Chat active")
        elif st.session_state.mode == "Demo" and not st.session_state.get("demo_loaded", False):
            progress = st.session_state.get("demo_progress", 0)
            total = st.session_state.get("demo_total", 0)
            st.metric("Status", "Importing...", delta=f"{progress}/{total} URLs")
        else:
            st.metric("Status", "Not Ready", delta="Process docs first")

        st.title("📚 Knowledge Base")
        st.divider()

        mode = st.radio("Mode", ["Demo", "Custom Upload"], index=0 if st.session_state.mode == "Demo" else 1)
        st.session_state.mode = mode
        
        if mode == "Custom Upload":
            client_name = st.text_input("Client/Company Name", value=st.session_state.client_name)
            st.session_state.client_name = client_name
            st.divider()
            uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
            urls_input = st.text_area("Or enter website URLs (one per line)", height=100)
            
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                if not uploaded_files and not urls_input:
                    st.error("Please upload files or enter URLs first!")
                else:
                    with st.spinner("Processing documents..."):
                        vectorstore, num_chunks = process_documents(uploaded_files, urls_input, client_name)
                        if vectorstore:
                            st.success(f"✅ Processed {num_chunks} document chunks!")
                            st.session_state.qa_chain = create_qa_chain(vectorstore, client_name)
                        else:
                            st.error("No documents were processed.")

            st.divider()
            if st.button("Load Existing Knowledge Base", use_container_width=True):
                vectorstore = get_vectorstore(client_name)
                if vectorstore:
                    st.session_state.qa_chain = create_qa_chain(vectorstore, client_name)
                    st.success("✅ Loaded!")
                else:
                    st.warning(f"No database for '{client_name}'")

        if mode == "Demo":
            st.info("ℹ️ Demo Mode: Sample data active.")
            if not st.session_state.get("demo_loaded", False):
                with st.expander("📊 Import Progress", expanded=True):
                    progress = st.session_state.get("demo_progress", 0)
                    total = st.session_state.get("demo_total", 0)
                    st.write(f"Synced: **{progress} / {total}** URLs")
                    if total > 0: st.progress(progress / total)
                    st.caption("Syncing in background...")

            if st.button("🔄 Reset to Demo Data", type="primary", use_container_width=True):
                st.session_state.demo_loaded = False
                st.session_state.demo_progress = 0
                st.session_state.last_reset_time = datetime.datetime.now()
                st.rerun()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # 🚀 PRO SETTINGS PANEL
        with st.expander("🚀 Pro Settings (Custom Options)", expanded=False):
            st.subheader("CTA Configuration")
            st.session_state.cta_links = st.text_area("Promoted CTA Links (config)", value=st.session_state.cta_links, height=80)
            
            st.subheader("Monetization")
            st.session_state.upsell_links = st.text_input("Upsell/Cross-sell Config", value=st.session_state.upsell_links)
            
            if st.button("🎟️ Integrate WP Coupons", use_container_width=True):
                handle_pro_click("Coupons")
            
            if st.button("📊 Analytics Dashboard", use_container_width=True):
                handle_pro_click("Analytics")

        st.divider()
        if st.button("📧 Support Email", use_container_width=True):
             st.markdown(f"[Click to Email Support]({PRO_SUPPORT_URL})")
        
        with st.expander("⚙️ Configuration"):
            c1, c2 = st.columns(2)
            with c1: st.write("Groq: " + ("✅" if GROQ_API_KEY else "❌"))
            with c2: st.write("Pinecone: " + ("✅" if PINECONE_API_KEY else "❌"))

    # Main UI
    st.title(f"🔊 Echo AI - {st.session_state.client_name}")
    st.caption("Your intelligent assistant powered by persistent knowledge")
    
    # Suggested Topics
    if not st.session_state.messages:
        st.subheader("💡 Suggested Topics")
        topics = [
            ("Importing CSV Data", "How do I import CSV data?"),
            ("Exporting Data", "How can I export my WordPress data?"),
            ("API Integration", "Tell me about API integration Features."),
            ("WooCommerce Sync", "How to sync WooCommerce products?"),
            ("Troubleshooting", "Common issues during import?"),
            ("AI Features", "How does AI integration work?")
        ]
        cols = st.columns(3)
        for i, (label, prompt) in enumerate(topics[:6]):
            with cols[i % 3]:
                if st.button(label, use_container_width=True, key=f"sug_{i}"):
                    st.session_state.suggested_prompt = prompt
        
        # More topics link
        if st.link_button("🔗 More topics...", PRO_SUPPORT_URL, use_container_width=True):
            st.toast("Pro Custom Options: Contact for full topic list.", icon="ℹ️")

    # Handle suggestions
    if "suggested_prompt" in st.session_state:
        prompt = st.session_state.suggested_prompt
        del st.session_state.suggested_prompt
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.caption(f"{i}. {source}")
            if message["role"] == "assistant" and "cta" in message:
                st.markdown(f'<div class="cta-box">{message["cta"]}</div>', unsafe_allow_html=True)

    # Chat Input
    prompt_input = st.chat_input("What would you like to know?")
    
    active_prompt = None
    if prompt_input:
        active_prompt = prompt_input
    elif st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and (len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] == "assistant"):
        active_prompt = st.session_state.messages[-1]["content"]

    if active_prompt:
        if not st.session_state.qa_chain:
            st.error("⚠️ Please process documents first using the sidebar!")
        else:
            if prompt_input: 
                with st.chat_message("user"): st.markdown(active_prompt)
                st.session_state.messages.append({"role": "user", "content": active_prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.qa_chain({"question": active_prompt})
                        answer = response["answer"]
                        source_docs = response.get("source_documents", [])
                        
                        if "NO_ANSWER_FOUND" in answer:
                            answer = "I couldn't find an answer in documentation. How would you like to proceed?"
                            st.markdown(answer)
                            c1, c2 = st.columns(2)
                            with c1:
                                mailto_support = f"mailto:rajue@smackcoders.com?subject=Support%20Request&body=Question:%20{quote(active_prompt)}"
                                st.link_button("📧 Contact Support", mailto_support, use_container_width=True)
                            with c2:
                                if st.button("🎫 Create as a Ticket", use_container_width=True):
                                    handle_pro_click("Ticketing")
                        else:
                            st.markdown(answer)
                            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in source_docs]))
                            
                            # Intelligent CTA injection (1 in 3 chance or if historical)
                            cta_to_add = None
                            if random.random() < 0.4:
                                links = [l.strip() for l in st.session_state.cta_links.split('\n') if l.strip()]
                                if links: cta_to_add = random.choice(links)

                            msg_obj = {"role": "assistant", "content": answer}
                            if sources: 
                                msg_obj["sources"] = sources
                                with st.expander("📖 Sources"):
                                    for i, s in enumerate(sources, 1): st.caption(f"{i}. {s}")
                            
                            if cta_to_add:
                                msg_obj["cta"] = cta_to_add
                                st.markdown(f'<div class="cta-box">{cta_to_add}</div>', unsafe_allow_html=True)
                            
                            st.session_state.messages.append(msg_obj)
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # Bottom Link and Footer
    st.markdown('<div id="bottom"></div>', unsafe_allow_html=True)
    st.markdown('<p class="bottom-anchor"><a href="#top">↑ Back to Top</a></p>', unsafe_allow_html=True)
    
    st.divider()
    fc1, fc2, fc3 = st.columns(3)
    with fc1: st.caption("🤖 Powered by smackcoders")
    with fc2: st.caption("v2.2 Pro-Lite")
    with fc3: st.link_button("Contact for Full Version", PRO_SUPPORT_URL)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
