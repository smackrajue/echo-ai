"""
Echo AI - Production RAG Chatbot
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

import datetime
from urllib.parse import quote

# Load environment variables
load_dotenv()

# Set User Agent to avoid warnings
os.environ["USER_AGENT"] = "EchoAI/2.0"

# ==================== CONFIGURATION ====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = "echo-ai"
DEMO_FILE_PATH = "demo.txt"
RESET_INTERVAL_HOURS = 24

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
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            # Create index (384 dimensions for all-MiniLM-L6-v2)
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=PINECONE_ENVIRONMENT
                )
            )
            # Wait for index to be ready
            time.sleep(1)
        
        return pc
    except Exception as e:
        st.error(f"Pinecone initialization error: {str(e)}")
        return None

def process_documents(uploaded_files, urls, client_name):
    """
    Process uploaded files and URLs into vector store
    """
    documents = []
    
    # Process uploaded PDFs
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                # Add metadata
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                    doc.metadata['client'] = client_name
                documents.extend(docs)
            except Exception as e:
                st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    # Process URLs
    if urls:
        url_list = [url.strip() for url in urls.split('\n') if url.strip()]
        for url in url_list:
            try:
                # Use a specific headers to avoid blocks
                loader = WebBaseLoader(url)
                # WebBaseLoader uses BeautifulSoup under the hood. 
                # We can add a small timeout indirectly or just hope for the best.
                docs = loader.load()
                if docs:
                    for doc in docs:
                        doc.metadata['source'] = url
                        doc.metadata['client'] = client_name
                    documents.extend(docs)
                else:
                    st.warning(f"URL loaded but no content found: {url}")
            except Exception as e:
                st.warning(f"Could not load {url}: {str(e)}")
    
    if not documents:
        return None, 0
    
    try:
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            return None, 0

        # Initialize Pinecone
        pc = init_pinecone()
        if not pc:
            st.error("Pinecone not initialized. Check API key and environment.")
            return None, 0
        
        # Create vector store with namespace for client separation
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
    """Check if 24 hours passed or demo needs loading"""
    now = datetime.datetime.now()
    
    # Initialize reset time if not set
    if "last_reset_time" not in st.session_state:
        st.session_state.last_reset_time = now
        st.session_state.demo_loaded = False
        st.session_state.demo_progress = 0
        st.session_state.demo_total = 0
    
    # Check 24hr reset
    time_diff = now - st.session_state.last_reset_time
    if time_diff.total_seconds() > (RESET_INTERVAL_HOURS * 3600):
        st.session_state.last_reset_time = now
        st.session_state.demo_loaded = False
        st.session_state.messages = []
        st.session_state.qa_chain = None
        st.session_state.demo_progress = 0
        
    # If already loaded and chain exists, we're good
    if st.session_state.get("demo_loaded", False) and st.session_state.qa_chain:
        return

    # Check via Pinecone stats first to avoid unnecessary reloading
    if not st.session_state.get("demo_loaded", False):
        pc = init_pinecone()
        if pc:
            try:
                index = pc.Index(INDEX_NAME)
                stats = index.describe_index_stats()
                # Check if "demo_user" namespace has significant data
                count = stats.get('namespaces', {}).get('demo_user', {}).get('vector_count', 0)
                if count > 100:
                    st.session_state.demo_loaded = True
                    # Re-initialize chain with existing data
                    vectorstore = get_vectorstore("Demo User")
                    if vectorstore:
                        st.session_state.qa_chain = create_qa_chain(vectorstore, "Smackcoders")
                        st.session_state.client_name = "Smackcoders"
                        return  
            except Exception as e:
                # st.warning(f"Pinecone check skipped: {str(e)}")
                pass

    # Load demo data if needed (Incremental loading to avoid timeouts)
    if not st.session_state.get("demo_loaded", False):
        if os.path.exists(DEMO_FILE_PATH):
            with open(DEMO_FILE_PATH, "r") as f:
                urls = [u.strip() for u in f.read().split('\n') if u.strip()]
            
            total_urls = len(urls)
            st.session_state.demo_total = total_urls
            
            if total_urls > 0:
                current_idx = st.session_state.get("demo_progress", 0)
                
                if current_idx < total_urls:
                    batch_size = 3 # Smaller batch for better responsiveness
                    end_idx = min(current_idx + batch_size, total_urls)
                    batch_urls = urls[current_idx:end_idx]
                    batch_text = "\n".join(batch_urls)
                    
                    with st.sidebar:
                        status_text = f"⏳ Demo Mode: Loading Docs ({end_idx}/{total_urls})"
                        st.info(status_text)
                        st.progress(end_idx / total_urls)
                    
                    # Process batch
                    vectorstore, chunks = process_documents(None, batch_text, "Demo User")
                    
                    # Update progress
                    st.session_state.demo_progress = end_idx
                    
                    if end_idx >= total_urls:
                        # Finalize
                        st.session_state.demo_loaded = True
                        vectorstore = get_vectorstore("Demo User")
                        st.session_state.qa_chain = create_qa_chain(vectorstore, "Smackcoders")
                        st.session_state.client_name = "Smackcoders"
                        st.rerun()
                    else:
                        # Re-run after processing a chunk to update UI and continue
                        st.rerun()
                else:
                    # All URLs processed but demo_loaded still False? 
                    # This shouldn't happen but let's handle it
                    st.session_state.demo_loaded = True
                    vectorstore = get_vectorstore("Demo User")
                    if vectorstore:
                         st.session_state.qa_chain = create_qa_chain(vectorstore, "Smackcoders")
                    st.session_state.client_name = "Smackcoders"
                    st.rerun()

def get_vectorstore(client_name):
    """Load existing vector store from Pinecone"""
    pc = init_pinecone()
    if not pc:
        return None
    
    try:
        embeddings = get_embeddings()
        namespace = client_name.replace(' ', '_').lower()
        
        # Check if namespace has data
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        
        if namespace not in stats.get('namespaces', {}):
            return None
        
        vectorstore = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=namespace
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def create_qa_chain(vectorstore, client_name):
    """Create conversational QA chain"""
    
    # Initialize Groq LLM
    llm = ChatGroq(
        temperature=0.2,
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",  # Fast and free!
        streaming=True
    )
    
    # Custom prompt template
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
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 relevant chunks
        ),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# ==================== STREAMLIT UI ====================

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "client_name" not in st.session_state:
        st.session_state.client_name = "Smackcoders"
    if "mode" not in st.session_state:
        st.session_state.mode = "Demo"

    # Auto-load demo data logic
    check_and_reset_demo()

    # CSS Injection for Sidebar and Layout
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                padding-top: 1rem;
            }
            .block-container {
                padding-top: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar - Document Ingestion
    with st.sidebar:
        st.title("📚 Knowledge Base")
        
        # Mode Selection
        mode = st.radio("Mode", ["Demo", "Custom Upload"], index=0 if st.session_state.mode == "Demo" else 1)
        st.session_state.mode = mode
        
        if mode == "Custom Upload":
            # Client name
            client_name = st.text_input(
                "Client/Company Name",
                value=st.session_state.client_name,
                help="This will be used in the bot's responses"
            )
            st.session_state.client_name = client_name
            
            st.divider()
            
            # Document upload section
            st.subheader("📄 Upload Documents")
            
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload product manuals, FAQs, documentation, etc."
            )
            
            urls_input = st.text_area(
                "Or enter website URLs (one per line)",
                height=100,
                placeholder="https://example.com/docs\nhttps://example.com/faq"
            )
            
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                if not uploaded_files and not urls_input:
                    st.error("Please upload files or enter URLs first!")
                else:
                    with st.spinner("Processing documents... This may take 2-5 minutes..."):
                        vectorstore, num_chunks = process_documents(
                            uploaded_files, 
                            urls_input, 
                            client_name
                        )
                        
                        if vectorstore:
                            st.success(f"✅ Processed {num_chunks} document chunks!")
                            
                            # Create QA chain
                            st.session_state.qa_chain = create_qa_chain(vectorstore, client_name)
                            
                            st.info("You can now chat with your documents!")
                        else:
                            st.error("No documents were processed. Please check your files/URLs.")

            st.divider()
            
            # Load existing knowledge base
            st.subheader("💾 Load Existing")
            
            if st.button("Load Existing Knowledge Base", use_container_width=True):
                vectorstore = get_vectorstore(client_name)
                if vectorstore:
                    st.session_state.qa_chain = create_qa_chain(vectorstore, client_name)
                    st.success("✅ Loaded existing knowledge base!")
                else:
                    st.warning(f"No knowledge base found for '{client_name}'")

        if mode == "Demo":
            st.info("ℹ️ You are in Demo Mode. The system is pre-loaded with sample data.")
            
            # Show progress if loading
            if not st.session_state.get("demo_loaded", False):
                with st.expander("📊 Import Progress", expanded=True):
                    progress = st.session_state.get("demo_progress", 0)
                    total = st.session_state.get("demo_total", 0)
                    st.write(f"URLs Processed: **{progress} / {total}**")
                    if total > 0:
                        st.progress(progress / total)
                    st.info("Please wait while we sync the demo documentation. This happens automatically in the background.")

            if st.button("🔄 Reset to Demo Data", type="primary", use_container_width=True):
                st.session_state.demo_loaded = False
                st.session_state.demo_progress = 0 # Reset progress too
                st.session_state.last_reset_time = datetime.datetime.now() # Reset timer
                st.rerun()

        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Support
        if st.button("📧 Support Email"):
             st.markdown("[Click to Email Support](mailto:rajue@smackcoders.com?subject=EchoAI%20Support)")
        
        # API Key status (Hidden/Collapsed)
        with st.expander("⚙️ Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                if GROQ_API_KEY:
                    st.success("✅ Groq")
                else:
                    st.error("❌ Groq")
            with col2:
                if PINECONE_API_KEY:
                    st.success("✅ Pinecone")
                else:
                    st.error("❌ Pinecone")
        
        # Stats
        if st.session_state.qa_chain:
            st.metric("Status", "Ready", delta="Chat active")
        elif st.session_state.mode == "Demo" and not st.session_state.get("demo_loaded", False):
            progress = st.session_state.get("demo_progress", 0)
            total = st.session_state.get("demo_total", 0)
            st.metric("Status", "Importing...", delta=f"{progress}/{total} URLs")
        else:
            st.metric("Status", "Not Ready", delta="Process docs first")
    
    # Main chat interface
    st.title(f"🔊 Echo AI - {st.session_state.client_name}")
    st.caption("Your intelligent assistant powered by persistent knowledge")
    
    # Suggested Topics (if history is empty)
    if not st.session_state.messages:
        st.subheader("💡 Suggested Topics")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Importing CSV Data", use_container_width=True):
                st.session_state.suggested_prompt = "How do I import CSV data using the plugin?"
        with col2:
            if st.button("Exporting Data", use_container_width=True):
                st.session_state.suggested_prompt = "How can I export my WordPress data?"
        with col3:
            if st.button("API Integration", use_container_width=True):
                st.session_state.suggested_prompt = "Tell me about AI integration features."

    # Handle suggested prompt
    if "suggested_prompt" in st.session_state:
        prompt = st.session_state.suggested_prompt
        del st.session_state.suggested_prompt
        # Trigger same logic as chat input below
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Note: We need to trigger the assistant response logic. 
        # Streamlit's chat_input is separate, so we'll need to refactor slightly 
        # to handle both manual input and suggested input logic.
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.caption(f"{i}. {source}")
    
    # Chat input
    prompt_input = st.chat_input("What would you like to know?")
    
    # Use suggested prompt if set (from above) or user input
    # If the user clicks a suggestion, the message is already appended to history above.
    # We just need to grab the last message if it was a user message and hasn't been answered yet.
    
    process_input = False
    
    if prompt_input:
        prompt = prompt_input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        process_input = True
    elif st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        # This handles the case where we appended a suggested topic message but haven't answered it yet
        prompt = st.session_state.messages[-1]["content"]
        process_input = True

    if process_input:
        # Check if QA chain is ready
        if not st.session_state.qa_chain:
            st.error("⚠️ Please process documents first using the sidebar!")
            st.stop()
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain({
                        "question": prompt
                    })
                    
                    answer = response["answer"]
                    source_docs = response.get("source_documents", [])
                    
                    # Handle NO_ANSWER_FOUND
                    if "NO_ANSWER_FOUND" in answer:
                        answer = "I couldn't find an answer in the documentation. Would you like to contact support?"
                        # Generating mailto link
                        subject = quote(f"Support Request: {prompt[:50]}...")
                        body = quote(f"User Question: {prompt}\n\nContext: No answer found in documentation.")
                        mailto = f"mailto:rajue@smackcoders.com?subject={subject}&body={body}"
                        st.markdown(f"{answer}")
                        st.link_button("📧 Contact Support", mailto)
                        
                        # Add to chat history (modified)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer + " [Use 'Contact Support' button]",
                            "sources": []
                        })
                    else:
                        # Display answer
                        st.markdown(answer)
                        
                        # Show sources
                        if source_docs:
                            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in source_docs]))
                            with st.expander("📖 Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.caption(f"{i}. {source}")
                        else:
                            sources = []
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Footer info
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🤖 Powered by smackcoders")
    with col2:
        st.caption("v2.0 Demo")
    with col3:
        st.link_button("Contact for Full Version", "https://www.smackcoders.com/contact.html")

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
