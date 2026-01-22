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

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = "echo-ai"

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
            finally:
                os.unlink(tmp_path)  # Clean up temp file
    
    # Process URLs
    if urls:
        url_list = [url.strip() for url in urls.split('\n') if url.strip()]
        for url in url_list:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = url
                    doc.metadata['client'] = client_name
                documents.extend(docs)
            except Exception as e:
                st.warning(f"Could not load {url}: {str(e)}")
    
    if not documents:
        return None, 0
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # Initialize Pinecone
    pc = init_pinecone()
    if not pc:
        st.error("Pinecone not initialized. Check API key.")
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
2. If you cannot find the answer in the context, say "I don't have that information in my knowledge base. Please contact our support team."
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
        st.session_state.client_name = "Demo Company"
    
    # Sidebar - Document Ingestion
    with st.sidebar:
        st.title("📚 Knowledge Base Setup")
        
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
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # API Key status
        st.subheader("🔑 Configuration")
        
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
        else:
            st.metric("Status", "Not Ready", delta="Process docs first")
    
    # Main chat interface
    st.title(f"� Echo AI - {client_name}")
    st.caption("Your intelligent assistant powered by persistent knowledge")
    
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
    if prompt := st.chat_input("What would you like to know?"):
        # Check if QA chain is ready
        if not st.session_state.qa_chain:
            st.error("⚠️ Please process documents first using the sidebar!")
            st.stop()
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain({
                        "question": prompt
                    })
                    
                    answer = response["answer"]
                    source_docs = response.get("source_documents", [])
                    
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
        st.caption("🤖 Powered by Groq")
    with col2:
        st.caption("🧠 HuggingFace Embeddings")
    with col3:
        st.caption("� Pinecone Vector DB")

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
