# app.py

import base64
import os
import shutil
import time
import streamlit as st
from PIL import Image
from src.ingest import build_vector_db
from src.rag_pipeline import get_rag_chain
from src.config import PDF_FOLDER, CHROMA_PERSIST_DIR



# ============================================================
# LOGO LOADER (base64 encode for HTML use)
# ============================================================
def get_base64_image(image_path: str) -> str:
    """Load an image and return it as a base64 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""


LOGO_PATH = "assets/logo.jpg"
LOGO_B64 = get_base64_image(LOGO_PATH)


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon=Image.open("assets/logo.jpg"),
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS (Modern AI Assistant Look)
# ============================================================
st.markdown("""
<style>
    /* ---------- Global ---------- */
    .stApp {
        background: radial-gradient(circle at 20% 0%, #1a1033 0%, #0F0F14 45%) fixed;
    }
    
    /* ---------- Hide default menu/footer, but KEEP header for sidebar toggle ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    header[data-testid="stHeader"] {
        background: transparent;
        height: 2.5rem;
    }

    /* ---------- Sidebar Toggle Button (visible when collapsed) ---------- */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        color: #A78BFA !important;
        background: rgba(124, 58, 237, 0.15) !important;
        border: 1px solid rgba(124, 58, 237, 0.4) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        margin: 8px !important;
        transition: all 0.2s ease;
    }

    [data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="collapsedControl"]:hover {
        background: rgba(124, 58, 237, 0.35) !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.5);
    }
    
    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14141C 0%, #0F0F14 100%);
        border-right: 1px solid #26263A;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }
    
    /* ---------- Headings ---------- */
    h1, h2, h3 {
        color: #F3F4F6 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* ---------- Hero Header ---------- */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A78BFA 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .hero-sub {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    
    /* ---------- Sidebar Brand ---------- */
    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.6rem 0.4rem 1rem 0.4rem;
        border-bottom: 1px solid #26263A;
        margin-bottom: 1rem;
    }
    
    .brand-logo {
        width: 44px; height: 44px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(124, 58, 237, 0.55);
        flex-shrink: 0;
    }

    .brand-logo img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 12px;
    }
    
    .brand-text {
        font-weight: 700;
        color: #F3F4F6;
        font-size: 1rem;
        line-height: 1.1;
    }
    
    .brand-sub {
        font-size: 0.72rem;
        color: #9CA3AF;
    }
    
    /* ---------- Section labels ---------- */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #9CA3AF;
        margin: 1.2rem 0 0.5rem 0;
    }
    
    /* ---------- Status card ---------- */
    .status-card {
        background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(236,72,153,0.08));
        border: 1px solid rgba(124,58,237,0.35);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #E5E7EB;
    }
    
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 10px #10B981;
        margin-right: 8px;
    }
    
    .status-dot.off {
        background: #EF4444;
        box-shadow: 0 0 10px #EF4444;
    }
    
    /* ---------- Chat bubbles ---------- */
    .chat-row {
        display: flex;
        margin: 0.7rem 0;
        animation: fadeIn 0.35s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    
    .chat-row.user { justify-content: flex-end; }
    .chat-row.bot  { justify-content: flex-start; }
    
    .bubble {
        max-width: 78%;
        padding: 0.85rem 1.1rem;
        border-radius: 16px;
        line-height: 1.55;
        font-size: 0.95rem;
        word-wrap: break-word;
    }
    
    .bubble.user {
        background: linear-gradient(135deg, #7C3AED, #A855F7);
        color: #FFFFFF;
        border-bottom-right-radius: 4px;
        box-shadow: 0 6px 24px rgba(124,58,237,0.35);
    }
    
    .bubble.bot {
        background: #1A1A24;
        color: #E5E7EB;
        border: 1px solid #26263A;
        border-bottom-left-radius: 4px;
    }
    
    .avatar {
        width: 32px; height: 32px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px;
        flex-shrink: 0;
        margin: 0 8px;
    }
    
    .avatar.user {
        background: linear-gradient(135deg, #EC4899, #F59E0B);
    }
    
    .avatar.bot {
        background: transparent;
        overflow: hidden;
    }

    .avatar.bot img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 50%;
    }
    
    /* ---------- Buttons ---------- */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #7C3AED, #A855F7);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(124,58,237,0.45);
    }
    
    /* Secondary (Clear) button */
    .stButton > button[kind="secondary"] {
        background: #1A1A24;
        border: 1px solid #26263A;
        color: #E5E7EB;
    }
    
    /* ---------- File uploader ---------- */
    [data-testid="stFileUploader"] {
        background: #14141C;
        border: 1px dashed #3F3F5A;
        border-radius: 12px;
        padding: 0.6rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #7C3AED;
    }
    
    /* ---------- Chat input ---------- */
    [data-testid="stChatInput"] {
        background: #1A1A24;
        border: 1px solid #26263A;
        border-radius: 14px;
    }
    
    [data-testid="stChatInput"]:focus-within {
        border-color: #7C3AED;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.2);
    }
    
    /* ---------- Alert boxes ---------- */
    .stAlert {
        border-radius: 10px;
        border: 1px solid #26263A;
    }
    
    /* ---------- Divider ---------- */
    hr {
        border-color: #26263A;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = os.path.exists(CHROMA_PERSIST_DIR)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(str(LOGO_PATH), width=44)
    with col2:
        st.markdown("""
        <div style="padding-top:6px;">
            <div class="brand-text">AI PDF Assistant</div>
            <div class="brand-sub">Powered by Ollama</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Upload Section
    st.markdown('<div class="section-label">Upload Document</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        label_visibility="collapsed",
    )
    
    if uploaded_file is not None:
        os.makedirs(PDF_FOLDER, exist_ok=True)
        pdf_path = os.path.join(PDF_FOLDER, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ {uploaded_file.name} uploaded")
    
    # Process Button
    if st.button("⚡ Process Documents", use_container_width=True):
        with st.spinner("Building embeddings..."):
            vectorstore = build_vector_db()
            if vectorstore:
                st.session_state.pdf_ready = True
                st.success("Documents are ready!")
            else:
                st.error("No PDF found. Please upload a file first.")
    
    # Clear Button
    if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)
        st.session_state.pdf_ready = False
        st.session_state.messages = []
        st.success("Database cleared.")
    
    # Status Card
    st.markdown('<div class="section-label">📊 Status</div>', unsafe_allow_html=True)
    
    if st.session_state.pdf_ready:
        st.markdown("""
        <div class="status-card">
            <span class="status-dot"></span>
            Knowledge Base Active
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card">
            <span class="status-dot off"></span>
            No Document Loaded
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Clear Chat Button
    if st.button("💬 New Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# MAIN AREA
# ============================================================

# Hero header
st.markdown("""
<div class="hero-title">AI PDF Assistant</div>
<div class="hero-sub">Ask anything about your PDF — every answer comes straight from your document.</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='margin: 0.5rem 0 1rem 0;'>", unsafe_allow_html=True)


# ---------- Welcome / Empty State ----------
if not st.session_state.messages:
    st.markdown("""
    <div style="
        text-align:center;
        padding: 2.5rem 1rem;
        color:#9CA3AF;
    ">
        <div style="font-size: 3rem;">💬</div>
        <div style="font-size: 1.1rem; color:#E5E7EB; font-weight:600; margin-top:0.5rem;">
            Ask a question to get started
        </div>
        <div style="font-size: 0.85rem; margin-top: 0.4rem;">
            Try: "What skills are listed in this resume?" or "Give me a summary of this document."
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------- Chat History ----------
def render_message(role: str, content: str):
    if role == "user":
        st.markdown(f"""
        <div class="chat-row user">
            <div class="bubble user">{content}</div>
            <div class="avatar user">🧑</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-row bot">
            <div class="avatar bot">
                <img src="data:image/logo.jpg;base64,{LOGO_B64}" alt="Logo">
            </div>
            <div class="bubble bot">{content}</div>
        </div>
        """, unsafe_allow_html=True)


for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])


# ---------- Chat Input ----------
prompt = st.chat_input("Ask a question about your PDF...")

if prompt:
    if not st.session_state.pdf_ready:
        st.warning("⚠️ Please upload a PDF and click 'Process Documents' first.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_message("user", prompt)
        
        # Assistant response (typing indicator)
        with st.spinner("🤖 Thinking..."):
            try:
                chain = get_rag_chain()
                response = chain.invoke(prompt)
            except Exception as e:
                response = f"⚠️ Error: {str(e)}"
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        render_message("assistant", response)