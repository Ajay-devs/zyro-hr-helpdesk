"""
🏢 Zyro Dynamics HR Help Desk - Streamlit Chatbot App

A premium, modern chatbot interface for employees to ask HR policy questions.
Built with Streamlit, powered by RAG (FAISS + Groq LLM).
"""
import streamlit as st
import os
import time

# ─── Page Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Acrux Dynamics (Zyro Dynamics) HR Help Desk",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for Premium Dark Theme ─────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Styles ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main Header ── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.8rem !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }

    /* ── Source Citation Card ── */
    .source-card {
        background: linear-gradient(135deg, rgba(102,126,234,0.08), rgba(118,75,162,0.06));
        border: 1px solid rgba(102,126,234,0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        border-color: rgba(102,126,234,0.5);
        box-shadow: 0 4px 12px rgba(102,126,234,0.15);
    }
    .source-card .source-title {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .source-card .source-preview {
        color: rgba(255,255,255,0.6);
        font-size: 0.8rem;
        line-height: 1.4;
    }

    /* ── Sidebar Styles ── */
    .sidebar-section {
        background: linear-gradient(135deg, rgba(102,126,234,0.1), rgba(118,75,162,0.05));
        border: 1px solid rgba(102,126,234,0.15);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .sidebar-section h3 {
        color: #667eea;
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.8rem;
    }

    /* ── Quick Action Buttons ── */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
        border: 1px solid rgba(102,126,234,0.3) !important;
    }
    .stButton > button:hover {
        border-color: #667eea !important;
        box-shadow: 0 4px 12px rgba(102,126,234,0.2) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Status Badges ── */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-online {
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 1rem;
        color: rgba(255,255,255,0.3);
        font-size: 0.75rem;
        margin-top: 2rem;
    }

    /* ── Animated Dots ── */
    @keyframes pulse {
        0%, 80%, 100% { opacity: 0.3; }
        40% { opacity: 1; }
    }
    .typing-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #667eea;
        margin: 0 2px;
        animation: pulse 1.4s infinite;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Session State ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ─── Pipeline Initialization ───────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_pipeline():
    """Initialize the RAG pipeline (cached so it only runs once)."""
    from rag_pipeline import initialize_pipeline
    rag_chain, retriever, vectorstore = initialize_pipeline()
    return rag_chain, retriever


def ensure_pipeline():
    """Ensure the pipeline is loaded."""
    if not st.session_state.pipeline_ready:
        with st.spinner("🔄 Initializing HR Knowledge Base... This may take a moment on first load."):
            rag_chain, retriever = init_pipeline()
            st.session_state.rag_chain = rag_chain
            st.session_state.retriever = retriever
            st.session_state.pipeline_ready = True


# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 800;">
            🏢 Acrux Dynamics
        </h2>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.3rem;">
            HR Help Desk Assistant
        </p>
        <div style="margin-top: 0.5rem;">
            <span class="status-badge status-online">● Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Quick Topics
    st.markdown("""
    <div class="sidebar-section">
        <h3>💡 Quick Topics</h3>
    </div>
    """, unsafe_allow_html=True)

    quick_questions = {
        "📋 Leave Policy": "What are the different types of leaves available and their entitlements?",
        "🏠 Work From Home": "What is the work from home policy at Acrux Dynamics?",
        "💰 Compensation": "Explain the compensation and benefits structure at Acrux Dynamics.",
        "📊 Performance Review": "How does the performance review process work?",
        "🛡️ Code of Conduct": "What are the key points of the code of conduct?",
        "🔒 IT Security": "What are the IT and data security policies?",
        "✈️ Travel & Expenses": "What is the travel and expense reimbursement policy?",
        "👋 Onboarding": "What does the onboarding process look like for new employees?",
        "⚖️ POSH Policy": "What is the Prevention of Sexual Harassment policy?",
        "🏢 Company Profile": "Tell me about Acrux Dynamics as a company.",
    }

    for label, question in quick_questions.items():
        if st.button(label, use_container_width=True, key=f"quick_{label}"):
            st.session_state.quick_question = question

    st.divider()

    # About Section
    st.markdown("""
    <div class="sidebar-section">
        <h3>ℹ️ About</h3>
        <p style="font-size: 0.82rem; color: rgba(255,255,255,0.6); line-height: 1.6;">
            I'm your AI-powered HR assistant trained on all
            <strong>11 HR policy documents</strong> for Acrux Dynamics (also known as Zyro Dynamics).
            I can help you with questions about:
        </p>
        <ul style="font-size: 0.8rem; color: rgba(255,255,255,0.5); line-height: 1.8;">
            <li>Leave policies & entitlements</li>
            <li>Work from home guidelines</li>
            <li>Compensation & benefits</li>
            <li>Performance reviews</li>
            <li>IT & data security</li>
            <li>Code of conduct</li>
            <li>Travel & expense claims</li>
            <li>Onboarding & separation</li>
            <li>POSH policy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Clear Chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Footer
    st.markdown("""
    <div class="footer">
        Powered by RAG | FAISS + Groq LLM<br>
        Built for Acrux Dynamics / Zyro Dynamics
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content Area ─────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🏢 HR Help Desk Assistant</h1>
    <p>Ask me anything about Acrux Dynamics (Zyro Dynamics) HR policies — I'm here to help!</p>
</div>
""", unsafe_allow_html=True)

# Initialize pipeline
ensure_pipeline()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

        # Show sources if available
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 View Source Citations", expanded=False):
                for src in message["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">📄 {src['source']} — Page {src['page']}</div>
                        <div class="source-preview">{src['content_preview']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Handle quick question from sidebar
if "quick_question" in st.session_state:
    quick_q = st.session_state.pop("quick_question")
    st.session_state.messages.append({"role": "user", "content": quick_q})
    with st.chat_message("user", avatar="👤"):
        st.markdown(quick_q)

    # Get answer
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching HR policies..."):
            from rag_pipeline import ask_question
            result = ask_question(
                quick_q,
                st.session_state.rag_chain,
                st.session_state.retriever,
            )

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📚 View Source Citations", expanded=False):
                for src in result["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">📄 {src['source']} — Page {src['page']}</div>
                        <div class="source-preview">{src['content_preview']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
    st.rerun()

# Chat Input
if prompt := st.chat_input("Ask me about HR policies... (e.g., 'What is the leave policy?')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching HR policies..."):
            from rag_pipeline import ask_question
            result = ask_question(
                prompt,
                st.session_state.rag_chain,
                st.session_state.retriever,
            )

        # Typewriter effect for the answer
        answer_placeholder = st.empty()
        displayed = ""
        for char in result["answer"]:
            displayed += char
            answer_placeholder.markdown(displayed + "▌")
            time.sleep(0.005)
        answer_placeholder.markdown(result["answer"])

        # Show sources
        if result["sources"]:
            with st.expander("📚 View Source Citations", expanded=True):
                for src in result["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">📄 {src['source']} — Page {src['page']}</div>
                        <div class="source-preview">{src['content_preview']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        if result["is_out_of_scope"]:
            st.info("💡 This question is outside the scope of HR policies. Try asking about leave, compensation, WFH, or other HR topics!")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
