


# app.py
# Streamlit chat interface for Aravindh's AI Resume Chatbot.
# Run locally with: streamlit run app.py
 
import streamlit as st
import qrcode
import io
from chatbot import build_rag_chain
 
# ─── Page config ─────────────────────────────────────────────────────────────
 
st.set_page_config(
    page_title="Aravindh's AI Resume Assistant",
    page_icon="🤖",
    layout="centered"
)
 
# ─── Header ──────────────────────────────────────────────────────────────────
 
st.title("🤖 Aravindh's AI Resume Assistant")
st.markdown(
    "Ask me anything about **Aravindh S** — his skills, education, "
    "projects, experience, strengths, or career goals. "
    "I'll respond in a professional, interview-ready style."
)
st.divider()
 
# ─── Suggested questions ─────────────────────────────────────────────────────
 
st.markdown("**💡 Try asking:**")
suggested = [
    "Tell me about Aravindh",
    "What are his technical skills?",
    "What projects has he built?",
    "Why should we hire him?",
    "What are his career goals?",
    "What are his strengths?",
    "Describe his personality",
    "What is his final year project?",
    "What are his hobbies?",
    "Where does he see himself in 5 years?"
]
 
cols = st.columns(3)
for i, q in enumerate(suggested):
    if cols[i % 3].button(q, use_container_width=True):
        st.session_state["prefill_question"] = q
 
st.divider()
 
# ─── Load RAG chain once (cached across all sessions) ────────────────────────
 
@st.cache_resource
def load_chain():
    with st.spinner("🔄 Loading AI model... please wait a moment"):
        return build_rag_chain()
 
chain = load_chain()
 
# ─── Chat history ─────────────────────────────────────────────────────────────
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Render all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
# ─── Handle suggestion button click ──────────────────────────────────────────
 
prefill = st.session_state.pop("prefill_question", None)
 
# ─── Chat input ───────────────────────────────────────────────────────────────
 
user_input = st.chat_input("Ask about Aravindh...") or prefill
 
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    # Stream the AI answer word by word
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
 
        for chunk in chain.stream(user_input):
            full_response += chunk
            placeholder.markdown(full_response + "▌")  # typing cursor effect
 
        placeholder.markdown(full_response)  # final clean render
 
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })
 
# ─── Clear chat button ────────────────────────────────────────────────────────
 
if st.session_state.messages:
    if st.button("🗑️ Clear chat", use_container_width=False):
        st.session_state.messages = []
        st.rerun()
 
# ─── QR Code section ─────────────────────────────────────────────────────────
 
st.divider()
 
with st.expander("📱 Share this chatbot — Scan QR Code"):
    # Replace this URL with your actual Streamlit Cloud URL after deployment
    app_url = st.text_input(
        "Your deployed app URL",
        value="https://your-app-name.streamlit.app",
        help="Paste your Streamlit Cloud URL here after deploying"
    )
 
    if app_url and app_url != "https://your-app-name.streamlit.app":
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=8,
            border=2
        )
        qr.add_data(app_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
 
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
 
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(buf, width=180, caption="Scan to open chatbot")
        with col2:
            st.markdown("**Share this link:**")
            st.code(app_url)
            st.markdown(
                "Anyone who scans this QR or clicks the link "
                "can ask questions about Aravindh — no installation needed."
            )
    else:
        st.info(
            "Deploy your app to Streamlit Cloud first, "
            "then paste your live URL above to generate the QR code."
        )
 
# ─── Footer ───────────────────────────────────────────────────────────────────
 
st.caption(
    "Powered by LangChain · ChromaDB · SentenceTransformers · "
    "Groq (LLaMA 3) · Streamlit"
)