# app.py — with streaming enabled

import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from chatbot import build_rag_chain

st.set_page_config(
    page_title="Aravindh's AI Resume Bot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Aravindh's Personal AI Assistant")
st.markdown(
    "Ask me anything about **Aravindh's** skills, education, projects, "
    "experience, or career goals."
)
st.divider()

# Suggested questions
st.markdown("**💡 Try asking:**")
suggested = [
    "Tell me about Aravindh",
    "What are his technical skills?",
    "What projects has he worked on?",
    "Why should we hire him?",
    "What are his career goals?",
    "What are his strengths?"
]
cols = st.columns(3)
for i, q in enumerate(suggested):
    if cols[i % 3].button(q, use_container_width=True):
        st.session_state["prefill_question"] = q

st.divider()

# Load chain once and cache it
@st.cache_resource
def load_chain():
    with st.spinner("🔄 Loading AI model..."):
        return build_rag_chain()

chain = load_chain()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prefill = st.session_state.pop("prefill_question", None)
user_input = st.chat_input("Ask about Aravindh...") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # ── STREAMING: words appear one by one instead of all at once ──
        response_placeholder = st.empty()
        full_response = ""

        for chunk in chain.stream(user_input):   # .stream() instead of .invoke()
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")  # ▌ = typing cursor

        response_placeholder.markdown(full_response)  # final clean render

    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.divider()
st.caption("Powered by LangChain · ChromaDB · SentenceTransformers · Ollama · Streamlit")