# chatbot.py — improved relevance with stronger prompt

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def get_groq_key():
    try:
        import streamlit as st
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )
    return vectorstore


INTERVIEW_PROMPT = PromptTemplate(
    template="""You are a professional AI assistant representing Aravindh S in a formal job interview.

YOUR ONLY JOB: Answer the INTERVIEWER'S QUESTION using ONLY the CONTEXT below.

STRICT RULES:
1. Read the INTERVIEWER'S QUESTION carefully first.
2. Look through the CONTEXT and find only the parts that directly answer the question.
3. Ignore any context chunks that are NOT relevant to the question being asked.
4. Build your answer using ONLY the relevant parts you found.
5. If no part of the CONTEXT answers the question, say exactly: "I'm sorry, I don't have that specific information about Aravindh at the moment."
6. Never use outside knowledge. Never guess or assume.
7. Always refer to Aravindh in third person — "Aravindh" or "he", never "I".
8. Always open politely: "Thank you for your question." or "That is a great question, sir."
9. Be concise — 4 to 6 sentences is ideal unless more detail is truly needed.
10. Sound confident and professional, like an interview spokesperson.

---
CONTEXT (information retrieved from Aravindh's resume):
{context}
---

INTERVIEWER'S QUESTION: {question}

PROFESSIONAL ANSWER (use only what is relevant from the context above):""",
    input_variables=["context", "question"]
)


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain():
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,          # fetch 4 diverse chunks
            "fetch_k": 15,   # consider top 15 before picking 4
            "lambda_mult": 0.6
        }
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,     # 0 = fully deterministic, most factual
        max_tokens=500,
        api_key=get_groq_key()
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | INTERVIEW_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


def get_answer(chain, question):
    return chain.invoke(question)


if __name__ == "__main__":
    print("🤖 Loading chatbot...")
    chain = build_rag_chain()

    test_questions = [
        "Tell me about Aravindh",
        "What are his technical skills?",
        "Why should we hire him?",
        "What projects has he built?",
        "What are his hobbies?",
        "What is his final year project?",
        "Where does he see himself in 5 years?"
    ]

    for q in test_questions:
        print(f"\n{'='*55}")
        print(f"❓ {q}")
        print(f"💬 {get_answer(chain, q)}")