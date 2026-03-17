# chatbot.py — improved retrieval and strict prompt

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )
    return vectorstore

# ─── Stricter prompt ──────────────────────────────────────────────────────────
# Key changes:
# 1. Explicitly forbids answering from outside the context
# 2. Tells the LLM the candidate's name upfront
# 3. Gives a clear fallback instruction

INTERVIEW_PROMPT = PromptTemplate(
    template="""You are a professional AI assistant representing a job candidate named Aravindh S during an interview.

STRICT RULES — follow these exactly:
1. Answer ONLY using the CONTEXT provided below. Do not use any outside knowledge.
2. If the answer is not found in the CONTEXT, respond exactly with: "I'm sorry, I don't have that specific information about Aravindh at the moment."
3. Always refer to Aravindh in third person — use "Aravindh" or "he", never "I".
4. Always begin your answer politely, for example: "Thank you for your question." or "That is a great question."
5. Keep answers professional, clear, and structured — as if speaking in a formal job interview.
6. Do not repeat the question back. Go straight to the answer after the polite opening.
7. Do not add any information that is not present in the CONTEXT.

---
CONTEXT (verified information about Aravindh S):
{context}
---

INTERVIEWER'S QUESTION: {question}

PROFESSIONAL ANSWER:""",
    input_variables=["context", "question"]
)

def format_docs(docs):
    # Show chunk content clearly separated
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",          # MMR = Maximum Marginal Relevance
                                    # fetches diverse chunks, not just similar ones
                                    # this prevents the LLM from seeing
                                    # the same info repeated 4 times
        search_kwargs={
            "k": 6,                 # retrieve 6 chunks (more coverage)
            "fetch_k": 20,          # consider top 20 before picking 6 diverse ones
            "lambda_mult": 0.7      # 0 = max diversity, 1 = max relevance
        }
    )

    llm = Ollama(
        model="gemma2:2b",
        temperature=0.1,            # lower = more factual, less hallucination
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
    print("Loading chatbot...")
    chain = build_rag_chain()

    questions = [
        "Tell me about Aravindh",
        "What are his technical skills?",
        "Why should we hire him?",
        "What projects has he built?",
        "What are his hobbies?"
    ]

    for q in questions:
        print(f"\n{'='*50}")
        print(f"Question: {q}")
        print(f"Answer: {get_answer(chain, q)}")