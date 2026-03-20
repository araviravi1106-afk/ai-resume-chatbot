# ingest.py — uses FAISS instead of ChromaDB (no compilation issues)

import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings


TOPIC_TAGS = {
    "INTRODUCTION": "[TOPIC: Introduction and Profile]",
    "PERSONAL INFORMATION": "[TOPIC: Personal Information]",
    "EDUCATION": "[TOPIC: Education and Academic Background]",
    "TECHNICAL SKILLS": "[TOPIC: Technical Skills and Programming]",
    "STRENGTHS": "[TOPIC: Strengths and Strong Points]",
    "WEAKNESSES": "[TOPIC: Weaknesses and Areas of Improvement]",
    "CAREER GOALS": "[TOPIC: Career Goals and Future Plans]",
    "FIVE YEAR": "[TOPIC: Five Year Plan and Long Term Goals]",
    "FINAL YEAR PROJECT": "[TOPIC: Final Year Project and AI Chatbot]",
    "TECHNOLOGIES USED": "[TOPIC: Technologies Used in Project]",
    "PROJECTS": "[TOPIC: Projects and Applications Built]",
    "SALARY": "[TOPIC: Salary Expectation]",
    "LEADERSHIP": "[TOPIC: Leadership Experience]",
    "UNIQUENESS": "[TOPIC: Uniqueness and What Makes Aravindh Different]",
    "PERSONALITY": "[TOPIC: Personality and Work Ethic]",
    "PROBLEM SOLVING": "[TOPIC: Problem Solving Approach]",
    "DAILY ROUTINE": "[TOPIC: Daily Routine and Schedule]",
    "HOBBIES": "[TOPIC: Hobbies and Personal Interests]",
    "INTERESTS": "[TOPIC: Interests and Passion Areas]",
    "WHY HIRE": "[TOPIC: Why Hire Aravindh]",
}


def detect_topic_tag(section_text):
    upper = section_text.upper()
    for keyword, tag in TOPIC_TAGS.items():
        if keyword in upper:
            return tag
    return "[TOPIC: General Information about Aravindh]"


def load_data(folder="data"):
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                content = f.read()
                sections = content.split("============================================================")
                for section in sections:
                    section = section.strip()
                    if len(section) > 50:
                        tag = detect_topic_tag(section)
                        tagged_section = f"{tag}\n\n{section}"
                        documents.append(tagged_section)
    print(f"✅ Loaded {len(documents)} tagged sections")
    return documents


def chunk_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)
    print(f"✅ Created {len(all_chunks)} chunks")
    return all_chunks


def store_in_faiss(chunks):
    """Embed chunks and save to FAISS index folder."""
    if os.path.exists("vectorstore"):
        shutil.rmtree("vectorstore")
        print("🗑️  Cleared old vectorstore")

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    # Save FAISS index to disk
    vectorstore.save_local("vectorstore")
    print("✅ FAISS index saved to ./vectorstore")
    return vectorstore


if __name__ == "__main__":
    print("\n📄 Loading data...")
    docs = load_data("data")

    print("\n✂️  Chunking...")
    chunks = chunk_text(docs)

    print("\n🔢 Embedding and storing in FAISS...")
    store_in_faiss(chunks)

    print("\n🎉 Done! Run: streamlit run app.py\n")