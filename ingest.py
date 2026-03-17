# ingest.py
# This script reads your resume data, splits it into chunks,
# creates embeddings, and stores them in ChromaDB.
# You only need to run this ONCE (or whenever you update your resume).

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter        # ← fixed
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ─── STEP 1: Load your data files ────────────────────────────────────────────

def load_data(folder="data"):
    """Read all .txt files from the data folder and combine them."""
    all_text = ""
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
                content = f.read()
                all_text += f"\n\n--- {filename} ---\n\n" + content
    return all_text

# ─── STEP 2: Split text into chunks ──────────────────────────────────────────

def chunk_text(text):
    """
    Split long text into smaller chunks.
    chunk_size=400: each chunk is about 400 characters
    chunk_overlap=80: chunks overlap by 80 chars so we don't lose context at boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} text chunks.")
    return chunks

# ─── STEP 3: Create embeddings and store in ChromaDB ─────────────────────────

def store_in_chromadb(chunks):
    """
    Convert each chunk into a vector (a list of numbers that represent meaning),
    then store those vectors in ChromaDB so we can search them later.
    """
    # SentenceTransformers is a free, local embedding model — no API key needed
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"  # Small, fast, works great on CPU
    )

    # Store everything in ChromaDB (saves to the 'vectorstore' folder)
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )

    print("✅ Embeddings stored in ChromaDB at ./vectorstore")
    return vectorstore

# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("📄 Loading resume data...")
    raw_text = load_data("data")

    print("✂️  Splitting into chunks...")
    chunks = chunk_text(raw_text)

    print("🔢 Creating embeddings and storing in ChromaDB...")
    store_in_chromadb(chunks)

    print("\n🎉 Ingestion complete! Your resume is now searchable.")