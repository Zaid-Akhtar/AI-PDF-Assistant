# src/ingest.py

import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from src.config import (
    PDF_FOLDER,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_pdfs():
    """PDF files load karta hai."""
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        return []
    
    loader = PyPDFDirectoryLoader(PDF_FOLDER)
    documents = loader.load()
    return documents


def split_documents(documents):
    """Documents ko chunks mein todta hai."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_embedding_model():
    """Ollama embedding model return karta hai."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def build_vector_db():
    """Poora ingestion pipeline chalata hai."""
    
    # 1. PDF Load
    print("📚 PDFs load ho rahi hain...")
    documents = load_pdfs()
    
    if not documents:
        print("❌ Koi PDF nahi mili. data/pdfs/ folder mein PDF rakhein.")
        return None
    
    print(f"✅ {len(documents)} pages load hui.")
    
    # 2. Chunking
    print("✂️ Chunks ban rahe hain...")
    chunks = split_documents(documents)
    print(f"✅ {len(chunks)} chunks bane.")
    
    # 3. Purana DB delete karna (fresh start ke liye)
    if os.path.exists(CHROMA_PERSIST_DIR):
        print("🧹 Purana vector DB delete ho raha hai...")
        shutil.rmtree(CHROMA_PERSIST_DIR)
    
    # 4. Embedding + Vector DB
    print("🧠 Embeddings ban rahi hain aur ChromaDB mein save ho rahi hain...")
    embeddings = get_embedding_model()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    
    print("✅ Vector DB tayyar ho gaya!")
    return vectorstore


if __name__ == "__main__":
    build_vector_db()