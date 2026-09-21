# src/config.py

# Ollama Model Settings
EMBEDDING_MODEL = "nomic-embed-text"   # Embedding ke liye model
LLM_MODEL = "llama3.2"                  # Jawab generate karne ke liye model

# PDF Folder Path
PDF_FOLDER = "data/pdfs"

# Vector Database (ChromaDB) Path
CHROMA_PERSIST_DIR = "chroma_db"

# Chunking Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Settings
RETRIEVAL_K = 4  # Kitne chunks retrieve karne hain