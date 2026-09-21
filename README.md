# PDF Chatbot with Ollama

Local PDF chatbot jo Ollama use karta hai. Koi API key ki zaroorat nahi.

## Flow
PDF → Chunking → Embedding → ChromaDB → RAG → Ollama LLM

## Setup

### 1. Ollama Install Karein
https://ollama.com/download se Ollama install karein.

### 2. Models Pull Karein
```bash
ollama pull nomic-embed-text
ollama pull llama3.2