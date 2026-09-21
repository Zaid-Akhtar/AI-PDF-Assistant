# PDF Chatbot with Ollama

A modern, local-first PDF chatbot powered by **Ollama**, **RAG**, and **Streamlit**. Upload any PDF and chat with it — every answer comes straight from your document. No API keys, no cloud, 100% local.

![AI PDF Assistant Demo](assets/logo.png)

---

## ✨ Features

- 📄 **Upload any PDF** and chat with it
- 🧠 **Local LLM** via Ollama (llama3.2)
- 🔎 **RAG pipeline** for accurate, context-based answers
- 💾 **Persistent vector DB** (ChromaDB)
- 🎨 **Modern dark UI** with gradient accents
- 🔒 **100% private** — nothing leaves your machine

---

## 🔄 How It Works

PDF → Chunking → Embedding → Vector DB → RAG → LLM (Ollama)

--------------------------------------------------------
|        Step       |              Tool                |
|-------------------|----------------------------------|
| PDF Loading       | `PyPDFDirectoryLoader`           |
| Chunking          | `RecursiveCharacterTextSplitter` |
| Embeddings        | `nomic-embed-text` (via Ollama)  |
| Vector DB         | `ChromaDB`                       |
| RAG Orchestration | `LangChain`                      |
| LLM               | `llama3.2` (via Ollama)          |
| UI                | `Streamlit`                      |
--------------------------------------------------------

## 🚀 Setup

### 1. Install Ollama

Download from [ollama.com/download](https://ollama.com/download).

### 2. Pull the models

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
