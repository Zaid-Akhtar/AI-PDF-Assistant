# src/rag_pipeline.py

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    RETRIEVAL_K,
)


def load_vectorstore():
    """Pehle se bani hui ChromaDB load karta hai."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectorstore


def get_rag_chain():
    """RAG chain return karta hai."""
    
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    
    llm = OllamaLLM(model=LLM_MODEL)
    
    # Prompt template: LLM ko sirf context se jawab dena hai
    prompt = ChatPromptTemplate.from_template("""
Aap ek helpful assistant hain. Neeche diye gaye context ke based par sawal ka jawab dein.
Agar context mein jawab nahi hai, to saaf saaf keh dein ke "Mujhe is document mein iska jawab nahi mila."

Context:
{context}

Question:
{question}

Answer:
""")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # RAG chain: Retriever -> Prompt -> LLM -> Output
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def ask_question(question: str):
    """Sawal pooch kar jawab return karta hai."""
    chain = get_rag_chain()
    answer = chain.invoke(question)
    return answer