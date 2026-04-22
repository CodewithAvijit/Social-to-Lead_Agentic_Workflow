"""
RAG Pipeline for AutoStream Knowledge Base.
Uses HuggingFace sentence-transformers for local embeddings (no API key needed)
and ChromaDB as the vector store.
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_PATH = os.path.join(_BASE_DIR, "data", "knowledge_base.md")

_vectorstore = None

def _build_vectorstore():
    global _vectorstore

    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"[RAG] WARNING: Knowledge base not found at {KNOWLEDGE_BASE_PATH}")
        return

    try:
        loader = TextLoader(KNOWLEDGE_BASE_PATH, encoding="utf-8")
        raw_docs = loader.load()

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )
        split_docs = splitter.split_text(raw_docs[0].page_content)

        # Convert to LangChain Document objects if needed
        from langchain.schema import Document
        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in split_docs
        ]

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        _vectorstore = Chroma.from_documents(documents, embeddings)
        print("[RAG] Vector store initialised successfully.")

    except Exception as exc:
        print(f"[RAG] ERROR building vector store: {exc}")
        _vectorstore = None


_build_vectorstore()


def query_knowledge_base(query: str, k: int = 3) -> str:
    if _vectorstore is None:
        return ""

    try:
        docs = _vectorstore.similarity_search(query, k=k)
        return "\n\n".join(doc.page_content for doc in docs) if docs else ""
    except Exception as exc:
        print(f"[RAG] Query error: {exc}")
        return ""