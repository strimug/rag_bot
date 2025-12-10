"""
RAG (Retrieval-Augmented Generation) пакет для Telegram-бота (ProxyAPI версия).

Этот пакет содержит все компоненты для работы RAG-системы через ProxyAPI:
- Embedder: создание векторных представлений через HTTP
- VectorStore: хранение и поиск векторов (FAISS)
- Retriever: извлечение релевантных документов
- Pipeline: координация всех компонентов
"""

__version__ = "1.0.0"
__author__ = "RAG Bot Team"

from .embedder import ProxyAPIEmbedder
from .vectorstore import FAISSVectorStore
from .retriever import DocumentRetriever
from .pipeline import RAGPipeline

__all__ = [
    "ProxyAPIEmbedder",
    "FAISSVectorStore",
    "DocumentRetriever", 
    "RAGPipeline"
]

