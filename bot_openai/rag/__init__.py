"""
RAG (Retrieval-Augmented Generation) пакет для Telegram-бота.

Этот пакет содержит все компоненты для работы RAG-системы:
- Embedder: создание векторных представлений текста
- VectorStore: хранение и поиск векторов (FAISS)
- Retriever: извлечение релевантных документов
- Pipeline: координация всех компонентов
"""

__version__ = "1.0.0"
__author__ = "RAG Bot Team"

from .embedder import OpenAIEmbedder
from .vectorstore import FAISSVectorStore
from .retriever import DocumentRetriever
from .pipeline import RAGPipeline

__all__ = [
    "OpenAIEmbedder",
    "FAISSVectorStore", 
    "DocumentRetriever",
    "RAGPipeline"
]

