"""
Модуль для извлечения релевантных документов из векторного хранилища.

Retriever (извлекатель) работает одинаково для обеих версий бота,
так как зависит только от интерфейса embedder'а и vectorstore.
"""

import logging
from typing import List, Tuple
from rag.embedder import ProxyAPIEmbedder
from rag.vectorstore import FAISSVectorStore
from config import TOP_K_RESULTS

# Настраиваем логирование
logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Класс для извлечения релевантных документов.
    
    Процесс работы:
    1. Принимает текстовый запрос пользователя
    2. Преобразует запрос в эмбеддинг (вектор)
    3. Ищет похожие векторы в FAISS
    4. Возвращает тексты найденных документов
    """
    
    def __init__(self, embedder: ProxyAPIEmbedder, vectorstore: FAISSVectorStore):
        """
        Инициализация retriever'а.
        
        Args:
            embedder: Объект для создания эмбеддингов
            vectorstore: Векторное хранилище с документами
        """
        self.embedder = embedder
        self.vectorstore = vectorstore
        logger.info("DocumentRetriever инициализирован")
    
    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Tuple[str, str, float]]:
        """
        Извлекает наиболее релевантные документы для запроса.
        
        Args:
            query: Текстовый запрос пользователя
            top_k: Количество документов для извлечения
            
        Returns:
            Список кортежей (текст документа, источник, score релевантности)
        """
        logger.info(f"Поиск документов для запроса: '{query[:50]}...'")
        
        # Шаг 1: Преобразуем запрос в вектор
        query_embedding = self.embedder.embed_text(query)
        logger.debug(f"Эмбеддинг запроса создан, размерность: {len(query_embedding)}")
        
        # Шаг 2: Ищем похожие документы в FAISS
        results = self.vectorstore.search(query_embedding, k=top_k)
        
        # Логируем результаты
        logger.info(f"Найдено {len(results)} релевантных документов")
        for i, (text, source, distance) in enumerate(results):
            logger.debug(f"Документ {i+1}: источник={source}, "
                        f"расстояние={distance:.4f}, "
                        f"длина={len(text)} символов")
        
        return results
    
    def retrieve_context(self, query: str, top_k: int = TOP_K_RESULTS, 
                        max_length: int = 3000) -> str:
        """
        Извлекает релевантные документы и объединяет их в единый контекст.
        
        Args:
            query: Текстовый запрос пользователя
            top_k: Количество документов для извлечения
            max_length: Максимальная длина контекста в символах
            
        Returns:
            Объединенный текст релевантных документов
        """
        # Получаем релевантные документы
        results = self.retrieve(query, top_k)
        
        if not results:
            logger.warning("Релевантные документы не найдены")
            return "Релевантная информация не найдена в базе знаний."
        
        # Формируем контекст из найденных документов
        context_parts = []
        total_length = 0
        
        for i, (text, source, distance) in enumerate(results, 1):
            # Добавляем документ с заголовком
            doc_text = f"[Документ {i} из {source}]\n{text}\n"
            
            # Проверяем, не превышает ли длина лимит
            if total_length + len(doc_text) > max_length:
                # Если превышает, обрезаем текст
                remaining = max_length - total_length
                if remaining > 100:  # Добавляем только если есть смысл
                    doc_text = doc_text[:remaining] + "...\n"
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
        context = "\n".join(context_parts)
        logger.info(f"Контекст сформирован, длина: {len(context)} символов")
        
        return context
    
    def get_relevant_sources(self, query: str, top_k: int = TOP_K_RESULTS) -> List[str]:
        """
        Возвращает список источников релевантных документов.
        
        Args:
            query: Текстовый запрос пользователя
            top_k: Количество документов для анализа
            
        Returns:
            Список уникальных источников (имен файлов)
        """
        results = self.retrieve(query, top_k)
        
        # Извлекаем уникальные источники
        sources = list(set(source for _, source, _ in results))
        logger.debug(f"Найдено источников: {sources}")
        
        return sources

