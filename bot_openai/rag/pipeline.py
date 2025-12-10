"""
Модуль RAG-пайплайна - главный компонент системы.

RAG (Retrieval-Augmented Generation) пайплайн объединяет:
1. Retrieval - поиск релевантных документов в базе знаний
2. Augmented - дополнение запроса найденным контекстом
3. Generation - генерация ответа языковой моделью
"""

import logging
from typing import Dict, List
from openai import OpenAI
from rag.embedder import OpenAIEmbedder
from rag.vectorstore import FAISSVectorStore
from rag.retriever import DocumentRetriever
from config import (
    OPENAI_API_KEY, 
    CHAT_MODEL, 
    VISION_MODEL,
    SYSTEM_PROMPT, 
    RAG_PROMPT_TEMPLATE,
    TOP_K_RESULTS,
    MAX_CONTEXT_LENGTH
)

# Настраиваем логирование
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Основной класс RAG-пайплайна.
    
    Координирует работу всех компонентов:
    - Embedder для создания векторов
    - VectorStore для хранения документов
    - Retriever для поиска релевантных документов
    - OpenAI для генерации ответов
    """
    
    def __init__(self):
        """
        Инициализация RAG-пайплайна.
        """
        logger.info("Инициализация RAG Pipeline...")
        
        # Инициализируем OpenAI клиент
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Инициализируем компоненты RAG
        self.embedder = OpenAIEmbedder()
        self.vectorstore = FAISSVectorStore()
        self.retriever = DocumentRetriever(self.embedder, self.vectorstore)
        
        # Пытаемся загрузить существующий индекс
        self.is_loaded = self.vectorstore.load()
        
        if self.is_loaded:
            logger.info("RAG Pipeline инициализирован с загруженным индексом")
        else:
            logger.warning("RAG Pipeline инициализирован без индекса. "
                         "Выполните индексацию документов.")
    
    def query(self, user_query: str, top_k: int = TOP_K_RESULTS) -> Dict[str, any]:
        """
        Выполняет полный RAG-запрос: поиск контекста + генерация ответа.
        
        Args:
            user_query: Запрос пользователя
            top_k: Количество документов для поиска
            
        Returns:
            Словарь с результатами:
            - answer: Ответ модели
            - context: Использованный контекст
            - sources: Список источников
            - model: Использованная модель
        """
        return self.query_with_history(user_query, [], top_k)
    
    def query_with_history(self, user_query: str, history: list = None, top_k: int = TOP_K_RESULTS) -> Dict[str, any]:
        """
        Выполняет RAG-запрос с учетом истории разговора.
        
        Args:
            user_query: Запрос пользователя
            history: История сообщений [{"role": "user/assistant", "content": "..."}]
            top_k: Количество документов для поиска
            
        Returns:
            Словарь с результатами
        """
        logger.info(f"Обработка запроса: '{user_query[:50]}...' (история: {len(history) if history else 0} сообщений)")
        
        if not self.is_loaded:
            logger.error("Индекс не загружен, невозможно выполнить запрос")
            return {
                "answer": "❌ База знаний не загружена. Выполните команду /ingest для индексации документов.",
                "context": "",
                "sources": [],
                "model": CHAT_MODEL
            }
        
        # Шаг 1: Извлекаем релевантный контекст из базы знаний
        context = self.retriever.retrieve_context(
            user_query, 
            top_k=top_k,
            max_length=MAX_CONTEXT_LENGTH
        )
        
        # Шаг 2: Получаем источники
        sources = self.retriever.get_relevant_sources(user_query, top_k)
        
        # Шаг 3: Формируем промпт для текущего вопроса с контекстом
        prompt_with_context = RAG_PROMPT_TEMPLATE.format(
            context=context,
            query=user_query
        )
        
        # Шаг 4: Формируем историю сообщений для модели
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Добавляем историю разговора (последние 10 пар сообщений = 20 сообщений)
        if history:
            recent_history = history[-20:] if len(history) > 20 else history
            messages.extend(recent_history)
            logger.info(f"Добавлено {len(recent_history)} сообщений из истории")
        
        # Добавляем текущий вопрос с RAG контекстом
        messages.append({"role": "user", "content": prompt_with_context})
        
        # Шаг 5: Генерируем ответ через OpenAI
        logger.info(f"Отправка запроса к модели {CHAT_MODEL} (всего сообщений: {len(messages)})")
        try:
            response = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Ответ получен, длина: {len(answer)} символов")
            
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            answer = f"❌ Ошибка при генерации ответа: {str(e)}"
        
        return {
            "answer": answer,
            "context": context,
            "sources": sources,
            "model": CHAT_MODEL,
            "clean_query": user_query  # Чистый вопрос без RAG контекста для сохранения в историю
        }
    
    def process_image(self, image_url: str, user_query: str = None) -> Dict[str, any]:
        """
        Обрабатывает изображение с помощью vision модели.
        
        Может работать в двух режимах:
        1. Только извлечение текста с изображения
        2. Извлечение текста + RAG-обработка с запросом
        
        Args:
            image_url: URL или base64 изображения
            user_query: Опциональный запрос пользователя
            
        Returns:
            Словарь с результатами обработки
        """
        logger.info(f"Обработка изображения с vision моделью")
        
        try:
            # Шаг 1: Извлекаем текст/описание с изображения
            vision_prompt = "Пожалуйста, извлеки и опиши весь текст с этого изображения."
            
            response = self.client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": vision_prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            extracted_text = response.choices[0].message.content
            logger.info(f"Текст извлечен из изображения, длина: {len(extracted_text)} символов")
            
            # Шаг 2: Если есть дополнительный запрос, обрабатываем через RAG
            if user_query and self.is_loaded:
                logger.info("Обработка извлеченного текста через RAG")
                
                # Формируем комбинированный запрос
                combined_query = f"{user_query}\n\nТекст с изображения: {extracted_text}"
                rag_result = self.query(combined_query)
                
                return {
                    "extracted_text": extracted_text,
                    "rag_answer": rag_result["answer"],
                    "sources": rag_result["sources"],
                    "model": VISION_MODEL
                }
            
            # Возвращаем только извлеченный текст
            return {
                "extracted_text": extracted_text,
                "rag_answer": None,
                "sources": [],
                "model": VISION_MODEL
            }
            
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения: {e}")
            return {
                "extracted_text": None,
                "rag_answer": None,
                "sources": [],
                "error": str(e),
                "model": VISION_MODEL
            }
    
    def index_documents(self, documents: List[str], sources: List[str]) -> bool:
        """
        Индексирует документы в векторное хранилище.
        
        Args:
            documents: Список текстов документов
            sources: Список источников (имен файлов)
            
        Returns:
            True если индексация успешна, False в случае ошибки
        """
        logger.info(f"Начало индексации {len(documents)} документов")
        
        try:
            # Шаг 1: Создаем эмбеддинги для всех документов
            embeddings = self.embedder.embed_texts(documents)
            
            # Шаг 2: Создаем новый индекс
            dimension = len(embeddings[0])
            self.vectorstore.create_index(dimension)
            
            # Шаг 3: Добавляем документы в индекс
            self.vectorstore.add_documents(documents, embeddings, sources)
            
            # Шаг 4: Сохраняем на диск
            self.vectorstore.save()
            
            self.is_loaded = True
            logger.info("Индексация завершена успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при индексации документов: {e}")
            return False
    
    def get_stats(self) -> Dict[str, any]:
        """
        Возвращает статистику по RAG-системе.
        
        Returns:
            Словарь со статистикой
        """
        stats = self.vectorstore.get_stats()
        stats.update({
            "is_loaded": self.is_loaded,
            "embed_model": self.embedder.model,
            "chat_model": CHAT_MODEL,
            "vision_model": VISION_MODEL
        })
        return stats

