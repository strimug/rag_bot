"""
Модуль для создания эмбеддингов (векторных представлений текста) через OpenAI API.

Эмбеддинги - это числовые векторы, которые представляют смысл текста.
Тексты с похожим значением будут иметь похожие векторы.
Это позволяет находить семантически близкие документы.
"""

import logging
from typing import List
from openai import OpenAI
from config import OPENAI_API_KEY, EMBED_MODEL

# Настраиваем логирование
logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """
    Класс для создания эмбеддингов текста с использованием OpenAI API.
    
    Эмбеддинги используются для:
    1. Преобразования документов в векторы при индексации
    2. Преобразования запроса пользователя в вектор при поиске
    3. Сравнения семантической близости текстов
    """
    
    def __init__(self, model: str = EMBED_MODEL):
        """
        Инициализация эмбеддера.
        
        Args:
            model: Название модели OpenAI для создания эмбеддингов
                  (по умолчанию text-embedding-3-small)
        """
        self.model = model
        # Создаем клиент OpenAI с API ключом
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"OpenAIEmbedder инициализирован с моделью: {model}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Создает эмбеддинг (векторное представление) для одного текста.
        
        Args:
            text: Текст для преобразования в вектор
            
        Returns:
            Список чисел с плавающей точкой - вектор эмбеддинга
            (обычно 1536 измерений для text-embedding-3-small)
        """
        try:
            logger.debug(f"Создание эмбеддинга для текста длиной {len(text)} символов")
            
            # Вызываем API OpenAI для создания эмбеддинга
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"  # Возвращаем вектор как список чисел
            )
            
            # Извлекаем вектор из ответа
            embedding = response.data[0].embedding
            logger.debug(f"Эмбеддинг создан, размерность: {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддинга: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги для нескольких текстов.
        
        Обрабатывает тексты по одному, чтобы избежать превышения лимита токенов.
        
        Args:
            texts: Список текстов для преобразования
            
        Returns:
            Список векторов эмбеддингов (по одному вектору на текст)
        """
        try:
            logger.info(f"Создание эмбеддингов для {len(texts)} текстов")
            
            embeddings = []
            for i, text in enumerate(texts, 1):
                logger.debug(f"Обработка текста {i}/{len(texts)} (длина: {len(text)} символов)")
                
                # Обрабатываем каждый текст отдельно
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    encoding_format="float"
                )
                
                embeddings.append(response.data[0].embedding)
            
            logger.info(f"Успешно создано {len(embeddings)} эмбеддингов")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Возвращает размерность векторов эмбеддингов для данной модели.
        
        Используется при инициализации FAISS индекса, чтобы знать
        размерность векторного пространства.
        
        Returns:
            Размерность вектора (1536 для text-embedding-3-small)
        """
        # Создаем тестовый эмбеддинг для определения размерности
        test_embedding = self.embed_text("test")
        dimension = len(test_embedding)
        logger.debug(f"Размерность эмбеддингов: {dimension}")
        return dimension

