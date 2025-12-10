"""
Модуль для создания эмбеддингов через ProxyAPI.

В отличие от версии с OpenAI SDK, здесь используются прямые HTTP запросы
к API endpoint. Это дает больше гибкости и позволяет работать с любыми
OpenAI-совместимыми сервисами.
"""

import logging
from typing import List
import requests
from config import PROXY_API_URL, PROXY_API_KEY, EMBED_MODEL, REQUEST_TIMEOUT

# Настраиваем логирование
logger = logging.getLogger(__name__)


class ProxyAPIEmbedder:
    """
    Класс для создания эмбеддингов через ProxyAPI.
    
    Использует HTTP запросы вместо SDK, что позволяет:
    - Работать с любыми OpenAI-совместимыми API
    - Полный контроль над запросами
    - Возможность добавления кастомных headers
    - Легкая отладка и логирование запросов
    """
    
    def __init__(self, api_url: str = PROXY_API_URL, 
                 api_key: str = PROXY_API_KEY,
                 model: str = EMBED_MODEL):
        """
        Инициализация эмбеддера.
        
        Args:
            api_url: Базовый URL API
            api_key: API ключ для авторизации
            model: Название модели для создания эмбеддингов
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        
        # Формируем endpoint для эмбеддингов
        self.embeddings_endpoint = f"{self.api_url}/embeddings"
        
        # Подготавливаем заголовки для всех запросов
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"ProxyAPIEmbedder инициализирован")
        logger.info(f"Endpoint: {self.embeddings_endpoint}")
        logger.info(f"Модель: {self.model}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Создает эмбеддинг для одного текста через HTTP запрос.
        
        Args:
            text: Текст для преобразования в вектор
            
        Returns:
            Список чисел с плавающей точкой - вектор эмбеддинга
        """
        try:
            logger.debug(f"Создание эмбеддинга для текста длиной {len(text)} символов")
            
            # Формируем тело запроса в формате OpenAI API
            payload = {
                "model": self.model,
                "input": text,
                "encoding_format": "float"
            }
            
            # Отправляем POST запрос к API
            response = requests.post(
                self.embeddings_endpoint,
                headers=self.headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            
            # Проверяем статус ответа
            response.raise_for_status()
            
            # Парсим JSON ответ
            data = response.json()
            
            # Извлекаем вектор эмбеддинга
            embedding = data['data'][0]['embedding']
            logger.debug(f"Эмбеддинг создан, размерность: {len(embedding)}")
            
            return embedding
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка HTTP запроса: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Ошибка парсинга ответа API: {e}")
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка при создании эмбеддинга: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги для нескольких текстов.
        
        Обрабатывает тексты по одному, чтобы избежать превышения лимита токенов.
        
        Args:
            texts: Список текстов для преобразования
            
        Returns:
            Список векторов эмбеддингов
        """
        try:
            logger.info(f"Создание эмбеддингов для {len(texts)} текстов")
            
            embeddings = []
            for i, text in enumerate(texts, 1):
                logger.debug(f"Обработка текста {i}/{len(texts)} (длина: {len(text)} символов)")
                
                # Формируем запрос для одного текста
                payload = {
                    "model": self.model,
                    "input": text,
                    "encoding_format": "float"
                }
                
                # Отправляем запрос
                response = requests.post(
                    self.embeddings_endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
                
                # Проверяем статус
                response.raise_for_status()
                
                # Парсим ответ
                data = response.json()
                embeddings.append(data['data'][0]['embedding'])
            
            logger.info(f"Успешно создано {len(embeddings)} эмбеддингов")
            
            return embeddings
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка HTTP запроса: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Ошибка парсинга ответа API: {e}")
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка при создании эмбеддингов: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Определяет размерность векторов эмбеддингов.
        
        Returns:
            Размерность вектора
        """
        # Создаем тестовый эмбеддинг
        test_embedding = self.embed_text("test")
        dimension = len(test_embedding)
        logger.debug(f"Размерность эмбеддингов: {dimension}")
        return dimension
    
    def test_connection(self) -> bool:
        """
        Проверяет доступность API и корректность настроек.
        
        Returns:
            True если API доступен и работает корректно
        """
        try:
            logger.info("Проверка подключения к ProxyAPI...")
            
            # Пытаемся создать тестовый эмбеддинг
            self.embed_text("test connection")
            
            logger.info("✅ Подключение к ProxyAPI успешно")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к ProxyAPI: {e}")
            return False

