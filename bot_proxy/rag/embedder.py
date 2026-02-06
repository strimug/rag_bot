"""
Модуль для создания эмбеддингов через OpenRouter API.

Использует прямые HTTP запросы к OpenRouter API endpoint.
OpenRouter предоставляет доступ к различным моделям эмбеддингов через
единый интерфейс, совместимый с OpenAI API.
"""

import logging
from typing import List
import requests
from config import OPENROUTER_BASE_URL, OPENROUTER_API_KEY, EMBED_MODEL, REQUEST_TIMEOUT

# Настраиваем логирование
logger = logging.getLogger(__name__)


class OpenRouterEmbedder:
    """
    Класс для создания эмбеддингов через OpenRouter API.
    
    Использует HTTP запросы к OpenRouter API, что позволяет:
    - Работать с различными моделями эмбеддингов
    - Полный контроль над запросами
    - Специальные headers для OpenRouter
    - Легкая отладка и логирование запросов
    """
    
    def __init__(self, api_url: str = OPENROUTER_BASE_URL, 
                 api_key: str = OPENROUTER_API_KEY,
                 model: str = EMBED_MODEL):
        """
        Инициализация эмбеддера.
        
        Args:
            api_url: Базовый URL OpenRouter API
            api_key: API ключ для авторизации
            model: Название модели для создания эмбеддингов (формат: openai/text-embedding-3-small)
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        
        # Формируем endpoint для эмбеддингов
        self.embeddings_endpoint = f"{self.api_url}/embeddings"
        
        # Подготавливаем заголовки для всех запросов (OpenRouter требует специальные headers)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com",  # Для OpenRouter rankings
            "X-Title": "RAG Bot",  # Для OpenRouter rankings
        }
        
        logger.info(f"OpenRouterEmbedder инициализирован")
        logger.info(f"Endpoint: {self.embeddings_endpoint}")
        logger.info(f"Модель: {self.model}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Создает эмбеддинг для одного текста через HTTP запрос к OpenRouter.
        
        Args:
            text: Текст для преобразования в вектор
            
        Returns:
            Список чисел с плавающей точкой - вектор эмбеддинга
        """
        try:
            logger.debug(f"Создание эмбеддинга для текста длиной {len(text)} символов")
            
            # Формируем тело запроса в формате OpenAI API (OpenRouter совместим)
            payload = {
                "model": self.model,
                "input": [text],  # OpenRouter ожидает список
                "encoding_format": "float"
            }
            
            # Отправляем POST запрос к OpenRouter API
            response = requests.post(
                self.embeddings_endpoint,
                headers=self.headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            
            # Проверяем статус ответа
            if response.status_code != 200:
                error_text = response.text
                try:
                    error_json = response.json()
                    error_msg = f"OpenRouter API вернул статус {response.status_code}: {error_json}"
                except:
                    error_msg = f"OpenRouter API вернул статус {response.status_code}: {error_text}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Парсим JSON ответ
            data = response.json()
            
            # Проверяем на ошибки в ответе
            if "error" in data:
                error_info = data["error"]
                error_msg = f"OpenRouter API ошибка: {error_info.get('message', 'Unknown error')} (code: {error_info.get('code', 'unknown')})"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Извлекаем вектор эмбеддинга
            if "data" not in data or not data["data"]:
                error_msg = f"Нет 'data' в ответе OpenRouter. Ответ: {data}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if "embedding" not in data["data"][0]:
                error_msg = f"Нет 'embedding' в ответе OpenRouter. Ответ: {data}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            embedding = data['data'][0]['embedding']
            if not embedding:
                error_msg = "Пустой эмбеддинг в ответе OpenRouter"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.debug(f"Эмбеддинг создан, размерность: {len(embedding)}")
            
            return embedding
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка HTTP запроса к OpenRouter: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Ошибка парсинга ответа OpenRouter API: {e}")
            raise
        except Exception as e:
            logger.error(f"Неожиданная ошибка при создании эмбеддинга: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Создает эмбеддинги для нескольких текстов через OpenRouter.
        
        Обрабатывает тексты батчами для эффективности (OpenRouter поддерживает до 2048 входов).
        
        Args:
            texts: Список текстов для преобразования
            
        Returns:
            Список векторов эмбеддингов
        """
        try:
            logger.info(f"Создание эмбеддингов для {len(texts)} текстов через OpenRouter")
            
            # OpenRouter поддерживает батчи до 2048 входов
            batch_size = 100
            all_embeddings = []
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            
            for i, batch in enumerate(batches, 1):
                logger.debug(f"Обработка батча {i}/{len(batches)}, размер: {len(batch)}")
                
                try:
                    # Формируем запрос для батча
                    payload = {
                        "model": self.model,
                        "input": batch,  # OpenRouter принимает список текстов
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
                    if response.status_code != 200:
                        error_text = response.text
                        try:
                            error_json = response.json()
                            error_msg = f"OpenRouter API вернул статус {response.status_code}: {error_json}"
                        except:
                            error_msg = f"OpenRouter API вернул статус {response.status_code}: {error_text}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    # Парсим ответ
                    data = response.json()
                    
                    # Проверяем на ошибки
                    if "error" in data:
                        error_info = data["error"]
                        error_msg = f"OpenRouter API ошибка: {error_info.get('message', 'Unknown error')} (code: {error_info.get('code', 'unknown')})"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    # Извлекаем эмбеддинги из батча
                    if "data" not in data:
                        error_msg = f"Нет 'data' в ответе OpenRouter. Ответ: {data}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    batch_embeddings = []
                    for item in data["data"]:
                        if "embedding" not in item:
                            logger.warning(f"Элемент без 'embedding': {list(item.keys()) if isinstance(item, dict) else type(item)}")
                            continue
                        if not item["embedding"]:
                            logger.warning(f"Пустой эмбеддинг в элементе")
                            continue
                        batch_embeddings.append(item["embedding"])
                    
                    if not batch_embeddings:
                        error_msg = f"Нет валидных эмбеддингов в батче {i}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    all_embeddings.extend(batch_embeddings)
                    logger.debug(f"Батч {i} завершен, получено {len(batch_embeddings)} эмбеддингов, всего: {len(all_embeddings)}")
                    
                except requests.exceptions.RequestException as batch_error:
                    logger.error(f"HTTP ошибка для батча {i}: {batch_error}")
                    raise
                except Exception as batch_error:
                    logger.error(f"Ошибка обработки батча {i}: {batch_error}")
                    raise
            
            if len(all_embeddings) != len(texts):
                logger.warning(f"Создано {len(all_embeddings)} эмбеддингов, ожидалось {len(texts)}")
            
            logger.info(f"Успешно создано {len(all_embeddings)} эмбеддингов через OpenRouter")
            
            return all_embeddings
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка HTTP запроса к OpenRouter: {e}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Ошибка парсинга ответа OpenRouter API: {e}")
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
        Проверяет доступность OpenRouter API и корректность настроек.
        
        Returns:
            True если API доступен и работает корректно
        """
        try:
            logger.info("Проверка подключения к OpenRouter API...")
            
            # Пытаемся создать тестовый эмбеддинг
            self.embed_text("test connection")
            
            logger.info("✅ Подключение к OpenRouter API успешно")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к OpenRouter API: {e}")
            return False

