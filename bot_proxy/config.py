"""
Конфигурация для Telegram-бота с RAG на основе OpenRouter API.

Этот модуль содержит настройки для работы бота через OpenRouter API,
совместимый с OpenAI API. Это позволяет использовать различные модели LLM
через единый интерфейс OpenRouter.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения из файла .env
load_dotenv()

# ========== TELEGRAM НАСТРОЙКИ ==========
# Токен бота получаем от @BotFather в Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не установлен в переменных окружения!")

# ========== OPENROUTER API НАСТРОЙКИ ==========
# OpenRouter Base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# API ключ для доступа к OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY не установлен в переменных окружения!")

# Для обратной совместимости
OPENAI_API_KEY = OPENROUTER_API_KEY
OPENAI_BASE_URL = OPENROUTER_BASE_URL

# Модель для создания эмбеддингов (векторных представлений текста)
# Формат OpenRouter: openai/text-embedding-3-small
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

# Модель для генерации ответов (чат)
# Формат OpenRouter: openai/gpt-4o-mini
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")

# Модель для обработки изображений (vision)
# Формат OpenRouter: openai/gpt-4o-mini
VISION_MODEL = os.getenv("VISION_MODEL", "openai/gpt-4o-mini")

# Таймаут для HTTP запросов (в секундах)
REQUEST_TIMEOUT = 60

# ========== FAISS НАСТРОЙКИ ==========
# Путь к индексному файлу FAISS (векторная база данных)
BASE_DIR = Path(__file__).parent
FAISS_INDEX_PATH = BASE_DIR / "index.faiss"

# Путь к файлу с метаданными (тексты документов)
FAISS_METADATA_PATH = BASE_DIR / "metadata.json"

# Путь к директории с документами для индексации
DOCS_PATH = BASE_DIR.parent / "data" / "docs"

# ========== RAG НАСТРОЙКИ ==========
# Количество документов для извлечения из базы знаний
TOP_K_RESULTS = 3

# Максимальная длина контекста (в символах)
MAX_CONTEXT_LENGTH = 3000

# ========== ПРОМПТЫ ==========
# Системный промпт для RAG-ассистента
SYSTEM_PROMPT = """
Ты - ......
"""

# Шаблон для формирования промпта с контекстом
RAG_PROMPT_TEMPLATE = """Контекст из базы знаний:
{context}

Вопрос пользователя: {query}

Ответ:"""

# ========== ЛОГИРОВАНИЕ ==========
# Уровень логирования (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = "INFO"

# Формат сообщений в логах
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


