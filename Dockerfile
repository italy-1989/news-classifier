# Dockerfile
FROM python:3.12-slim

# Создание рабочей директории
WORKDIR /app

# Установка системных зависимостей (минимальный набор)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Копирование файла необходимых зависимостей
COPY requirements.txt .
# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копирование только необходимых файлов
COPY fastapi_app.py .
COPY class_mapping.json .
COPY news_classification_model_best.onnx .
COPY news_classification_model_best.onnx.data .

# Создание папки для модели и копирование токенизатора
RUN mkdir -p /app/news_classification_model_best
COPY news_classification_model_best/config.json /app/news_classification_model_best/
COPY news_classification_model_best/tokenizer.json /app/news_classification_model_best/
COPY news_classification_model_best/tokenizer_config.json /app/news_classification_model_best/
COPY news_classification_model_best/special_tokens_map.json /app/news_classification_model_best/
COPY news_classification_model_best/vocab.txt /app/news_classification_model_best/

# Создание непривилегированного пользователя
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

USER appuser

# Открытие порта
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Запуск приложения
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
