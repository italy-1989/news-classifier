# Russian News Classification API

FastAPI приложение для классификации русских новостей на 11 категорий.

##  Быстрый запуск


```bash
# Через Docker
# Скачать и запустить
docker run -p 8000:8000 italy1989/news-classifier:latest

# Или собрать из исходников
git clone https://github.com/italy-1989/news-classifier.git
cd news-classifier
    # через Docker
    docker build -t news-classifier .
    docker run -p 8000:8000 news-classifier
    # или через docker-compose
    docker-compose up -d

