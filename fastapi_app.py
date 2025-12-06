from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import json
from typing import Dict
import os

app = FastAPI(
    title="Russian News Classification API",
    description="API для классификации русских новостей на 11 категорий",
    version="1.0.0"
)


class ClassificationRequest(BaseModel):
    text: str


class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]



class TextClassifier:
    def __init__(self,  
                onnx_model_path="./news_classification_model_best.onnx", 
                tokenizer_path="./news_classification_model_best"
                ):
        # Проверяем наличие ONNX модели
        if not os.path.exists(onnx_model_path):
            print(f"ONNX модель не найдена по пути: {onnx_model_path}")
            print("Запускаем процесс создания модели...")
            self.create_model()
        
        # Проверяем наличие токенизатора
        if not os.path.exists(tokenizer_path):
            print(f"Токенизатор не найден по пути: {tokenizer_path}")
            print("Запускаем процесс создания модели...")
            self.create_model()
        
        self.session = ort.InferenceSession(onnx_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("Модель загружена.")
        
        # Проверяем наличие mapping классов
        if not os.path.exists('class_mapping.json'):
            print("class_mapping.json не найден. Запускаем подготовку данных...")
            self.create_model()
            
        with open('class_mapping.json', 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)
        print(f"Классы: {list(self.class_mapping.values())}")
    
    def create_model(self):
        """Запуск всего pipeline для создания модели"""
        print("Запускаем полный pipeline создания модели...")
        
        # 1. Подготовка данных
        if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
            print("Этап 1: Подготовка данных...")
            import data_preparation
            data_preparation.prepare_data()
        
        # 2. Обучение модели
        if not os.path.exists('news_classification_model_best'):
            print("Этап 2: Обучение модели...")
            import training_pipeline
            pipeline = training_pipeline.NewsClassificationPipeline()
            pipeline.train()
        
        # 3. Конвертация в ONNX
        if not os.path.exists('./news_classification_model_best.onnx'):
            print("Этап 3: Конвертация в ONNX...")
            import convert_to_onnx
            convert_to_onnx.convert_to_onnx()
        
        print("Pipeline создания модели завершен.")

    def predict(self, text: str) -> ClassificationResponse:
        # Токенизация
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )

        # ONNX инференс
        outputs = self.session.run(
            None,
            {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
        )

        # Обработка результатов
        logits = outputs[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        predicted_class_idx = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class_idx])

        # Создание response
        all_probs = {
            self.class_mapping[str(i)]: float(prob)
            for i, prob in enumerate(probabilities)
        }

        return ClassificationResponse(
            predicted_class=self.class_mapping[str(predicted_class_idx)],
            confidence=confidence,
            all_probabilities=all_probs
        )


# Инициализация классификатора при старте приложения
classifier = TextClassifier()


@app.get("/")
async def root():
    return {
        "message": "Russian News Classification API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": [
            "/text_classification",
            "/health",
            "/classes"
        ]
    }


@app.post("/text_classification", response_model=ClassificationResponse)
async def classify_text(request: ClassificationRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        if len(request.text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Text is too short (minimum 10 characters)")

        result = classifier.predict(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.get("/health")
async def health_check():
    try:
        # Проверяем что модель загружена и работает
        test_text = "Тестовая новость для проверки работы модели."
        classifier.predict(test_text)
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_path": os.getenv("MODEL_PATH", "default")
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {str(e)}")


@app.get("/classes")
async def get_classes():
    """Возвращает список всех классов"""
    return {
        "classes": list(classifier.class_mapping.values()),
        "count": len(classifier.class_mapping)
    }


@app.get("/model-info")
async def get_model_info():
    """Информация о загруженной модели"""
    return {
        "model_type": "ONNX",
        "classes_count": len(classifier.class_mapping),
        "tokenizer": "rubert-tiny2",
        "max_length": 512
    }



if __name__ == "__main__":
    # Проверяем наличие всех необходимых файлов
    required_files = [
        'news_classification_model_best.onnx',
        'news_classification_model_best',
        'class_mapping.json'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Не найдены файлы: {missing_files}")
        print("Автоматически запускаем pipeline создания модели...")
        
        # Создаем экземпляр классификатора, который запустит pipeline
        classifier = TextClassifier(
            onnx_model_path="./news_classification_model_best.onnx",
            tokenizer_path="./news_classification_model_best"
        )
    
    import uvicorn
    print("Для доступа к swagger перейди по адресу: http://0.0.0.0:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)