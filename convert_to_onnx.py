import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import onnxruntime as ort
import numpy as np
import os

def convert_to_onnx(path="news_classification_model_best"):
    """Конвертация модели в ONNX формат"""
    print("Конвертируем модель в ONNX формат")

    model_path = os.path.abspath(path)
    onnx_path = "./news_classification_model_best.onnx"

    # Загрузка модели и токенизатора
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()

    # Создание примера входных данных
    dummy_input ='Пример новости для классификации' #Пример новости для классификации

    inputs = tokenizer(
        dummy_input,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Экспорт в ONNX
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_shapes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        },
        export_params=True,
    )

    print("Модель конвертирована в ONNX")

    # Тестирование ONNX модели
    session = ort.InferenceSession(onnx_path)

    # Подготовка входных данных
    input_ids = inputs['input_ids'].cpu().numpy()
    attention_mask = inputs['attention_mask'].cpu().numpy()

    # Инференс
    outputs = session.run(
        None,
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    )

    print("Тестирование ONNX модели выполнено")
    print(f"Формат выходных данных: {outputs[0].shape}")

    # Сравнение результатов PyTorch и ONNX
    with torch.no_grad():
        torch_outputs = model(inputs['input_ids'], inputs['attention_mask'])
        torch_logits = torch_outputs.logits.cpu().numpy()

    onnx_logits = outputs[0]

    diff = np.abs(torch_logits - onnx_logits).max()
    print(f"Максимальное расхождение между PyTorch и ONNX моделями: {diff}")

if __name__ == "__main__":
    # Проверяем наличие обученной модели
    if not os.path.exists('news_classification_model_best'):
        print("Обученная модель не найдена. Запускаем обучение...")
        import training_pipeline
        pipeline = training_pipeline.NewsClassificationPipeline()
        pipeline.train()
    # Конвертация в ONNX
    convert_to_onnx()