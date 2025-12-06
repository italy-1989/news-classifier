from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import json
import os

def test(path_onnx_model="./news_classification_model_best.onnx"):
    """Тестовый инференс на нескольких примерах"""
    print("Тестовый инференс onnx модели")

    # Загружаем ONNX модель
    session = ort.InferenceSession(path_onnx_model)
    tokenizer = AutoTokenizer.from_pretrained("./news_classification_model_best")

    with open('class_mapping.json', 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)

    # Искусственные примеры
    artificial_texts = [
        "Решение было принято на фоне стабилизации инфляции и роста промышленного производства. Аналитики прогнозируют, что это положительно скажется на курсе национальной валюты и притоке инвестиций.",
        "С помощью космического телескопа международная группа астрономов открыла планету, находящуюся на идеальном расстоянии от своей звезды, что позволяет предположить возможность наличия на ее поверхности жидкой воды.",
        "В драматичном финальном матче со счетом 3:2 была одержана победа над многократным чемпионом. Решающий гол был забит на последней минуте игры, что привело в восторг тысячи болельщиков на стадионе."
    ]

    for text in artificial_texts:
        # Токенизация
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np"
        )

        # ONNX инференс
        outputs = session.run(
            None,
            {
                'input_ids': inputs['input_ids'],
                'attention_mask': inputs['attention_mask']
            }
        )
        print('output', outputs)
        # Обработка результатов
        logits = outputs[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]

        print(f"Текст: {text}")
        print(f"Предсказанный класс: {class_mapping[str(predicted_class_idx)]}")
        print(f"Уверенность: {confidence}")

        # Все предсказания
        sorted_indices = np.argsort(probabilities)[::-1]
        print("Все предсказания:")
        for idx in sorted_indices:
            print(f"   {class_mapping[str(idx)]}: {probabilities[idx]:.4f}")


if __name__ == "__main__":
    # Проверяем наличие ONNX модели
    if not os.path.exists("./news_classification_model_best.onnx"):
        print("ONNX модель не найдена. Запускаем конвертацию...")
        import convert_to_onnx
        convert_to_onnx.convert_to_onnx()
    
    # Тест ONNX модели
    test()