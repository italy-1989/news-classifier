import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
import os

def test(max_samples=100, test_path=None):
    """Оценка модели небольшом количестве примеров"""
    print("Оценка модели")

    # Загружаем данные
    if test_path is None:
        test_df = pd.read_csv('test.csv')
    else:
        test_df = pd.read_csv(f'{test_path}')
    # Ограничиваем количество примеров если нужно
    if len(test_df) > max_samples:
        print(f"Ограничиваем количество примеров до {max_samples}")
        test_df = test_df.sample(n=max_samples).reset_index(drop=True)

    # Загружаем модель и токенизатор
    model_path = os.path.abspath("news_classification_model_best")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Переносим модель на GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Используется устройство: {device}")

    # Загружаем mapping классов
    with open('class_mapping.json', 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)

    # Токенизация всего датасета
    print("Токенизируем test set...")
    tokenized = tokenizer(
        test_df['text'].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Создаем DataLoader
    dataset = TensorDataset(tokenized['input_ids'], tokenized['attention_mask'], torch.tensor(test_df['label'].values))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Предсказания
    predictions = []
    true_labels = []

    print("Делаем предсказания")
    processed_count = 0
    total_samples = len(test_df)

    for batch in dataloader:
        input_ids, attention_mask, labels = batch

        # Переносим на GPU
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        predictions.extend(batch_predictions)
        true_labels.extend(labels.numpy())

        processed_count += len(batch_predictions)
        print(
            f"Обработано {processed_count}/{total_samples} примеров ({(processed_count / total_samples) * 100}%)")

    # Метрики
    print("Сичтаем метрики:")
    print(classification_report(true_labels, predictions,
                                target_names=list(class_mapping.values())))

    # Матрица ошибок
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_mapping.values(),
                yticklabels=class_mapping.values())
    plt.title('Матрица ошибок')
    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Матрица ошибок сохранена")


if __name__ == "__main__":
    # Проверяем наличие обученной модели
    if not os.path.exists('news_classification_model_best'):
        print("Обученная модель не найдена. Запускаем обучение...")
        import training_pipeline
        pipeline = training_pipeline.NewsClassificationPipeline()
        pipeline.train()

    # Оценка модели
    test()

    print("Тестирование завершено")