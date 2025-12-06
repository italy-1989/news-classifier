from datasets import load_dataset
import pandas as pd
import json
import os


def prepare_data():
    print("Загрузка датасета rus_news_classifier")

    # Проверка, есть ли уже датасет
    if os.path.exists('train.csv') and os.path.exists('test.csv'):
        print("Датасет уже скачан, пропускаем загрузку")
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')
    else:
        # Загрузка датасета
        dataset = load_dataset("data-silence/rus_news_classifier")



        # Создаем DataFrame для train и test
        train_df = pd.DataFrame({
            'text': dataset['train']['news'],
            'label': dataset['train']['labels']
        })

        test_df = pd.DataFrame({
            'text': dataset['test']['news'],
            'label': dataset['test']['labels']
        })

        # Сохраняем данные
        train_df.to_csv('train.csv', index=False)
        test_df.to_csv('test.csv', index=False)

    # Анализируем данные
    print("\nРазмеры датасетов:")
    print(f"Train: {len(train_df)} примеров")
    print(f"Test: {len(test_df)} примеров")

    print(f"Количество классов: {train_df['label'].nunique()}")

    # Анализируем распределение классов в train
    print("\nРаспределение классов в train:")
    train_label_counts = train_df['label'].value_counts().sort_index()
    for label, count in train_label_counts.items():
        print(f"Класс {label}: {count} примеров ({count / len(train_df) * 100}%)")

    # Анализируем распределение классов в test
    print("\nРаспределение классов в test:")
    test_label_counts = test_df['label'].value_counts().sort_index()
    for label, count in test_label_counts.items():
        print(f"Класс {label}: {count} примеров ({count / len(test_df) * 100}%)")


    # Проверка, есть ли уже mapping классов
    if os.path.exists('class_mapping.json'):
        print("\nИспользуем mapping классов из файла class_mapping.json")
        with open('class_mapping.json', 'r', encoding='utf-8') as f:
            class_mapping = json.load(f)

    else:
        "\nСоздаем mapping классов на основе уникальных значений"
        unique_labels = sorted(train_df['label'].unique())
        class_mapping = {str(i): f"class_{label}" for i, label in enumerate(unique_labels)}
        # Сохраняем mapping классов
        with open('class_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)

    print(f"Маппинг классов: {class_mapping}\n")

    # Выводим несколько примеров
    print("Первые 3 примера из train:")
    for i in range(3):
        print(f"Текст: {train_df['text'][i][:100]}...")
        print(f"Метка: {train_df['label'][i]}\n")


    print(f"Данные подготовлены:")
    # Возвращаем DataFrame для использования в других скриптах
    return train_df, test_df, class_mapping

if __name__ == "__main__":
    prepare_data()
