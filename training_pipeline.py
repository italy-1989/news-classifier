import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

class NewsClassificationPipeline:
    def __init__(self, model_name="cointegrated/rubert-tiny2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = None
        self.class_mapping = None
        print(f"Используется устройство: {self.device}")

    def load_data(self):
        """Загрузка данных"""
        print("Загружаем данные...")
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('test.csv')

        # Загружаем mapping классов
        with open('class_mapping.json', 'r', encoding='utf-8') as f:
            self.class_mapping = json.load(f)

        self.num_labels = len(self.class_mapping)  # Теперь здесь инициализируем
        print(f"Количество классов: {self.num_labels}")
        print(f"Названия классов: {list(self.class_mapping.values())}")

        # Создаем datasets
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        return train_dataset, test_dataset

    def tokenize_function(self, examples):
        """Токенизация текстов"""
        tokenized = self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        # Добавляем labels в выходные данные
        tokenized['labels'] = examples['label']
        return tokenized

    def compute_metrics(self, eval_pred):
        """Вычисление метрик"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        precision_macro = precision_score(labels, predictions, average='macro')
        precision_weighted = precision_score(labels, predictions, average='weighted')
        recall_macro = recall_score(labels, predictions, average='macro')
        recall_weighted = recall_score(labels, predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
        }

    def load_trained_model(self, model_path=None):
        """Загрузка уже обученной модели"""
        if model_path is None:
            model_path = os.path.abspath("news_classification_model_best")

        print("Загружаем обученную модель")

        # Загружаем токенизатор и модель
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        print("Модель успешно загружена")
        return self.model, self.tokenizer

    def train(self, force_retrain=False):
        # Проверяем, есть ли уже обученная модель
        if not force_retrain and os.path.exists("./news_classification_model_best"):
            print("Найдена существующая модель, загружаем ее")

            return self.load_trained_model()

        """Обучение модели"""
        print("Загружаем модель и токенизатор")

        '''Загружаем данные'''
        train_dataset, test_dataset = self.load_data()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"  # Явно указываем тип классификации
        )
        self.model.train()

        # Токенизация
        print("Токенизируем данные")
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        tokenized_test = test_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=test_dataset.column_names
        )

        # Аргументы обучения
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=500,  # Увеличиваем eval_steps
            save_strategy="steps",
            save_steps=500,  # Теперь save_steps = eval_steps
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            save_total_limit=1,
            report_to=[],
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )

        # Создание тренера
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        # Обучение
        print("Начинаем обучение")
        train_result = trainer.train()

        # Сохранение модели
        print("Сохраняем модель")
        trainer.save_model("./news_classification_model")
        self.tokenizer.save_pretrained("./news_classification_model")
        print("Модель сохранена")

        # Финальная оценка на test set
        print("Финальная оценка модели на test set...")
        eval_results = trainer.evaluate()

        print("Обучение завершено!")
        print("Финальные метрики:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")

        # Сохраняем метрики
        with open('./news_classification_model/training_metrics.json', 'w', encoding='utf-8') as f:
            json.dump({k: float(v) for k, v in eval_results.items()}, f, indent=2)

        # Сохраняем информацию о модели
        model_info = {
            "model_path": "./news_classification_model_best",
            "num_labels": self.num_labels,
            "class_mapping": self.class_mapping
        }
        with open('./news_classification_model/model_info.json', 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)

        # Сохраняем модель в архив для распространения
        def zip_folder(folder_path, output_path):
            import zipfile
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, folder_path)
                        zipf.write(file_path, arcname)
        zip_folder('./news_classification_model', 'news_classification_model_best.zip')

        return trainer

if __name__ == "__main__":
    # Проверяем наличие данных
    if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
        print("Данные не найдены. Запускаем подготовку данных...")
        import data_preparation
        data_preparation.prepare_data()
    
    pipeline = NewsClassificationPipeline()
    pipeline.train()