import os

def run_full_pipeline():
    print("Запуск полного pipeline")
        
    # 1. Подготовка данных
    if not os.path.exists('train.csv') or not os.path.exists('test.csv'):
        print("Подготовка данных")
        import data_preparation
        data_preparation.prepare_data()
    else:
        print("Данные уже подготовлены")
    
    # 2. Обучение модели
    if not os.path.exists('news_classification_model_best'):
        print("Обучение модели")
        import training_pipeline
        pipeline = training_pipeline.NewsClassificationPipeline()
        pipeline.train()
    else:
        print("Модель уже обучена")
    
    # 3. Оценка модели
    print("Оценка модели...")
    import test_model
    test_model.test(max_samples=100)
    
    # 4. Конвертация в ONNX
    if not os.path.exists('./news_classification_model_best.onnx'):
        print("Конвертация в ONNX")
        import convert_to_onnx
        convert_to_onnx.convert_to_onnx()
    else:
        print("ONNX модель уже создана")
    
    # 5. Оценка модели ONNX
    print("Оценка модели ONNX")
    import test_model_onnx
    test_model_onnx.test()
    
    print("Pipeline успешно завершен")

if __name__ == "__main__":
    run_full_pipeline()