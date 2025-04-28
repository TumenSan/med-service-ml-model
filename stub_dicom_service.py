from flask import Flask, request, jsonify
import numpy as np
import pydicom
from PIL import Image
import tensorflow as tf
import io
import logging
import os
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


def preprocess_image(image_bytes, target_size=(128, 128)):
    """Предобработка изображения для классификатора"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Ошибка предобработки изображения: {e}")
        raise

# Обработка DICOM и получение предсказания
def run_inference(model, image_or_dicom_path_or_data):
    # Путь к локальному файлу
    img_path = 'horse_picture.png'
    #img_path = 'zebra_picture.png'

    # Чтение и предобработка файла
    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    # Преобразование
    img_data = preprocess_image(img_bytes)

    CLASS_NAMES = ['horse', 'zebra']

    # Предсказание
    prediction = model.predict(img_data)
    class_idx = int(prediction[0][0] > 0.5)
    confidence = float(prediction[0][0] if class_idx == 1 else 1 - prediction[0][0])
    class_name = CLASS_NAMES[class_idx]
    logger.info(f"Предсказание: {class_name} (уверенность: {confidence:.2%})")

    return {
        "class_name": class_name,
        "confidence": confidence
    }


@app.route('/api/process', methods=['POST'])
def process_dicom():
    data = request.json
    logger.info(f"Получены данные:", data)
    dicom_input = data.get("parameters")
    print("Получены данные:", data)

    # Загружаем модель (один раз)
    model_path = "horse_zebra_classifier.h5"
    try:
        model = tf.keras.models.load_model(model_path)
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return jsonify({"error": "Не удалось загрузить модель."}), 500

    try:
        result = run_inference(model, dicom_input)  # результат анализа
        response = {
            "task_id": data.get("taskId"),
            "model_id": data.get("modelId"),
            "data": data.get("parameters"),
            #"data": "{\"bloodPressure\": \"120/80\",\"glucose\": 5.4}",
            "conclusion": result["class_name"]
            #"confidence": result["confidence"],
        }
    except Exception as e:
        response = {
            "task_id": data.get("taskId"),
            "model_id": data.get("modelId"),
            "data": data.get("parameters"),
            "error": str(e)
        }
    finally:
        return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
