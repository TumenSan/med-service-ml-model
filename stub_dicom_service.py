from flask import Flask, request, jsonify
import numpy as np
import pydicom
from PIL import Image
import tensorflow as tf
import io
import os
from datetime import datetime
import time
#from model_inference import load_model, run_inference

app = Flask(__name__)


# Загружаем модель (один раз)
model = tf.keras.models.load_model("horse_zebra_classifier.h5")
#model = load_model("pretrained_model.h5")

CLASS_NAMES = ['horse', 'zebra']

def preprocess_image(image_bytes, target_size=(128, 128)):
    """Предобработка изображения для классификатора"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Обработка DICOM и получение предсказания
def run_inference(model, dicom_path_or_data):
    # Предположим, приходит путь к DICOM файлу
    ds = pydicom.dcmread(dicom_path_or_data)
    img = ds.pixel_array
    # заглушка
    img = Image.open("00000003_001.png")
    # Меняем размер на 320x320
    resized_image = img.resize((320, 320))

    # Сохраняем результат
    resized_image.save("output_image_320x320.jpg")

    # Пример предобработки (масштабирование, изменение размера и т.д.)
    #img_resized = cv2.resize(img, (224, 224))  # адаптируй под свою модель
    #img_normalized = img_resized / 255.0
    #img_input = np.expand_dims(img_normalized, axis=0)

    # Предсказание
    #prediction = model.predict(img_input)
    prediction = model.predict(resized_image)
    #class_idx = int(np.argmax(prediction))
    #confidence = float(np.max(prediction))

    print(prediction)

    return {
        prediction
        #"class": class_idx,
        #"confidence": confidence
    }


@app.route('/api/process', methods=['POST'])
def process_dicom():
    data = request.json
    dicom_input = data.get("parameters")
    print("Stub получил данные:", data)
    # time.sleep(1)  # Имитация задержки
    # Преобразование и предсказание
    # Путь к локальному файлу
    img_path = 'zebra_picture.png'

    # Чтение и предобработка файла
    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    img_data = preprocess_image(img_bytes)

    # Предсказание
    prediction = model.predict(img_data)
    class_idx = int(prediction[0][0] > 0.5)
    confidence = float(prediction[0][0] if class_idx == 1 else 1 - prediction[0][0])
    class_name = CLASS_NAMES[class_idx]
    print(class_name, confidence)

    try:
        result = run_inference(model, dicom_input)  # результат анализа
        response = {
            "task_id": data.get("taskId"),
            "model_id": data.get("modelId"),
            "data": data.get("parameters"),
            #"data": "{\"bloodPressure\": \"120/80\",\"glucose\": 5.4}",
            "conclusion": class_name
            #"timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        response = {
            "task_id": data.get("taskId"),
            "model_id": data.get("modelId"),
            "data": data.get("parameters"),
            "error": str(e)
        }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
