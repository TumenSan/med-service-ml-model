from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import logging
import os
import base64
import pydicom
import time
from prometheus_client import Summary, Counter, start_http_server
import onnxruntime as ort

# Настройки логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Метрики Prometheus
INFERENCE_TIME = Summary('model_inference_seconds', 'Time spent per model inference')
REQUEST_COUNTER = Counter('inference_requests_total', 'Total inference requests by model', ['model'])

# Поддерживаемые модели
MODEL_PATHS = {
    #"horse_v1": {"path": "models/horse_zebra_classifier.h5", "type": "keras"},
    #"horse_onnx": {"path": "models/horse_zebra.onnx", "type": "onnx"}
    "1": {"path": "models/horse_zebra_classifier.h5", "type": "keras"},
    "2": {"path": "models/horse_zebra.onnx", "type": "onnx"}
}

CLASS_NAMES = ["zebra", "horse"]

# Конфигурация предобработки
MODEL_CONFIG = {
    "input_size": (128, 128),
    "normalize_range": (0, 1),
    "color_mode": "RGB"
}


class ModelManager:
    def __init__(self):
        self.models = {}

    def load_keras_model(self, path):
        return tf.keras.models.load_model(path)

    def load_onnx_model(self, path):
        return ort.InferenceSession(path)

    def load_all_models(self):
        for name, info in MODEL_PATHS.items():
            try:
                if info['type'] == 'keras':
                    self.models[name] = {
                        'model': self.load_keras_model(info['path']),
                        'predict_fn': self._tf_predict
                    }
                elif info['type'] == 'onnx':
                    self.models[name] = {
                        'model': self.load_onnx_model(info['path']),
                        'predict_fn': self._onnx_predict
                    }
                logger.info(f"Модель {name} успешно загружена")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели {name}: {e}")

    def _tf_predict(self, model, input_tensor):
        return model.predict(input_tensor)

    def _onnx_predict(self, model, input_tensor):
        inputs = {model.get_inputs()[0].name: input_tensor.numpy()}
        return model.run(None, inputs)[0]


manager = ModelManager()
manager.load_all_models()


def preprocess_image(config=MODEL_CONFIG) -> tf.Tensor:
    try:
        # Путь к вашему изображению
        image_path = "horse_picture.png"

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Файл {image_path} не найден. "
                "Сервер ожидает этот файл на диске, либо нужно "
                "передавать изображение в parameters"
            )
        else: print("good")

        # Загрузка изображения
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        # Проверка на DICOM (работаем с байтами)
        if image_bytes.startswith(b'\x00\x00\x01\x00'):
            dataset = pydicom.dcmread(io.BytesIO(image_bytes))
            img_array = dataset.pixel_array
            img = Image.fromarray(img_array).convert(config["color_mode"])
        else:
            # Обычное изображение
            img = Image.open(io.BytesIO(image_bytes)).convert(config["color_mode"])

        # Преобразование в Base64
        #encoded_image = base64.b64encode(image_data).decode('utf-8')
        #image_bytes = encoded_image

        # Изменение размера и нормализация
        img = img.resize(config["input_size"])
        img_array = np.array(img).astype(np.float32)
        img_array /= 255.0 * (config["normalize_range"][1] - config["normalize_range"][0])
        img_array += config["normalize_range"][0]

        # Используем tf.data.Dataset для оптимизации
        dataset = tf.data.Dataset.from_tensors(img_array)
        batched_dataset = dataset.batch(1)
        return next(iter(batched_dataset))

    except Exception as e:
        logger.error(f"Ошибка предобработки изображения: {e}")
        raise


def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            logger.warning("GPU не настроен:", e)


setup_gpu()


@app.route('/api/process', methods=['POST'])
def process_request():
    try:
        data = request.get_json(force=True)
        logger.info("Получен запрос:", data)
        print("Получен запрос:", data)

        print(111)
        model_name = data.get("modelId") or "default"
        if model_name not in manager.models:
            return jsonify({
                "error": f"Unsupported model type: {model_name}",
                "supported": list(manager.models.keys())
            }), 400

        print(222)
        REQUEST_COUNTER.labels(model=model_name).inc()

        #image_b64 = data.get("parameters")
        #if not image_b64:
        #    return jsonify({"error": "Missing 'parameters' field"}), 400

        # Декодируем Base64
        #image_bytes = base64.b64decode(image_b64)

        # Предобработка
        input_tensor = preprocess_image()

        # Выполняем инференс
        with INFERENCE_TIME.time():
            prediction = manager.models[model_name]['predict_fn'](manager.models[model_name]['model'], input_tensor)

        confidence = float(prediction[0][0])
        class_name = CLASS_NAMES[int(confidence > 0.5)]

        return jsonify({
            "task_id": data.get("taskId"),
            "model_id": data.get("modelId"),
            #"model_id": model_name,
            "data": data.get("parameters"),
            "conclusion": class_name
        })

    except Exception as e:
        logger.exception("Ошибка обработки запроса:")
        return jsonify({"error": str(e)}), 500


@app.route('/healthz', methods=['GET'])
def health_check():
    """Проверка работоспособности всех моделей"""
    status = {"status": "healthy", "models": {}}
    for name, model_info in manager.models.items():
        try:
            test_input = tf.random.uniform((1, *MODEL_CONFIG["input_size"], 3), 0, 1)
            if model_info['predict_fn'](model_info['model'], test_input) is not None:
                status["models"][name] = "OK"
        except Exception as e:
            status["models"][name] = f"Error: {str(e)}"
            status["status"] = "unhealthy"

    code = 200 if status["status"] == "healthy" else 503
    return jsonify(status), code


if __name__ == '__main__':
    start_http_server(9000)  # Метрики Prometheus
    app.run(host='0.0.0.0', port=8000, threaded=True)
