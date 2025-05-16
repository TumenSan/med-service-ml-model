from flask import Flask, request, jsonify
import base64
import io
import logging
import os
import numpy as np
from datetime import datetime
from prometheus_client import Summary, Counter, start_http_server
from ModelManager import ModelManager

# Логгирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# Метрики Prometheus
INFERENCE_TIME = Summary('model_inference_seconds', 'Время инференса')
REQUEST_COUNTER = Counter('inference_requests_total', 'Число запросов по модели', ['model'])

# Получаем тип модели из переменных окружения
MODEL_ID = "1"
MODEL_TYPE = "onnx"
MODEL_PATH = "models/horse_zebra.onnx"

# Инициализируем менеджер моделей
manager = ModelManager(model_path=MODEL_PATH, model_type=MODEL_TYPE)


@app.route("/api/process", methods=["POST"])
def process():
    data = request.get_json()
    logger.info(f"Получена задача: {data}")

    model_id = "1"

    try:
        # Попробуем получить изображение из запроса
        x = data.get("image") or data.get("image_b64")

        if not x:
            logger.warning("Нет изображения, используем zebra_picture.png")
            with open("zebra_picture.png", "rb") as f:
                image_bytes = f.read()
        else:
            image_bytes = base64.b64decode(x)

        input_tensor = manager.preprocess_image(image_bytes)
        with INFERENCE_TIME.time():
            result = manager.predict(input_tensor)

        confidence = float(result[0][0])
        class_name = "horse" if confidence > 0.5 else "zebra"

        return jsonify({
            "taskId": data.get("taskId"),
            "modelId": model_id,
            "conclusion": class_name,
            "confidence": confidence,
            "status": "COMPLETED"
        })

    except Exception as e:
        logger.exception("Ошибка обработки:")
        return jsonify({"error": str(e), "status": "failed"}), 500


@app.route("/healthz", methods=["GET"])
def health_check():
    status = {"modelId": MODEL_ID, "status": "healthy"}
    try:
        dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
        manager.predict(dummy_input)
        return jsonify(status), 200
    except Exception as e:
        status["status"] = "unhealthy"
        status["error"] = str(e)
        return jsonify(status), 503


if __name__ == "__main__":
    start_http_server(9000)  # Метрики Prometheus
    app.run(host="0.0.0.0", port=8000, threaded=True)