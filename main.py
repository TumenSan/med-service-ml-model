import os
import pika
import json
import logging
import requests
from flask import Flask, request, jsonify
import docker_launcher

print("DOCKER_HOST:", os.environ.get("DOCKER_HOST"))

# Настройки
RABBIT_HOST = "localhost"
SPRING_ENDPOINT = "http://localhost:8080/api/results"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Словарь доступных моделей: {model_id: service_url}
MODEL_ENDPOINTS = {}

class MedicalWorker:
    TASKS_QUEUE = 'medical_tasks'
    RESULTS_QUEUE = 'medical_results'

    def __init__(self):
        self.connection = None
        self.channel = None

    def connect(self):
        credentials = pika.PlainCredentials(username='admin', password='password')
        params = pika.ConnectionParameters(
            host=RABBIT_HOST,
            port=5672,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300
        )
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.TASKS_QUEUE, durable=True)

    def callback(self, ch, method, properties, body):
        try:
            task = json.loads(body)
            logger.info(f"Получена задача: {task['taskId']}")

            # Отправляем задачу в ML-сервис
            model_id = task.get("modelId")
            ml_service_url = MODEL_ENDPOINTS.get(model_id)

            if not ml_service_url:
                logger.warning(f"Модель {model_id} не найдена")
                return

            response = requests.post(ml_service_url, json=task, timeout=30)
            response.raise_for_status()

            # Отправляем результат в Spring Boot
            requests.post(SPRING_ENDPOINT, json=response.json())

        except Exception as e:
            logger.error(f"Ошибка обработки задачи: {e}")
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def start(self):
        self.connect()
        self.channel.basic_consume(
            queue=self.TASKS_QUEUE,
            on_message_callback=self.callback,
            auto_ack=False
        )
        logger.info("Ожидание задач...")
        self.channel.start_consuming()


@app.route("/api/add-model", methods=["POST"])
def add_model():
    data = request.get_json()
    model_id = data.get("modelId")
    model_type = data.get("modelType", "onnx")
    model_path = data.get("modelPath")

    url = docker_launcher.launch_model_container(model_id=model_id, model_type=model_type, model_path=model_path)

    if url:
        MODEL_ENDPOINTS[model_id] = url
        return jsonify({"status": "ok", "url": url})
    else:
        return jsonify({"status": "error"}), 500


# === Автозагрузка моделей из папки models/ ===
def load_models_from_folder(folder="models"):
    if not os.path.exists(folder):
        logger.warning(f"Папка {folder} не найдена")
        return

    for filename in os.listdir(folder):
        if filename.endswith(".onnx"):
            model_id = filename.replace(".onnx", "")
            model_path = os.path.join(folder, filename)

            logger.info(f"Запускаем модель: {model_id}")
            url = docker_launcher.launch_model_container(
                model_id=model_id,
                model_type="onnx",
                model_path=model_path
            )

            if url:
                MODEL_ENDPOINTS[model_id] = url
                logger.info(f"Модель {model_id} запущена по адресу {url}")
            else:
                logger.error(f"Не удалось запустить модель {model_id}")


# Запуск Flask API для приёма новых моделей в отдельном потоке
def start_api():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    # Список доступных моделей и их URL
    #MODEL_ENDPOINTS = {
    #    "1": "http://localhost:8000/api/process", # horse_zebra.onnx
    #    "2": "http://localhost:8001/api/process"
    #}

    from threading import Thread

    # 1. Загружаем модели из папки models/
    load_models_from_folder()

    # 2. Запускаем Flask API для добавления новых моделей
    api_thread = Thread(target=start_api)
    api_thread.daemon = True
    api_thread.start()

    # 3. Запускаем RabbitMQ worker
    worker = MedicalWorker()
    worker.start()