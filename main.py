import pika
import json
import logging
import requests
import os

# Настройки
RABBIT_HOST = "localhost"
SPRING_ENDPOINT = "http://localhost:8080/api/results"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    # Список доступных моделей и их URL
    MODEL_ENDPOINTS = {
        "1": "http://localhost:8000/api/process", # horse_zebra.onnx
        "2": "http://localhost:8001/api/process"
    }

    worker = MedicalWorker()
    worker.start()