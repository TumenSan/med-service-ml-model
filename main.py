import pika
import json
import logging
import requests
import hashlib
import redis
import asyncio  # Добавлен импорт
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os

# Конфигурация из переменных окружения
RABBIT_HOST = os.getenv('RABBIT_HOST', 'localhost')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
SPRING_ENDPOINT = os.getenv('SPRING_ENDPOINT', 'http://localhost:8080/api/results')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=0)


class MLRouter:
    MODEL_ENDPOINTS = {
        "dicom_analysis": os.getenv('DICOM_ENDPOINT', 'http://localhost:8000/api/process'),
        "lab_analysis": os.getenv('LAB_ENDPOINT', 'http://localhost:8001/api/process')
    }

    @staticmethod
    def get_cache_key(task: dict) -> str:
        task_data = json.dumps(task, sort_keys=True).encode()
        return hashlib.sha256(task_data).hexdigest()

    @staticmethod
    def route_task(task: dict) -> dict:
        # Отключаем кэш для тестирования
        # cache_key = MLRouter.get_cache_key(task)
        # cached = redis_client.get(cache_key)

        # if cached:
        #     logger.info(f"Cache hit for key {cache_key}")
        #     return json.loads(cached)

        task_type = task.get("type") or task.get("taskType")
        url = MLRouter.MODEL_ENDPOINTS.get(task_type)

        if not url:
            error_msg = f"unsupported_task_type: {task_type}"
            logger.error(error_msg)
            result = {"error": error_msg, "task_id": task.get("taskId"), "status": "failed"}
            # redis_client.setex(cache_key, 300, json.dumps(result))
            return result

        try:
            response = requests.post(url, json=task, timeout=30)
            response.raise_for_status()
            result = response.json()
            # redis_client.setex(cache_key, 3600, json.dumps(result))
            return result
        except Exception as e:
            logger.error(f"Ошибка обращения к модели {task_type}: {e}")
            return {"error": str(e), "task_id": task.get("taskId"), "status": "failed"}


class MedicalWorker:
    TASKS_QUEUE = 'medical_tasks'
    RESULTS_QUEUE = 'medical_results'
    WORKERS = int(os.getenv('WORKER_THREADS', 4))

    def __init__(self):
        self.connection = None
        self.channel = None
        self.executor = ThreadPoolExecutor(max_workers=self.WORKERS)

    def connect(self) -> bool:
        credentials = pika.PlainCredentials(
            os.getenv('RABBIT_USER', 'admin'),
            os.getenv('RABBIT_PASS', 'password')
        )

        params = pika.ConnectionParameters(
            host=RABBIT_HOST,
            port=5672,
            credentials=credentials,
            heartbeat=600,
            ssl_options=pika.SSLOptions() if os.getenv('SSL_ENABLED') else None
        )

        try:
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.TASKS_QUEUE, durable=True)
            self.channel.queue_declare(queue=self.RESULTS_QUEUE, durable=True)
            self.channel.basic_qos(prefetch_count=self.WORKERS * 2)
            logger.info("Успешное подключение к RabbitMQ")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения: {e}")
            return False

    def process_task(self, task: dict) -> dict:
        return MLRouter.route_task(task)

    def callback(self, ch, method, properties, body):
        try:
            task = json.loads(body)
            logger.info(f"Получена задача: {task['taskId']}")

            # Синхронная обработка через пул потоков
            future = self.executor.submit(self.process_task, task)
            result = future.result(timeout=30)

            adapted_result = {
                "taskId": result.get("task_id", task.get("taskId")),
                "modelId": result.get("model_id", task.get("modelId")),
                "data": result.get("data"),
                "conclusion": result.get("conclusion", "no_conclusion"),
                "status": "COMPLETED"
            }

            self.send_result(adapted_result)

        except Exception as e:
            logger.error(f"Ошибка обработки: {str(e)}")
        finally:
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def send_result(self, result: dict):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    SPRING_ENDPOINT,
                    json=result,
                    timeout=10
                )
                response.raise_for_status()
                logger.info(f"Результат {result['taskId']} отправлен в Spring")
                return
            except Exception as e:
                logger.warning(f"Попытка {attempt + 1}: Ошибка отправки в Spring: {e}")
                if attempt == max_retries - 1:
                    self.channel.basic_publish(
                        exchange='',
                        routing_key=self.RESULTS_QUEUE,
                        body=json.dumps(result),
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                    logger.info(f"Результат {result['taskId']} записан в очередь")

    def start(self):
        if not self.connect():
            return

        logger.info(f"Запуск {self.WORKERS} воркеров...")
        self.channel.basic_consume(
            queue=self.TASKS_QUEUE,
            on_message_callback=self.callback,
            auto_ack=False
        )

        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Работа остановлена пользователем")
        finally:
            self.executor.shutdown()
            if self.connection:
                self.connection.close()


if __name__ == '__main__':
    worker = MedicalWorker()
    worker.start()
