import pika
import json
import logging
import requests
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Класс маршрутизации задач на соответствующие ML-сервисы
class MLRouter:
    model_endpoints = {
        "dicom_analysis": "http://localhost:8000/api/process",
        "lab_analysis":   "http://lab-service:8001/api/process"
    }

    @staticmethod
    def route_task(task: dict) -> dict:
        task_type = task.get("type") or task.get("taskType") # поддержка обоих записей
        url = MLRouter.model_endpoints.get(task_type)
        if not url:
            error_msg = f"unsupported_task_type: {task_type}"
            logger.error(error_msg)
            return {"error": error_msg, "task_id": task.get("task_id"), "status": "failed"}

        try:
            response = requests.post(url, json=task, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Ошибка обращения к модели {task_type}: {e}")
            return {"error": str(e), "task_id": task.get("task_id"), "status": "failed"}


class MedicalWorker:
    RABBIT_HOST = 'localhost'
    TASKS_QUEUE = 'medical_tasks'
    RESULTS_QUEUE = 'medical_results'

    def __init__(self):
        self.connection = None
        self.channel = None

    def connect(self) -> bool:
        credentials = pika.PlainCredentials('admin', 'password')
        params = pika.ConnectionParameters(
            host=self.RABBIT_HOST,
            port=5672,
            credentials=credentials,
            heartbeat=600,
            connection_attempts=3,
            retry_delay=5
        )

        try:
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()
            # Объявляем durable очереди
            self.channel.queue_declare(queue=self.TASKS_QUEUE, durable=True)
            self.channel.queue_declare(queue=self.RESULTS_QUEUE, durable=True)
            # QoS (по одной задаче на воркер)
            self.channel.basic_qos(prefetch_count=1)

            logger.info("Успешное подключение к RabbitMQ")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения: {e}")
            return False

    def send_to_queue(self, result: dict):
        """Отправка результата в очередь medical_results"""
        try:
            self.channel.basic_publish(
                exchange='',
                routing_key=self.RESULTS_QUEUE,
                body=json.dumps(result),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            logger.info(f"Результат записан в очередь {self.RESULTS_QUEUE}")
        except Exception as e:
            logger.error(f"Не удалось отправить в очередь: {e}")

    def send_to_spring(self, result: dict) -> bool:
        """Пытаемся отправить результат напрямую в Spring Boot"""
        try:
            resp = requests.post("http://localhost:8080/api/results", json=result, timeout=10)
            resp.raise_for_status()
            logger.info("Результат успешно отправлен в Spring Boot")
            return True
        except Exception as e:
            logger.warning(f"Не удалось отправить в Spring: {e}")
            return False

    def process_task(self, task: dict) -> dict:
        return MLRouter.route_task(task)

    def callback(self, ch, method, properties, body):
        try:
            task = json.loads(body)
            logger.info(f"Получена задача: {task}")

            result = self.process_task(task)
            logger.info(f"Результат от модели: {result}")
            #result['timestamp'] = datetime.utcnow().isoformat()

            adapted_result = {
                "taskId": result.get("task_id", task.get("taskId")),
                "modelId": result.get("model_id", task.get("modelId")),
                "data": result.get("data", "unknown"),
                "conclusion": result.get("conclusion", "no_conclusion"),
                "status": "COMPLETED"  # можно адаптировать под бизнес-логику
            }

            # Отправляем результат в Spring Boot
            try:
                response = requests.post(
                    "http://localhost:8080/api/results",
                    json=adapted_result,
                    timeout=10
                )
                response.raise_for_status()
                logger.info("Результат успешно отправлен в Spring")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Не удалось отправить в Spring: {e}")

            # Продублируем в очередь
            self.channel.basic_publish(
                exchange='',
                routing_key='medical_results',
                body=json.dumps(adapted_result),
                properties=pika.BasicProperties(
                    delivery_mode=2  # persistent
                )
            )
            logger.info("Результат записан в очередь medical_results")

        except Exception as e:
            logger.error(f"Ошибка обработки: {str(e)}")
        finally:
            # Подтверждаем приём
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def start(self):
        if not self.connect():
            return

        logger.info("Ожидание задач в очереди medical_tasks...")
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
            if self.connection and self.connection.is_open:
                self.connection.close()
                logger.info("Соединение RabbitMQ закрыто")


if __name__ == '__main__':
    worker = MedicalWorker()
    worker.start()
