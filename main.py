import pika
import json
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicalWorker:
    def __init__(self):
        self.connection = None
        self.channel = None

    def connect(self):
        """Подключение к RabbitMQ с фиксированными параметрами"""
        credentials = pika.PlainCredentials('admin', 'password')
        parameters = pika.ConnectionParameters(
            host='localhost',
            port=5672,
            credentials=credentials
        )

        try:
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # Очистка и пересоздание очередей
            self.channel.queue_delete(queue='medical_tasks')
            self.channel.queue_delete(queue='medical_results')

            # Создание durable очередей
            self.channel.queue_declare(queue='medical_tasks', durable=True)
            self.channel.queue_declare(queue='medical_results', durable=True)

            logger.info("Успешное подключение к RabbitMQ")
            return True

        except Exception as e:
            logger.error(f"Ошибка подключения: {str(e)}")
            return False

    def process_task(self, task):
        """Обработка задачи (заглушка)"""
        # Ваша реальная логика обработки
        return {
            "task_id": task.get("task_id"),
            "status": "completed",
            "result": "Sample result"
        }

    def callback(self, ch, method, properties, body):
        """Обработчик сообщений"""
        try:
            task = json.loads(body)
            logger.info(f"Получена задача: {task}")

            result = self.process_task(task)

            self.channel.basic_publish(
                exchange='',
                routing_key='medical_results',
                body=json.dumps(result),
                properties=pika.BasicProperties(
                    delivery_mode=2  # persistent message
                )
            )
            logger.info(f"Отправлен результат: {result}")

        except Exception as e:
            logger.error(f"Ошибка обработки: {str(e)}")

    def start(self):
        """Запуск сервиса"""
        if not self.connect():
            return

        try:
            self.channel.basic_consume(
                queue='medical_tasks',
                on_message_callback=self.callback,
                auto_ack=True
            )

            logger.info("Ожидание задач...")
            self.channel.start_consuming()

        except KeyboardInterrupt:
            logger.info("Остановка по запросу пользователя")
        except Exception as e:
            logger.error(f"Ошибка в работе сервиса: {str(e)}")
        finally:
            if self.connection and self.connection.is_open:
                self.connection.close()


if __name__ == '__main__':
    worker = MedicalWorker()
    worker.start()
