import pika
import json


def callback(ch, method, properties, body):
    task = json.loads(body)
    print(f"Получена задача: {task}")

    # Обработка
    result = process_task(task)

    # Отправка результата в другую очередь
    channel.basic_publish(
        exchange='',
        routing_key='medical_results',
        body=json.dumps(result)
    )
    print("Результат отправлен в очередь medical_results")


def process_task(task):
    # Здесь обработка ML-моделью
    return {"result": "diagnose", "confidence": 0.95}


if __name__ == '__main__':
    credentials = pika.PlainCredentials('admin', 'password')
    parameters = pika.ConnectionParameters(
        host='localhost',
        port=5672,
        credentials=credentials
    )
    try:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        # channel.queue_declare(queue='medical_tasks')
        channel.queue_declare(queue='medical_results')
        channel.queue_declare(queue='medical_queue') # tasks?
        channel.basic_consume(queue='medical_queue',
                              on_message_callback=callback,
                              auto_ack=True)

        print('Ожидание сообщений...')
        channel.start_consuming()

    except Exception as e:
        print(f"Ошибка подключения: {e}")
