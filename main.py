import pika
import json


def process_dicom_analysis(task_data):
    # Обработка DICOM файлов
    print(f"Анализ DICOM файла {task_data['file_id']}")
    # Здесь ML-логика для DICOM
    return {
        "type": "dicom_analysis",
        "file_id": task_data["file_id"],
        "result": "normal",
        "confidence": 0.92
    }


def process_lab_results(task_data):
    # Обработка лабораторных результатов
    print(f"Анализ лабораторных данных пациента {task_data['patient_id']}")
    # Здесь аналитическая логика
    return {
        "type": "lab_analysis",
        "patient_id": task_data["patient_id"],
        "indicators": task_data["indicators"],
        "diagnosis": "diabetes",
        "severity": "moderate"
    }


def process_task(task):
    # Логика обработки
    if not isinstance(task, dict):
        return {"error": "invalid_task_format"}

    task_type = task.get("type")

    if not task_type:
        print("Получена задача без типа")
        return {"error": "missing_task_type"}

    print(f"Определен тип задачи: {task_type}")

    handlers = {
        "dicom_analysis": process_dicom_analysis,
        "lab_analysis": process_lab_results,
    }

    handler = handlers.get(task_type)
    if not handler:
        return {"error": f"unsupported_task_type: {task_type}"}

    try:
        return handler(task)
    except Exception as e:
        return {"error": str(e)}
    # return {"status": "processed", "task": task}


def callback(ch, method, properties, body):
    task = json.loads(body)
    print(f"Обработка задачи: {task}")
    result = process_task(task)
    channel.basic_publish(
        exchange='',
        routing_key='medical_results',
        body=json.dumps(result)
    )


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

        # Очистка и пересоздание очередей (для разработки)
        channel.queue_delete(queue='medical_tasks')
        channel.queue_delete(queue='medical_results')

        # Создание durable очередей
        channel.queue_declare(queue='medical_tasks', durable=True)
        channel.queue_declare(queue='medical_results', durable=True)

        channel.basic_consume(
            queue='medical_tasks',
            on_message_callback=callback,
            auto_ack=True
        )

        print('Ожидание задач...')
        channel.start_consuming()

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        if 'connection' in locals() and connection.is_open:
            connection.close()
