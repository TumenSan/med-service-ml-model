import pika

credentials = pika.PlainCredentials('admin', 'password')
parameters = pika.ConnectionParameters(
    host='localhost',
    port=5672,
    credentials=credentials
)


def callback(ch, method, properties, body):
    print(f"Получено сообщение: {body.decode()}")


if __name__ == '__main__':
    try:
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        channel.queue_declare(queue='medical_queue')
        channel.basic_consume(queue='medical_queue',
                              on_message_callback=callback,
                              auto_ack=True)

        print('Ожидание сообщений...')
        channel.start_consuming()

    except Exception as e:
        print(f"Ошибка подключения: {e}")
