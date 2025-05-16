import docker
import time
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
client = docker.from_env()

IDLE_TIMEOUT = timedelta(minutes=10)  # Время до удаления
CHECK_INTERVAL = 60  # Каждую минуту проверяем


def is_container_idle(container):
    try:
        res = requests.get(f"http://{container.name}:8000/healthz", timeout=5)
        last_used = datetime.fromisoformat(res.json()['lastUsed'])
        return datetime.now() - last_used > IDLE_TIMEOUT
    except Exception as e:
        logger.warning(f"Не могу получить статус {container.name}: {e}")
        return True  # Если сервис недоступен — удаляем


def cleaner_loop():
    while True:
        containers = client.containers.list(filters={"label": "ml-service"})

        for container in containers:
            if is_container_idle(container):
                logger.info(f"Удаляем неактивный контейнер: {container.name}")
                container.stop()
                container.remove()

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    cleaner_loop()