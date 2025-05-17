import os

# Устанавливаем правильный Docker сокет для Windows
os.environ["DOCKER_HOST"] = "npipe:////./pipe/docker_engine"

import random
import docker
import logging
import uuid
import requests
from datetime import timedelta, datetime

client = docker.from_env()
print(client.version())
logger = logging.getLogger(__name__)


def get_free_port():
    return 8000 + random.randint(1, 1000)  # Простой и неверный выбора порта


def launch_model_container(model_id=None, model_type="onnx", model_path=None):
    model_id = model_id or f"user_model_{uuid.uuid4().hex[:8]}"
    container_name = f"ml-{model_id}"

    existing = client.containers.list(filters={"name": container_name})
    if existing:
        logger.info(f"Контейнер {container_name} уже существует")
        port_binding = existing[0].attrs['NetworkSettings']['Ports']['8000/tcp'][0]['HostPort']
        return f"http://localhost:{port_binding}/api/process"

    internal_model_path = "/models/" + os.path.basename(model_path) if model_path else ""

    try:
        container = client.containers.run(
            image="crowdsourcing-ml:latest",
            name=container_name,
            ports={'8000/tcp': ('127.0.0.1', 0)},
            environment={
                "MODEL_ID": model_id,
                "MODEL_TYPE": model_type,
                "MODEL_PATH": internal_model_path
            },
            volumes={os.path.dirname(model_path): {"bind": "/models", "mode": "ro"}},
            labels={"app": "ml-service"},
            detach=True
        )

        container.reload()  # обновляем данные контейнера, чтобы получить порт
        port_binding = container.attrs['NetworkSettings']['Ports']['8000/tcp'][0]['HostPort']
        logger.info(f"Модель {model_id} запущена на порту {port_binding}")
        return f"http://localhost:{port_binding}/api/process"  # <---- вот тут важно

    except Exception as e:
        logger.error(f"Не удалось запустить модель {model_id}: {e}")
        return None
