from random import random

import docker
import logging
import os
import uuid
import requests
from datetime import timedelta, datetime

client = docker.from_env()
logger = logging.getLogger(__name__)


def get_free_port():
    return 8000 + random.randint(1, 1000)  # Простой и неверный выбора порта


def launch_model_container(model_id=None, model_type="onnx", model_path=None):
    """
    Запускает контейнер под нужную модель.
    Если model_id не указан → генерируется автоматически.
    """

    model_id = model_id or f"user_model_{uuid.uuid4().hex[:8]}"
    container_name = f"ml-{model_id}"

    # Проверяем, есть ли уже такой контейнер
    existing = client.containers.list(filters={"name": container_name})
    if existing:
        logger.info(f"Контейнер {container_name} уже существует")
        return f"http://{container_name}:8000/api/process"

    # Путь к модели внутри контейнера
    internal_model_path = "/models/" + os.path.basename(model_path) if model_path else ""

    try:
        container = client.containers.run(
            image="crowdsourcing-ml:latest",
            name=container_name,
            ports={'8000/tcp': None},
            environment={
                "MODEL_ID": model_id,
                "MODEL_TYPE": model_type,
                "MODEL_PATH": internal_model_path
            },
            volumes={os.path.dirname(model_path): {"bind": "/models", "mode": "ro"}},
            labels={"app": "ml-service"},
            detach=True
        )
        logger.info(f"Модель {model_id} запущена")
        return f"http://{container.name}:8000/api/process"

    except Exception as e:
        logger.error(f"Не удалось запустить модель {model_id}: {e}")
        return None