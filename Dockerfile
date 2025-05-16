FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Установка Python и pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip libgl1 libsm6 ffmpeg zlib1g-dev libjpeg-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements.txt
COPY requirements.txt .

# Обновляем pip и устанавливаем зависимости
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем исходные файлы
COPY . .

EXPOSE 8000

CMD ["python", "MLService.py"]