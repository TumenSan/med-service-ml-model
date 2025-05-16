FROM nvidia/cuda-python:12.1.1-base

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential python3-pip libgl1 libsm6 ffmpeg zlib1g-dev libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "MLService.py"]