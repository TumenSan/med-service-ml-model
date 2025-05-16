import onnxruntime as ort
import tensorflow as tf
from PIL import Image
import numpy as np
import logging
import io
import os

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_path, model_type):
        self.model_path = model_path
        self.model_type = model_type
        self.model = self.load_model()

    def load_model(self):
        """Загружает модель на основе MODEL_TYPE"""
        if self.model_type == "onnx":
            return ort.InferenceSession(self.model_path)
        elif self.model_type == "keras":
            return tf.keras.models.load_model(self.model_path)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def preprocess_image(self, image_bytes):
        """Общая функция предобработки изображения"""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((128, 128))
            img_array = np.array(img).astype(np.float32) / 255.0
            return np.expand_dims(img_array, axis=0).astype(np.float32)

        except Exception as e:
            logger.error(f"Ошибка предобработки: {e}")
            raise

    def predict(self, input_tensor):
        """Выполняет инференс модели"""
        if self.model_type == "onnx":
            inputs = {self.model.get_inputs()[0].name: input_tensor}
            return self.model.run(None, inputs)[0]
        elif self.model_type == "keras":
            return self.model.predict(input_tensor)
        else:
            raise ValueError(f"Невозможно выполнить инференс для типа {self.model_type}")