from django.db import models
import json
from django.contrib.auth.models import User
from PIL import Image


class UploadedImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='media/images/')
    processed_image_mobilenet = models.ImageField(upload_to='media/images/', null=True, blank=True)
    processed_image_yolo = models.ImageField(upload_to='media/images/', null=True, blank=True)
    detected_objects = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    processing_time_mobilenet = models.FloatField(null=True, blank=True)  # Время обработки для MobileNet
    processing_time_yolo = models.FloatField(null=True, blank=True)  # Время обработки для YOLO

    def __str__(self):
        return f"{self.user.username} - {self.image.name}"

    def set_detected_objects_for_model(self, model_name, objects):
        detected_data = self.get_detected_objects()
        detected_data[model_name] = objects
        self.detected_objects = json.dumps(detected_data)

    def get_detected_objects(self):
        if self.detected_objects:
            return json.loads(self.detected_objects)
        return {}

    def get_image_size(self):
        with Image.open(self.image.path) as img:
            return img.size
