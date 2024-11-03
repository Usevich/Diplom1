import cv2
import numpy as np
import os

# Расположение фалов модели
model_path = "models/mobilenet_iter_73000.caffemodel"
config_path = "models/mobilenet_ssd_deploy.prototxt"


# Загружаем модель
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Классы которая модель способна распознавать
class_names = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]


def detect_objects(image_path):
    # загружаем изображение
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    # подготовка изображения к загрузке в модель
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    # Передаем подготовленое изображение в сеть
    net.setInput(blob)
    # Запускаем нейлосеть
    detections = net.forward()

    results = [] # Список для хранения результатов
    for i in range(detections.shape[2]):
        # получаем вероятность для каждого входения объекта
        confidence = detections[0, 0, i, 2]

        # Устанавливаем порго вероятности Ю 50 процентов
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Добавляем обнаруженый объект в результаты
            results.append((class_names[idx], confidence, (startX, startY, endX, endY)))
    return results


def annotate_image_with_detections(image_path, detections):
    image = cv2.imread(image_path)
    # Проходимся по всем обнаруженым объектам
    for (label, confidence, box) in detections:
        # Получаем координаты углов прямоугольника
        (startX, startY, endX, endY) = box
        # формируем подпись к прямоугольнику
        label_text = f"{label}: {confidence:.2}"
        # рисуем прямоугольник
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
        # определяем положение подписи к прямоугольнику и рисуем текстовую метку
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label_text, (startX, y), cv2.FONT_ITALIC, 0.5, (255, 255, 0), 2)
    # Задаем имя нового файла
    filename = f"{image_path.split('/')[-1].split('.')[0]}_processed.jpg"
    annotated_image_path = os.path.join(os.path.dirname(image_path), 'processed', filename)
    # созраняем новое изображение
    cv2.imwrite(annotated_image_path, image)
    return annotated_image_path
