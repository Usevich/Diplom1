import cv2
import numpy as np

# Пути к файлам модели YOLO
YOLO_CONFIG_PATH = "models/yolov3.cfg"
YOLO_WEIGHTS_PATH = "models/yolov3.weights"
YOLO_CLASSES_PATH = "models/coco.names"

# Загрузка имен классов
with open(YOLO_CLASSES_PATH, "r") as f:
    LABELS = f.read().strip().split("\n") # читаем классы и разделяем по строкам

# Генерируем цвета для каждого из классов
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Загрузка YOLO
yolo_net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)


def detect_objects_yolo(image_path):
    image = cv2.imread(image_path) # читаем файл
    (H, W) = image.shape[:2]  # получаем высоту и ширину
    # Уменьшаем изображение и меняем каналы
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # передаем данные в модель
    yolo_net.setInput(blob)
    # Получаем имена выходных слоев
    ln = yolo_net.getLayerNames()
    try:
        ln = [ln[i - 1] for i in yolo_net.getUnconnectedOutLayers()] # Выбор нужных слоев
    except IndexError:
        ln = []

    # Перенаправляем входные данные и получаем результаты
    layer_outputs = yolo_net.forward(ln)
    # создаем массивы для хранения элементов
    boxes = []
    confidences = []
    class_ids = []

    # Каждый вывод слоя
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:] # вероятности для каждого класса
            class_id = np.argmax(scores) # класс с наибольшей вероятностью
            confidence = scores[class_id] # уверенность для данного класса

            # Отбрасываем обнаружения меньше 0.5
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H]) # восстанавливаем размер изображения
                (centerX, centerY, width, height) = box.astype("int")
                # Находим верхний угол рамки
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # Добавляем результаты
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Применяем "non-maxima suppression" для исключения покрывающихся объектов?
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    results = [] # массив для хранения результатов
    if len(idxs) > 0: # если есть обнаружения
        for i in idxs.flatten(): # пробегаемся по обаружениям
            x, y, w, h = boxes[i] # получаем координаты
            # Добавляем в результат
            results.append((LABELS[class_ids[i]], confidences[i], (x, y, w, h)))

    return results


def annotate_image_with_yolo(image_path, detections):
    image = cv2.imread(image_path) # читаем изображение
    for (label, confidence, box) in detections: # проходимся по обнаруженым объектам
        (x, y, w, h) = box # получаем координаты рамки
        color = [int(c) for c in COLORS[LABELS.index(label)]] # получаем цвет для класса
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2) # рисуем прямоугольник
        text = f"{label}: {confidence:.2f}" # подпись к прямоугольнику
        # рисуем текст
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Сохраняем изображение
    annotated_image_path = f"{image_path}_yolo_processed.jpg"
    cv2.imwrite(annotated_image_path, image)

    return annotated_image_path