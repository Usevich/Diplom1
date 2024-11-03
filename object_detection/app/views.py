import os,time
from typing import List, Dict, Any, Type

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import UploadedImage
from .forms import ImageUploadForm
from .utils import detect_objects, annotate_image_with_detections
from django.http import HttpResponse
from django.core.exceptions import ObjectDoesNotExist
from django.conf import settings
from .yolo_utils import detect_objects_yolo, annotate_image_with_yolo

def home(request):
    return render(request, 'home.html')
CLASS_TRANSLATIONS = {
  'aeroplane': 'самолет',
  'bicycle': 'велосипед',
  'bird': 'птица',
  'boat': 'лодка',
  'bottle': 'бутылка',
  'bus': 'автобус',
  'car': 'автомобиль',
  'cat': 'кошка',
  'chair': 'стул',
  'dog': 'собака',
  'horse': 'лошадь',
  'person': 'человек',
  'sheep': 'овца',
  'train': 'поезд',
  'tvmonitor': 'телевизор',
  'sofa': 'диван',
}

@login_required  # Защита маршрута от незарегистрированых пользователей
def dashboard(request):
    # поллучаем изображения для конкретного пользователя
    images = UploadedImage.objects.filter(user=request.user)
    for image in images:
        # Проверяем обработанность для каждой модели
        image.is_processed_mobilenet = bool(image.processed_image_mobilenet)
        image.is_processed_yolo = bool(image.processed_image_yolo)
    return render(request, 'dashboard.html', {'images': images})


def login_view(request):
    if request.method == 'POST': # Обрабатывваем POST запрос
        form = AuthenticationForm(data=request.POST) # создаем фрму с данными для запроса
        if form.is_valid(): # Проверяем на корректность введеных данных
            user = form.get_user() # Получаем данные пользователя
            login(request, user) # входим
            return redirect('dashboard') # переходим на дашбоард
    else:
       form = AuthenticationForm() # если такого пользователя нет, то предлагаем зарегистрироваться
    return render(request, 'login.html', {'form': form})


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST) # форма регистрации
        if form.is_valid():
            user = form.save() # сохранение нового пользователья
            login(request, user) # вход пользователя
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'registration.html', {'form': form})


def logout_view(request):
    logout(request) # Выход пользователя
    return redirect('home')


@login_required
def add_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES) # форма загрузки изображения
        if form.is_valid():
            image = form.save(commit=False) # получаем изображение
            image.user = request.user # приязываем к пользователю
            image.save() # сохраняем изображение на сервере
            return redirect('dashboard')
    else:
        form = ImageUploadForm()
    return render(request, 'add_image_feed.html', {'form': form})


@login_required
def process_image(request, image_id): # обработка изображения
  image = get_object_or_404(UploadedImage, id=image_id, user=request.user)

  if not image.processed_image_mobilenet:
      start_time = time.time() # запоминаем время
      detect_objects_and_save(image)  # обрабатываем с  MobileNet SSD
      end_time = time.time() # опять запоминаем время
      processing_time = end_time - start_time # вычисляем время работы
      image.processing_time_mobilenet = processing_time
      image.save() # сохраняем изменения в базу данных

  if not image.processed_image_yolo:
      start_time_yolo = time.time()
      classify_objects_with_yolo_and_save(image)  # YOLO
      end_time_yolo = time.time()
      processing_time_yolo = end_time_yolo - start_time_yolo
      image.processing_time_yolo = processing_time_yolo
      image.save()
  return redirect('dashboard')

@login_required
def delete_image(request, image_id):
    image = get_object_or_404(UploadedImage, id=image_id, user=request.user)
    os.remove(image.image.path) # удаляем с сервера оригинальное изображение
    os.remove(image.processed_image_mobilenet.path) # удаляем избражение обработаное MobileNet
    os.remove(image.processed_image_yolo.path) # удаляем для YOLO
    image.delete() # удаляем запись из базы данных
    return redirect('dashboard')

def detect_objects_and_save(image):
  detections = detect_objects(image.image.path) # Запускаем обработку MobileNet
  # Рисуем найденые объекты
  processed_image_path = annotate_image_with_detections(image.image.path, detections)

  if processed_image_path is not None:
    # Поучаем относительный путь к изображению
    relative_path = os.path.relpath(processed_image_path, settings.MEDIA_ROOT)
    # Сохраняем относительный путь в модели
    image.processed_image_mobilenet = relative_path
    # Переводим на русский язык объекты (не все)
    objects = [{'class': translate_label(label) , 'confidence': float(confidence), 'box': [int(coord) for coord in box]} for (label, confidence, box) in detections]
    image.set_detected_objects_for_model('mobilenet', objects) # добавляем обнаруженые объекты
    image.save() # записываем изменения в базу

def classify_objects_with_yolo_and_save(image):
  detections = detect_objects_yolo(image.image.path)
  processed_image_path = annotate_image_with_yolo(image.image.path, detections)

  if processed_image_path is not None:
    relative_path = os.path.relpath(processed_image_path, settings.MEDIA_ROOT)
    image.processed_image_yolo = relative_path
    objects = [{'class': label , 'confidence': float(confidence), 'box': [int(coord) for coord in box]} for (label, confidence, box) in detections]
    image.set_detected_objects_for_model('yolo', objects)
    image.save()

def translate_label(label):
    return CLASS_TRANSLATIONS.get(label, label)