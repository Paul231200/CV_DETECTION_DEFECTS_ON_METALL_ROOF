from __future__ import annotations
import argparse
import logging
import os
import time
import csv
import hashlib
import threading
from datetime import datetime, time as dtime, timezone, timedelta
from zoneinfo import ZoneInfo
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path

from publisher import Publisher
from mqtt_publisher import MQTTPublisher

# Установка уровня логирования на INFO для основной работы
logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)
logging.info("Приложение запущено Conf= 0.55")


EKATERINBURG_TZ = ZoneInfo("Asia/Yekaterinburg")

# Рабочее время: с 8:00 до 17:30
WORK_TIME_START = dtime(8, 0)
WORK_TIME_END = dtime(17, 30)

def is_work_time() -> bool:
    """
    Проверка, является ли текущее время рабочим в Екатеринбурге (с 8:00 до 17:30)
    
    :return: True если текущее время является рабочим, иначе False
    """
    # Текущее время в Екатеринбурге
    now = datetime.now(EKATERINBURG_TZ)
    current_time = now.time()
    
    # Проверка, находится ли текущее время между началом и концом рабочего времени
    is_work = WORK_TIME_START <= current_time <= WORK_TIME_END
    
    # Просто возвращаем результат без логирования
    
    return is_work

# Переопределение переменной окружения для ONNX Runtime
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"
os.environ["ORT_PROVIDERS"] = "CPUExecutionProvider"

import cv2 as cv
import numpy as np
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

# Удален импорт settings

# Настройка логирования
# logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO)

# Настройки
CONFIDENCE_THRESHOLD = 0.55  # Снизил порог уверенности для обнаружения дефектов
FRAME_SKIP = 0  # Пропуск каждого второго кадра
DETECTION_CONFIRMATION_THRESHOLD = 0  # Количество последовательных детекций для подтверждения
DETECTION_MEMORY_SIZE = 5  # Размер памяти для отслеживания детекций
SAVE_DELAY = 2.0  # Задержка между сохранениями одного и того же класса дефекта (в секундах)
MQTT_DEBOUNCE_SECONDS = 0.5  # Минимальный интервал между MQTT-триггерами по одному классу

# Область интереса (ROI) - только в этой области будут обрабатываться детекции
ROI_POINTS = np.array([(89, 445), (1829, 391), (1908, 680), (31, 741)], dtype=np.int32)

# Настройки для детектора листа
NO_LIST_DIR = os.environ.get('NO_LIST_DIR', '/app/No_list')  # Путь к папке с эталонными изображениями
SHEET_DETECTION_SCALE = float(os.environ.get('SHEET_DETECTION_SCALE', '0.5'))  # Масштаб для детекции листа
SHEET_DETECTION_FRAME_SKIP = int(os.environ.get('SHEET_DETECTION_FRAME_SKIP', '3'))  # Обрабатываем каждый N-й кадр
SHEET_DETECTION_CHECK_INTERVAL = float(os.environ.get('SHEET_DETECTION_CHECK_INTERVAL', '0.2'))  # Проверка каждые N секунд
SHEET_SSIM_THRESHOLD = float(os.environ.get('SHEET_SSIM_THRESHOLD', '0.65'))  # s < threshold = лист есть
SHEET_MAD_THRESHOLD = float(os.environ.get('SHEET_MAD_THRESHOLD', '25.0'))   # mad > threshold = лист есть
SHEET_DETECTION_DELAY = float(os.environ.get('SHEET_DETECTION_DELAY', '60.0'))  # Задержка в секундах перед запуском детекции дефектов

# Настройка ONNX Runtime для использования только CPU
try:
    import onnxruntime
    logging.info("ONNX Runtime успешно импортирован")
    # Установка политики глобально
    onnxruntime.set_default_logger_severity(3)  # Уровень WARNING (3) или ERROR (4)
    # Более современный способ установки провайдеров
    session_options = onnxruntime.SessionOptions()
    session_options.add_session_config_entry('session.set_denormal_as_zero', '1')
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    # Устанавливаем только CPU провайдер
    providers = ['CPUExecutionProvider']
    logging.info("ONNX Runtime настроен для использования CPU")
except ImportError:
    logging.info("ONNX Runtime не установлен")
except Exception as e:
    logging.warning(f"Ошибка при настройке ONNX Runtime: {e}")

class DefectTracker:
    """Класс для отслеживания и подтверждения дефектов на нескольких кадрах"""
    
    def __init__(self, confirmation_threshold: int, memory_size: int):
        """
        Инициализация трекера дефектов
        
        :param confirmation_threshold: Количество последовательных кадров для подтверждения дефекта
        :param memory_size: Размер памяти (количество кадров) для хранения истории дефектов
        """
        self.confirmation_threshold = confirmation_threshold
        self.detections_memory = {}  # Словарь для хранения истории детекций {class_id: deque([frame_counts])}
        self.confirmed_defects = set()  # Множество подтвержденных дефектов (class_id)
        self.memory_size = memory_size
        self.last_saved_time = {}  # Словарь для хранения времени последнего сохранения {class_id: timestamp}
    
    def update(self, detections: Dict[int, int]) -> Set[int]:
        """
        Обновление статуса детекций и возврат списка подтвержденных дефектов
        
        :param detections: словарь {class_id: count} с количеством детекций для каждого класса
        :return: множество подтвержденных дефектов (class_ids)
        """
        # Инициализация deque для новых классов
        for cls_id in detections:
            if cls_id not in self.detections_memory:
                self.detections_memory[cls_id] = deque([0] * self.memory_size, maxlen=self.memory_size)
        
        # Обновление счетчиков для текущего кадра
        for cls_id, count in detections.items():
            self.detections_memory[cls_id].append(count)
            
            # Проверка, есть ли достаточное количество последовательных детекций
            if sum(1 for x in self.detections_memory[cls_id] if x > 0) >= self.confirmation_threshold:
                self.confirmed_defects.add(cls_id)
        
        # Очистка памяти для отсутствующих классов
        for cls_id in list(self.detections_memory.keys()):
            if cls_id not in detections:
                self.detections_memory[cls_id].append(0)
                
            # Если в последних N кадрах нет детекций, удалить из подтвержденных
            if sum(self.detections_memory[cls_id]) == 0:
                if cls_id in self.confirmed_defects:
                    self.confirmed_defects.remove(cls_id)
                    
        return self.confirmed_defects
    
    def should_save(self, class_id: int) -> bool:
        """
        Проверка, нужно ли сохранять изображение для данного класса дефекта
        
        :param class_id: ID класса дефекта
        :return: True, если нужно сохранять, иначе False
        """
        current_time = time.time()
        
        # Если класс впервые обнаружен или прошло достаточно времени с последнего сохранения
        if class_id not in self.last_saved_time or (current_time - self.last_saved_time[class_id]) >= SAVE_DELAY:
            self.last_saved_time[class_id] = current_time
            return True
            
        return False

class SheetDetector:
    """Класс для определения наличия листа в станке"""
    
    def __init__(self, no_list_dir: str, scale: float = 0.5, ssim_threshold: float = 0.65, mad_threshold: float = 25.0):
        """
        Инициализация детектора листа
        
        :param no_list_dir: путь к папке с эталонными изображениями без листа
        :param scale: масштаб для обработки изображений
        :param ssim_threshold: порог SSIM для определения листа
        :param mad_threshold: порог MAD для определения листа
        """
        self.no_list_dir = no_list_dir
        self.scale = scale
        self.ssim_threshold = ssim_threshold
        self.mad_threshold = mad_threshold
        self.template_gray = None
        self.target_shape = None
        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.blur_ksize = (5, 5)
        
    def safe_imread(self, path):
        """Безопасное чтение изображения"""
        path = str(Path(path))
        try:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size > 0:
                img = cv.imdecode(data, cv.IMREAD_COLOR)
                if img is not None:
                    return img
        except Exception:
            pass
        try:
            img = cv.imread(path, cv.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass
        return None
    
    def preprocess_gray(self, img_bgr, scale=None):
        """Предобработка изображения для детекции листа"""
        if scale is None:
            scale = self.scale
            
        if scale != 1.0:
            img_bgr = cv.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        
        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        
        if self.blur_ksize is not None:
            gray = cv.GaussianBlur(gray, self.blur_ksize, 0)
        
        return gray
    
    def load_no_list_template(self, target_shape):
        """Загрузка эталонного шаблона без листа"""
        folder_path = Path(self.no_list_dir)
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP")
        files = []
        for ext in patterns:
            files.extend(folder_path.glob(ext))
        
        if not files:
            raise FileNotFoundError(f"В папке нет изображений: {self.no_list_dir}")

        processed, failed = [], []
        th, tw = target_shape
        
        for f in files:
            img = self.safe_imread(f)
            if img is None:
                failed.append(str(f))
                continue
            img = cv.resize(img, (tw, th), interpolation=cv.INTER_AREA)
            g = self.preprocess_gray(img, scale=1.0)
            processed.append(g)

        if not processed:
            err_txt = "Не удалось загрузить ни одного валидного эталонного изображения."
            if failed:
                err_txt += f"\nВсего файлов: {len(files)}. Не прочитались: {len(failed)}."
            raise RuntimeError(err_txt)

        # Используем медиану для создания эталона
        stack = np.stack(processed, axis=0)
        template = np.median(stack, axis=0).astype(np.uint8)
        
        logging.info(f"Эталонов успешно загружено: {len(processed)} из {len(files)}")
        if failed:
            logging.warning(f"Внимание: не прочитались {len(failed)} файлов.")
        
        return template
    
    def compute_metrics(self, cur_gray, template_gray, roi=None):
        """Вычисление метрик сравнения (SSIM и MAD)"""
        if roi is not None:
            x, y, w, h = roi
            cur_roi = cur_gray[y:y+h, x:x+w]
            tpl_roi = template_gray[y:y+h, x:x+w]
        else:
            cur_roi = cur_gray
            tpl_roi = template_gray
        
        # SSIM - основная метрика
        s = ssim(cur_roi, tpl_roi, data_range=255)
        
        # MAD - средняя абсолютная разность
        mad = float(np.mean(cv.absdiff(cur_roi, tpl_roi)))
        
        return s, mad
    
    def detect_sheet(self, gray, roi=None):
        """Определение наличия листа"""
        if self.template_gray is None:
            return False, 0.0, 0.0
            
        s, mad = self.compute_metrics(gray, self.template_gray, roi)
        
        # Ужесточенная логика: лист есть если SSIM низкий И MAD высокий (оба условия должны выполняться)
        has_sheet = (s < self.ssim_threshold) and (mad > self.mad_threshold)
        
        return has_sheet, s, mad
    
    def initialize(self, first_frame):
        """Инициализация детектора с первым кадром"""
        base_h, base_w = first_frame.shape[:2]
        proc_w = int(base_w * self.scale)
        proc_h = int(base_h * self.scale)
        self.target_shape = (proc_h, proc_w)
        
        logging.info(f"Инициализация детектора листа: размер входного кадра {base_w}x{base_h}, обработка {proc_w}x{proc_h}")
        
        try:
            self.template_gray = self.load_no_list_template(target_shape=(proc_h, proc_w))
            logging.info("Шаблон 'без листа' загружен успешно")
            return True
        except Exception as e:
            logging.error(f"Ошибка загрузки эталона для детекции листа: {e}")
        return False

def calculate_image_hash(image: np.ndarray) -> str:
    """
    Вычисление хеша изображения для кеширования результатов
    
    :param image: изображение в формате numpy array
    :return: строковый хеш изображения
    """
    # Уменьшить разрешение и конвертировать в оттенки серого для более эффективного хеширования
    small_img = cv.resize(image, (32, 32))
    gray_img = cv.cvtColor(small_img, cv.COLOR_BGR2GRAY) if len(small_img.shape) == 3 else small_img
    
    # Вычислить хеш изображения
    img_hash = hashlib.md5(gray_img.tobytes()).hexdigest()
    return img_hash

def convert_to_onnx(model_path: str) -> str:
    """
    Конвертация модели YOLO в формат ONNX
    
    :param model_path: путь к файлу модели YOLO
    :return: путь к сконвертированной модели ONNX или исходный путь, если конвертация не удалась
    """
    try:
        logging.info(f"Запуск процесса конвертации модели {model_path} в ONNX")
        onnx_path = os.path.splitext(model_path)[0] + ".onnx"
        
        # Проверяем, существует ли файл модели .pt
        if not os.path.exists(model_path):
            logging.error(f"Файл модели не существует: {model_path}")
            # Проверяем содержимое директории
            model_dir = os.path.dirname(model_path)
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                logging.info(f"Содержимое директории {model_dir}: {files}")
            return model_path
            
        logging.info(f"Размер файла модели .pt: {os.path.getsize(model_path)} байт")
        
        # Проверяем, существует ли уже ONNX файл
        if os.path.exists(onnx_path):
            logging.info(f"ONNX модель уже существует: {onnx_path}")
            logging.info(f"Размер ONNX модели: {os.path.getsize(onnx_path)} байт")
            
            if os.path.getsize(onnx_path) == 0:
                logging.warning(f"ONNX файл имеет нулевой размер, пробуем конвертировать заново")
                os.remove(onnx_path)
                logging.info(f"Удален пустой ONNX файл: {onnx_path}")
            else:
                # Проверяем валидность ONNX файла
                try:
                    import onnx
                    onnx_model = onnx.load(onnx_path)
                    onnx.checker.check_model(onnx_model)
                    logging.info(f"ONNX модель проверена и валидна")
                    return onnx_path
                except Exception as onnx_error:
                    logging.error(f"ONNX модель существует, но не валидна: {onnx_error}")
                    logging.info(f"Пробуем конвертировать модель заново")
                    os.remove(onnx_path)
                    logging.info(f"Удален некорректный ONNX файл: {onnx_path}")
        
        logging.info(f"Загрузка PyTorch модели из {model_path}")
        model = YOLO(model_path)
        logging.info(f"Модель PyTorch загружена успешно: {type(model)}")
        
        # Подробная информация о модели
        logging.info(f"Информация о модели: имя класса={model.__class__.__name__}, имена классов={model.names}")
        
        # Явно указываем использование CPU для экспорта
        logging.info(f"Начинаем экспорт в ONNX формат (dynamic=True, opset=12)...")
        try:
            result = model.export(format="onnx", dynamic=True, opset=12, simplify=True)
            logging.info(f"Результат экспорта: {result}")
        except Exception as export_error:
            logging.error(f"Ошибка при экспорте: {export_error}")
            
            # Пробуем другие настройки экспорта
            logging.info("Пробуем другие настройки экспорта ONNX...")
            try:
                result = model.export(format="onnx", dynamic=True, opset=11)
                logging.info(f"Экспорт с opset=11 успешен: {result}")
            except Exception as e2:
                logging.error(f"Вторая попытка экспорта тоже не удалась: {e2}")
                return model_path
        
        # Проверяем, создался ли ONNX файл
        if os.path.exists(onnx_path):
            logging.info(f"ONNX модель создана успешно: {onnx_path}")
            logging.info(f"Размер ONNX модели: {os.path.getsize(onnx_path)} байт")
            
            # Проверяем валидность ONNX файла
            try:
                import onnx
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                logging.info("ONNX модель успешно проверена, валидна")
            except Exception as onnx_error:
                logging.error(f"Созданная ONNX модель не валидна: {onnx_error}")
                return model_path
                
            return onnx_path
        else:
            logging.error(f"ONNX файл не был создан по пути: {onnx_path}")
            return model_path
            
    except Exception as e:
        logging.error(f"Ошибка конвертации модели в ONNX: {e}")
        logging.info("Будет использована исходная PyTorch модель")
        return model_path  # Возвращаем исходную модель в случае ошибки

def optimize_predict(model, frame: np.ndarray, conf_threshold: float, cached_results: Dict[str, any]) -> Optional[List]:
    """
    Оптимизированное предсказание с использованием кеширования и отслеживания
    
    :param model: модель YOLO
    :param frame: входной кадр
    :param conf_threshold: порог уверенности
    :param cached_results: словарь для кеширования результатов
    :return: результаты отслеживания
    """
    # Вычисление хеша изображения
    img_hash = calculate_image_hash(frame)
    
    # Проверка наличия в кеше
    if img_hash in cached_results:
        return cached_results[img_hash]
    
    # Выполняем отслеживание объектов с указанным порогом
    try:
        results = model.track(frame, conf=conf_threshold, verbose=False, persist=True)
    except AttributeError:
        results = model.predict(frame, conf=conf_threshold, verbose=False)
    
    # Проверяем результат
    if results and len(results) > 0:
        boxes = results[0].boxes
        logging.info(f"Результат отслеживания: обнаружено {len(boxes)} объектов")
    
    # Сохраняем в кеш
    cached_results[img_hash] = results
    
    # Если кеш слишком большой, удаляем старые записи
    if len(cached_results) > 100:  # Ограничение размера кеша
        for _ in range(10):  # Удаляем 10 старых записей
            cached_results.pop(next(iter(cached_results)), None)
            
    return results

def save_frame(frame: np.ndarray, class_name: str, defect_id: int, save_dir: str, boxes: list = None) -> str:
    """
    Сохранение кадра с дефектом для последующей разметки, с отрисовкой боксов
    :param frame: кадр для сохранения
    :param class_name: название класса дефекта
    :param defect_id: ID класса дефекта
    :param save_dir: директория для сохранения
    :param boxes: список боксов [(x1, y1, x2, y2), ...] для отрисовки
    :return: путь к сохраненному файлу
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{class_name}_{defect_id}.jpg"
    filepath = os.path.join(save_dir, filename)
    frame_to_save = frame.copy()
    
    # Отрисовываем ROI на сохраняемом изображении
    cv.polylines(frame_to_save, [ROI_POINTS], True, (0, 255, 0), 2)
    
    if boxes is not None:
        for (x1, y1, x2, y2) in boxes:
            cv.rectangle(frame_to_save, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    logging.info(f"Сохраняем изображение с дефектом {class_name} в {filepath}")
    cv.imwrite(filepath, frame_to_save)
    return filepath

def is_box_in_roi(box: Tuple[int, int, int, int], roi_points: np.ndarray) -> bool:
    """
    Проверяет, находится ли бокс внутри области интереса (ROI)
    Использует как центр бокса, так и углы для более надежного определения
    
    :param box: Координаты бокса в формате (x1, y1, x2, y2)
    :param roi_points: Точки, определяющие область интереса
    :return: True если бокс находится внутри ROI, иначе False
    """
    x1, y1, x2, y2 = box
    
    # Проверяем центр бокса
    center_x = int((x1 + x2) // 2)
    center_y = int((y1 + y2) // 2)
    center_point = (float(center_x), float(center_y))
    
    # Проверяем углы бокса
    top_left = (float(x1), float(y1))
    top_right = (float(x2), float(y1))
    bottom_left = (float(x1), float(y2))
    bottom_right = (float(x2), float(y2))
    
    # Проверяем центр и все углы
    center_in_roi = cv.pointPolygonTest(roi_points, center_point, False) >= 0
    tl_in_roi = cv.pointPolygonTest(roi_points, top_left, False) >= 0
    tr_in_roi = cv.pointPolygonTest(roi_points, top_right, False) >= 0
    bl_in_roi = cv.pointPolygonTest(roi_points, bottom_left, False) >= 0
    br_in_roi = cv.pointPolygonTest(roi_points, bottom_right, False) >= 0
    
    # Если хотя бы центр и один из углов внутри ROI, считаем, что объект внутри
    result = center_in_roi and (tl_in_roi or tr_in_roi or bl_in_roi or br_in_roi)
    
    # Если центр внутри, но все углы снаружи, это тоже считаем попаданием
    if center_in_roi and not (tl_in_roi or tr_in_roi or bl_in_roi or br_in_roi):
        result = True
    
    # Подробное логирование для отладки
    logging.debug(f"ROI проверка для бокса {box}: центр={center_in_roi}, углы={tl_in_roi},{tr_in_roi},{bl_in_roi},{br_in_roi}, результат={result}")
    
    return result

def log_detection(csv_path: str, image_path: str, class_name: str, timestamp: str) -> None:
    """
    Логирование информации о сохраненном изображении в CSV файл
    
    :param csv_path: путь к CSV файлу
    :param image_path: путь к сохраненному изображению
    :param class_name: название класса дефекта
    :param timestamp: время обнаружения
    """
    # Создание файла, если не существует
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'image_path', 'defect_class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            'timestamp': timestamp,
            'image_path': image_path,
            'defect_class': class_name
        })

def save_test_frame_on_startup(model, save_dir):
    """
    Создает и сохраняет тестовое изображение с детекциями при запуске
    :param model: загруженная модель YOLO
    :param save_dir: директория для сохранения
    """
    try:
        logging.info("Создание тестового изображения при запуске...")
        os.makedirs(save_dir, exist_ok=True)
        
        # Открываем видеопоток для получения одного кадра
        cap = cv.VideoCapture(os.environ.get('RTSP_URL', ''))
        if not cap.isOpened():
            logging.error("Не удалось открыть видеопоток для тестового изображения")
            # Создаем тестовое изображение с шаблоном
            test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            cv.putText(test_frame, "TEST IMAGE", (50, 320), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            cv.rectangle(test_frame, (100, 100), (540, 540), (0, 255, 0), 2)
        else:
            # Читаем один кадр
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                logging.error("Не удалось получить кадр из видеопотока для тестового изображения")
                test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                cv.putText(test_frame, "TEST IMAGE", (50, 320), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            else:
                test_frame = frame.copy()
        
        # Сохраняем исходный кадр
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filepath = os.path.join(save_dir, f"test_startup_original_{timestamp}.jpg")
        cv.imwrite(original_filepath, test_frame)
        logging.info(f"Исходное тестовое изображение сохранено: {original_filepath}")
        
        # Делаем предсказание
        try:
            logging.info("Запуск отслеживания на тестовом изображении...")
            try:
                results = model.track(test_frame, conf=CONFIDENCE_THRESHOLD, persist=True)
            except AttributeError:
                logging.warning("Метод track не найден, используем predict как запасной вариант")
                results = model.predict(test_frame, conf=CONFIDENCE_THRESHOLD)
                
            logging.info(f"Отслеживание выполнено, получено результатов: {len(results)}")
            
            # Получаем боксы и классы
            boxes = []
            detections_count = 0
            
            for r in results:
                boxes = r.boxes
                detections_count = len(boxes)
                
            logging.info(f"Обнаружено объектов: {detections_count}")
            
            # Создаем копию для отрисовки детекций
            annotated_frame = test_frame.copy()
            
            # Отрисовываем ROI
            cv.polylines(annotated_frame, [ROI_POINTS], True, (0, 255, 0), 2)
            
            if detections_count > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls_id]
                    
                    # Проверяем, находится ли объект в ROI
                    is_in_roi = is_box_in_roi((x1, y1, x2, y2), ROI_POINTS)
                    
                    # Цвет рамки: красный для объектов в ROI, серый для объектов вне ROI
                    color = (0, 0, 255) if is_in_roi else (128, 128, 128)
                    
                    # Рисуем бокс
                    cv.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv.putText(annotated_frame, f"{class_name}: {conf:.2f}", (x1, y1 - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    logging.info(f"Детекция: class={class_name}, conf={conf:.2f}, в ROI={is_in_roi}")
            
            # Сохраняем тестовое изображение с детекциями
            annotated_filepath = os.path.join(save_dir, f"test_startup_detections_{timestamp}.jpg")
            cv.imwrite(annotated_filepath, annotated_frame)
            logging.info(f"Тестовое изображение с детекциями сохранено: {annotated_filepath}")
            
        except Exception as predict_error:
            logging.error(f"Ошибка при выполнении отслеживания: {predict_error}")
        
    except Exception as e:
        logging.error(f"Ошибка при создании тестового изображения: {e}")

def calculate_iou(box1, box2):
    """
    Расчет IoU (Intersection over Union) между двумя боксами
    
    :param box1: первый бокс в формате (x1, y1, x2, y2)
    :param box2: второй бокс в формате (x1, y1, x2, y2)
    :return: значение IoU от 0 до 1
    """
    # Определяем координаты пересечения
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # Проверяем, есть ли пересечение
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Вычисляем площадь пересечения
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Вычисляем площади обоих боксов
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Вычисляем IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou


class SavedDefectsTracker:
    """
    Класс для отслеживания уже сохраненных дефектов на основе IoU
    """
    def __init__(self, iou_threshold=0.5, max_saved_defects=50):
        """
        Инициализация трекера сохраненных дефектов
        
        :param iou_threshold: порог IoU для определения одинаковых дефектов
        :param max_saved_defects: максимальное количество сохраняемых дефектов (для ограничения памяти)
        """
        self.saved_defects = []  # список сохраненных боксов
        self.iou_threshold = iou_threshold
        self.max_saved_defects = max_saved_defects
    
    def is_new_defect(self, box):
        """
        Проверка является ли бокс новым дефектом
        
        :param box: координаты бокса в формате (x1, y1, x2, y2)
        :return: True если дефект новый, False если похож на уже сохраненный
        """
        # Проверяем пересечение с каждым сохраненным дефектом
        for saved_box in self.saved_defects:
            iou = calculate_iou(box, saved_box)
            if iou > self.iou_threshold:
                logging.info(f"Повторный дефект обнаружен (IoU: {iou:.2f}), не сохраняется")
                return False
                
        return True
    
    def add_defect(self, box):
        """
        Добавление нового дефекта в список сохраненных
        
        :param box: координаты бокса в формате (x1, y1, x2, y2)
        """
        self.saved_defects.append(box)
        
        # Если превысили лимит, удаляем самый старый дефект
        if len(self.saved_defects) > self.max_saved_defects:
            self.saved_defects.pop(0)
            
        logging.info(f"Добавлен новый дефект в список отслеживания, всего: {len(self.saved_defects)}")

def send_defect_event(publisher, uuid_zone, event_state, delay_seconds=0, mqtt_publisher=None):
    """
    Отправка события о дефекте через AMQP и MQTT
    
    :param publisher: экземпляр Publisher для отправки событий через AMQP
    :param uuid_zone: UUID зоны наблюдения
    :param event_state: состояние события (1 - начало, 0 - окончание)
    :param delay_seconds: задержка перед отправкой в секундах
    :param mqtt_publisher: экземпляр MQTTPublisher для отправки в Wiren Board
    """
    if publisher is None and mqtt_publisher is None:
        return
    
    def send_event():
        # Сначала отправка через MQTT в Wiren Board - только событие включения (state=1)
        # Wirenboard сам отключит реле через 5 секунд по правилу
        if mqtt_publisher is not None and event_state == 1:
            try:
                t_mqtt_start = time.time()
                logging.info(f"[MQTT] start publish_defect_event_fast_both ts={t_mqtt_start:.3f}")
                # Отправляем сообщение в MQTT на K1 и K2
                result = mqtt_publisher.publish_defect_event_fast_both(event_state)
                t_mqtt_end = time.time()
                logging.info(f"[MQTT] end publish ok={result} duration_ms={(t_mqtt_end - t_mqtt_start) * 1000:.1f}")
                if result:
                    logging.info(f"Дефект обнаружен: сигнал отправлен в Wiren Board на K1 и K2")
            except Exception as e:
                logging.error(f"Ошибка при отправке в Wiren Board: {e}")

        # Затем отправка через AMQP
        if publisher is not None:
            try:
                t_amqp_start = time.time()
                logging.info(f"[AMQP] start publish ts={t_amqp_start:.3f}")
                publisher.publish('sauron', uuid_zone, 'Defect', event_state)
                t_amqp_end = time.time()
                logging.info(f"[AMQP] end publish duration_ms={(t_amqp_end - t_amqp_start) * 1000:.1f}")
                logging.info(f"Событие Defect отправлено через AMQP: event_state={event_state}")
            except Exception as e:
                logging.error(f"Ошибка при отправке события Defect через AMQP: {e}")
    
    if delay_seconds > 0:
        # Отложенная отправка через таймер
        timer = threading.Timer(delay_seconds, send_event)
        timer.daemon = True
        timer.start()
    else:
        # Немедленная отправка
        threading.Thread(target=send_event).start()

def main():
    parser = argparse.ArgumentParser()
    # Получаем RTSP URL из переменной окружения
    default_source = os.environ.get('RTSP_URL', '')
    parser.add_argument("--source", dest="source", default=default_source, help="Путь к видеофайлу или RTSP URL")
    parser.add_argument("--use-original", action="store_true", help="Использовать оригинальную модель без ONNX конвертации")
    parser.add_argument("--output", "-o", dest="output_dir", default="Defects_photo", help="Директория для сохранения обнаруженных дефектов")
    parser.add_argument("--test-image", dest="test_image", help="Путь к изображению для тестирования модели")
    parser.add_argument("--crop-roi", action="store_true", help="Вырезать только область ROI перед обработкой (улучшает производительность)")
    parser.add_argument("--model", dest="model_path", default="models/Defects_2.pt", help="Путь к локальной модели")
    args = parser.parse_args()
    
    # Принудительно используем оригинальную PyTorch модель
    args.use_original = True
    
    # Выводим информацию об источнике
    logging.info(f"Используемый источник: {args.source}")

    # Создаем директории для сохранения данных
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Файл для логирования сохраненных изображений
    log_file = os.path.join(save_dir, "detections_log.csv")

    # Используем локальную модель вместо загрузки из БД
    LOCAL_MODEL_PATH = args.model_path
    logging.info(f"Используем модель: {LOCAL_MODEL_PATH}")
    
    # Проверяем существование файла модели
    if os.path.exists(LOCAL_MODEL_PATH):
        # Выводим имя модели в терминал
        print("\n" + "=" * 80)
        print(f"ИСПОЛЬЗУЕМАЯ МОДЕЛЬ: {LOCAL_MODEL_PATH}")
        print(f"Файл модели существует, размер: {os.path.getsize(LOCAL_MODEL_PATH)} байт")
        print("=" * 80 + "\n")
    else:
        logging.error(f"Ошибка: локальная модель не найдена по пути {LOCAL_MODEL_PATH}")
        # Проверяем содержимое директории с моделями
        model_dir = os.path.dirname(LOCAL_MODEL_PATH)
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            logging.info(f"Содержимое директории с моделями: {files}")
            if files:
                # Используем первый найденный .pt файл
                pt_files = [f for f in files if f.endswith('.pt')]
                if pt_files:
                    LOCAL_MODEL_PATH = os.path.join(model_dir, pt_files[0])
                    logging.info(f"Используем первую найденную модель: {LOCAL_MODEL_PATH}")
                    print(f"Используем первую найденную модель: {LOCAL_MODEL_PATH}")
                else:
                    logging.error("Не найдено .pt файлов в директории models")
                    return
            else:
                logging.error("Директория с моделями пуста")
                return
        else:
            logging.error("Директория с моделями не существует")
            return
    
    # Загружаем модель напрямую из файла
    try:
        logging.info(f"Загрузка модели...")
        model = YOLO(LOCAL_MODEL_PATH, task='detect')
        logging.info("Модель успешно загружена")
        
        # Проверяем доступные классы
        if hasattr(model, 'names') and model.names:
            logging.info(f"Классы модели: {model.names}")
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели: {e}")
        return
    
    # Проверяем загруженную модель
    if model is not None:
        try:
            # Получаем информацию о классах
            class_names = model.names
            logging.info(f"Модель загружена успешно. Доступные классы: {class_names}")
            
            # Создаем тестовый кадр для проверки работоспособности
            test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            try:
                test_result = model.track(test_frame, verbose=False, persist=True)
            except AttributeError:
                logging.warning("Метод track не найден, используем predict как запасной вариант")
                test_result = model.predict(test_frame, verbose=False)
            logging.info(f"Тест модели пройден успешно. Результат: {len(test_result)} элементов")
            
            # Создаем и сохраняем тестовое изображение при запуске
            logging.info("Создание тестового изображения при запуске...")
            save_test_frame_on_startup(model, save_dir)

            # Если указано тестовое изображение, проверяем работу модели на нем
            if args.test_image and os.path.exists(args.test_image):
                logging.info(f"Тестирование модели на изображении: {args.test_image}")
                test_img = cv.imread(args.test_image)
                if test_img is not None:
                    # Получаем результаты детекции
                    try:
                        results = model.track(test_img, conf=CONFIDENCE_THRESHOLD, persist=True)
                    except AttributeError:
                        logging.warning("Метод track не найден, используем predict как запасной вариант")
                        results = model.predict(test_img, conf=CONFIDENCE_THRESHOLD)
                    
                    # Отрисовываем ROI
                    cv.polylines(test_img, [ROI_POINTS], True, (0, 255, 0), 2)
                    
                    # Отрисовываем результаты
                    boxes = []
                    valid_detections = 0
                    
                    for r in results:
                        boxes = r.boxes
                    
                    logging.info(f"Обнаружено объектов: {len(boxes)}")
                    
                    for i in range(len(boxes)):
                        box = boxes[i].xyxy[0].cpu().numpy().astype(int)
                        cls_id = int(boxes[i].cls[0])
                        conf = float(boxes[i].conf[0])
                        class_name = model.names[cls_id]
                        
                        # Проверяем, находится ли объект в ROI
                        is_in_roi = is_box_in_roi((box[0], box[1], box[2], box[3]), ROI_POINTS)
                        
                        # Цвет рамки: красный для объектов в ROI, серый для объектов вне ROI
                        color = (0, 0, 255) if is_in_roi else (128, 128, 128)
                        
                        if is_in_roi:
                            valid_detections += 1
                        
                        cv.rectangle(test_img, (box[0], box[1]), (box[2], box[3]), color, 2)
                        cv.putText(test_img, f"{class_name}: {conf:.2f}", (box[0], box[1] - 10),
                                 cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    logging.info(f"Из них в ROI: {valid_detections} объектов")
                    
                    # Сохраняем и показываем результат
                    output_path = os.path.join(args.output_dir, "test_detection_result.jpg")
                    cv.imwrite(output_path, test_img)
                    logging.info(f"Результат тестирования сохранен в: {output_path}")
                    
                    # Показываем изображение
                    # cv.imshow("Test Detection Result", test_img)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()
                    
                    if args.test_image:
                        logging.info("Тестирование завершено, выход из программы")
                        return
                else:
                    logging.error("Не удалось загрузить тестовое изображение")
            else:
                logging.info("Тестирование модели на изображении отключено.")
        except Exception as e:
            logging.error(f"Ошибка при тестировании модели: {e}")
    else:
        logging.error("Ошибка: модель не загружена")

    # Получаем UUID для публикации событий из переменных окружения
    uuid_publisher = os.environ.get('UUID_Publisher', 'default-uuid')
    
    # Инициализация издателей событий для отправки уведомлений о дефектах
    event_publisher = None
    mqtt_publisher = None
    
    # Инициализация AMQP Publisher
    try:
        event_publisher = Publisher()
        event_publisher.start()
        logging.info(f"AMQP Publisher успешно инициализирован с UUID: {uuid_publisher}")
    except Exception as e:
        logging.error(f"Ошибка при инициализации AMQP Publisher: {e}")
    
    # Инициализация MQTT Publisher для Wiren Board
    try:
        mqtt_publisher = MQTTPublisher()
        try:
            if mqtt_publisher.connect():
                logging.info("MQTT Publisher для Wiren Board успешно инициализирован")
                
                            # Тестовая отправка при инициализации
                logging.info("Тестовая отправка MQTT при запуске на K1 и K2")
                time.sleep(3)
                mqtt_publisher.publish_defect_event_fast_both(1)
            else:
                logging.warning("Не удалось подключиться к MQTT брокеру для Wiren Board, продолжаем без MQTT")
                mqtt_publisher = None
        except Exception as connect_error:
            logging.warning(f"Ошибка при подключении к MQTT брокеру: {connect_error}, продолжаем без MQTT")
            mqtt_publisher = None
    except Exception as e:
        logging.warning(f"Ошибка при инициализации MQTT Publisher: {e}, продолжаем без MQTT")
        mqtt_publisher = None

    # Инициализация детектора листа
    sheet_detector = SheetDetector(
        no_list_dir=NO_LIST_DIR,
        scale=SHEET_DETECTION_SCALE,
        ssim_threshold=SHEET_SSIM_THRESHOLD,
        mad_threshold=SHEET_MAD_THRESHOLD
    )
    
    # Пробуем открыть источник видео
    try:
        # Если источник число, конвертируем в int
        source = int(args.source) if str(args.source).isdigit() else args.source
        logging.info(f"Попытка открытия источника: {source}")
        
        # Для RTSP-потока используем FFMPEG с low-latency опциями
        if isinstance(source, str) and source.startswith("rtsp://"):
            logging.info("Открытие RTSP потока через FFMPEG (low-latency)...")
            try:
                # Настройки для минимальной задержки
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;tcp|max_delay;0|stimeout;2000000|buffer_size;102400|analyzeduration;0|probesize;32"
                )
            except Exception:
                pass
            cap = cv.VideoCapture(source, cv.CAP_FFMPEG)
            # Минимальный буфер
            try:
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        else:
            cap = cv.VideoCapture(source)
        
        # Проверка успешного открытия камеры
        if not cap.isOpened():
            logging.error(f"Не удалось открыть источник: {source}")
            
            # Если это была веб-камера, пробуем другие ID
            if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
                for alt_cam_id in range(3):  # Пробуем камеры с ID 0, 1, 2
                    logging.info(f"Пробуем альтернативную камеру с ID {alt_cam_id}")
                    cap = cv.VideoCapture(alt_cam_id)
                    if cap.isOpened():
                        logging.info(f"Успешно открыта камера с ID {alt_cam_id}")
                        # Получаем первый кадр для инициализации детектора листа
                        ret, test_frame = cap.read()
                        if ret:
                            logging.info(f"Первый кадр успешно получен, размер: {test_frame.shape}")
                            
                            # Инициализация детектора листа
                            if sheet_detector.initialize(test_frame):
                                logging.info("🟢 [ЛИСТ] Детектор листа инициализирован успешно")
                                logging.info(f"🟢 [ЛИСТ] Настройки: масштаб={SHEET_DETECTION_SCALE}, пропуск кадров={SHEET_DETECTION_FRAME_SKIP}, интервал={SHEET_DETECTION_CHECK_INTERVAL}с")
                                logging.info(f"🟢 [ЛИСТ] Пороги: SSIM<{SHEET_SSIM_THRESHOLD}, MAD>{SHEET_MAD_THRESHOLD}, задержка={SHEET_DETECTION_DELAY}с")
                                logging.info("🟢 [ЛИСТ] Система готова к мониторингу наличия листа в станке")
                            else:
                                logging.error("🔴 [ЛИСТ] Ошибка инициализации детектора листа, продолжаем без детекции листа")
                                sheet_detector = None
                        else:
                            logging.error("Не удалось получить первый кадр из альтернативной камеры")
                        break
            
            if not cap.isOpened():
                logging.error("Не удалось открыть ни один источник видео")
                return
        else:
            logging.info("Видеопоток успешно открыт!")
            ret, test_frame = cap.read()
            if ret:
                logging.info(f"Первый кадр успешно получен, размер: {test_frame.shape}")
                
                # Инициализация детектора листа
                if sheet_detector.initialize(test_frame):
                    logging.info("🟢 [ЛИСТ] Детектор листа инициализирован успешно")
                    logging.info(f"🟢 [ЛИСТ] Настройки: масштаб={SHEET_DETECTION_SCALE}, пропуск кадров={SHEET_DETECTION_FRAME_SKIP}, интервал={SHEET_DETECTION_CHECK_INTERVAL}с")
                    logging.info(f"🟢 [ЛИСТ] Пороги: SSIM<{SHEET_SSIM_THRESHOLD}, MAD>{SHEET_MAD_THRESHOLD}, задержка={SHEET_DETECTION_DELAY}с")
                    logging.info("🟢 [ЛИСТ] Система готова к мониторингу наличия листа в станке")
                else:
                    logging.error("🔴 [ЛИСТ] Ошибка инициализации детектора листа, продолжаем без детекции листа")
                    sheet_detector = None
            else:
                logging.error("Не удалось получить первый кадр из видеопотока")
    except Exception as e:
        logging.error(f"Ошибка при открытии источника: {e}")
        if 'sheet_detector' in locals():
            logging.info("🔴 [ЛИСТ] Система детекции листа остановлена из-за ошибки")
        return

    # Remove GUI initialization
    # cv.namedWindow('Defect Detection - Data Collection', cv.WINDOW_NORMAL)
    
    # Инициализация трекера дефектов
    defect_tracker = DefectTracker(
        confirmation_threshold=DETECTION_CONFIRMATION_THRESHOLD,
        memory_size=DETECTION_MEMORY_SIZE
    )
    
    # Переменные для управления детекцией дефектов
    defect_detection_enabled = False  # Включена ли детекция дефектов
    sheet_detection_timer = None  # Таймер для запуска детекции дефектов
    last_sheet_state = None  # Последнее состояние листа
    sheet_detection_frame_count = 0  # Счетчик кадров для детекции листа
    last_sheet_check_time = 0  # Время последней проверки листа
    last_ssim_value = 0.0  # Последнее значение SSIM
    last_mad_value = 0.0  # Последнее значение MAD
    last_periodic_log_time = 0  # Время последнего периодического вывода
    
    # Инициализация счетчиков для пропуска кадров
    frame_count = 0
    
    # Кеш для результатов предсказания
    cached_results = {}
    
    # Буфер последнего обработанного кадра с детекциями для отображения на пропускаемых кадрах
    last_processed_frame = None
    
    # Словарь для хранения количества детекций для каждого класса
    class_counts = defaultdict(int)
    
    # Счетчик сохраненных изображений
    saved_count = 0

    # Инициализация трекера дефектов и кэша предсказаний
    defect_tracker = DefectTracker(DETECTION_CONFIRMATION_THRESHOLD, DETECTION_MEMORY_SIZE)
    cached_results = {}
    
    # Инициализация трекера сохраненных дефектов для предотвращения дублирования
    saved_defects_tracker = SavedDefectsTracker(iou_threshold=0.5, max_saved_defects=100)

    # Последнее время отправки MQTT по каждому классу (для антидребезга)
    last_mqtt_sent_time = defaultdict(lambda: 0.0)

    # Набор классов, по которым уже отправлен сигнал в текущем "сеансе присутствия" (edge-trigger)
    active_signaled_classes = set()
    
    # Тестовая отправка перед началом основного цикла
    if mqtt_publisher is not None:
        logging.info("Тестовая отправка MQTT перед началом работы на K1 и K2")
        mqtt_publisher.publish_defect_event_fast_both(1)

    logging.info(f"Запуск сбора данных. Сохранение в директорию: {save_dir}")
    logging.info(f"Лог сохраняется в: {log_file}")
    
    # Логируем настройки детекции листа
    if sheet_detector is not None:
        logging.info("🟢 [ЛИСТ] Система детекции листа активна")
        logging.info(f"🟢 [ЛИСТ] Папка с эталонами: {NO_LIST_DIR}")
        logging.info(f"🟢 [ЛИСТ] Задержка включения детекции: {SHEET_DETECTION_DELAY} секунд")
        logging.info(f"🟢 [ЛИСТ] Детектор готов к инициализации с первым кадром")
    else:
        logging.warning("🔴 [ЛИСТ] Система детекции листа отключена")

    try:
        # Обработка видеопотока
        while True:
            # Скидываем несколько кадров, чтобы не читать устаревшие
            try:
                for _ in range(3):
                    cap.grab()
            except Exception:
                pass
            ret, frame = cap.read()
            if not ret:
                logging.warning("Поток прерван. Попытка переподключения...")
                time.sleep(1)
                cap.release()
                
                # Повторно открываем источник
                try:
                    if isinstance(source, str) and source.startswith("rtsp://"):
                        try:
                            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                                "rtsp_transport;tcp|max_delay;0|stimeout;2000000|buffer_size;102400|analyzeduration;0|probesize;32"
                            )
                        except Exception:
                            pass
                        cap = cv.VideoCapture(source, cv.CAP_FFMPEG)
                        try:
                            cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                    else:
                        cap = cv.VideoCapture(source)
                    
                    if not cap.isOpened():
                        logging.error("Не удалось переподключиться к источнику")
                        time.sleep(3)  # Ждем немного дольше перед следующей попыткой
                    else:
                        logging.info("Успешное переподключение к видеопотоку")
                        # Получаем первый кадр для инициализации детектора листа
                        ret, test_frame = cap.read()
                        if ret and sheet_detector is not None:
                            logging.info(f"Первый кадр после переподключения получен, размер: {test_frame.shape}")
                            
                            # Инициализация детектора листа
                            if sheet_detector.initialize(test_frame):
                                logging.info("🟢 [ЛИСТ] Детектор листа инициализирован после переподключения")
                            else:
                                logging.error("🔴 [ЛИСТ] Ошибка инициализации детектора листа после переподключения")
                                sheet_detector = None
                except Exception as e:
                    logging.error(f"Ошибка при переподключении: {e}")
                    time.sleep(3)
                
                continue
            
            # Обработка кадра с отрисовкой ROI на каждом кадре
            frame_count += 1
            
            # Skip frame logic without display
            if frame_count % (FRAME_SKIP + 1) != 0:
                continue

            # Сохраняем оригинальный кадр без изменений для возможного сохранения
            original_frame = frame.copy()
            
            # Логируем, что начинаем анализ кадра
            if defect_detection_enabled and frame_count % 100 == 0:  # Каждые 100 кадров
                logging.info(f"🔍 [ДЕТЕКЦИЯ] Анализ кадра {frame_count} - детекция дефектов активна")
            elif not defect_detection_enabled and frame_count % 100 == 0:
                logging.info(f"⏸️ [ДЕТЕКЦИЯ] Анализ кадра {frame_count} - детекция дефектов отключена")
            
            # Детекция листа (если детектор инициализирован)
            current_time = time.time()
            if sheet_detector is not None:
                sheet_detection_frame_count += 1
                
                # Проверяем лист только через заданный интервал
                if (current_time - last_sheet_check_time) >= SHEET_DETECTION_CHECK_INTERVAL:
                    # Пропускаем кадры для детекции листа
                    if sheet_detection_frame_count % SHEET_DETECTION_FRAME_SKIP == 0:
                        # Предобработка кадра для детекции листа
                        gray_for_sheet = sheet_detector.preprocess_gray(frame)
                        has_sheet, ssim_val, mad_val = sheet_detector.detect_sheet(gray_for_sheet)
                        
                        # Сохраняем последние значения для периодического вывода
                        last_ssim_value = ssim_val
                        last_mad_value = mad_val
                        
                        # Логируем состояние листа
                        status = "ЛИСТ ЕСТЬ" if has_sheet else "ЛИСТА НЕТ"
                        logging.info(f"[ЛИСТ] {status} | SSIM={ssim_val:.3f} MAD={mad_val:.2f}")
                        
                        # Проверяем изменение состояния листа
                        if last_sheet_state != has_sheet:
                            last_sheet_state = has_sheet
                            
                            if has_sheet:
                                # Лист появился - запускаем таймер на 60 секунд
                                logging.info("🟢 [ЛИСТ] Лист обнаружен! Запуск таймера на 60 секунд для включения детекции дефектов")
                                logging.info(f"🟢 [ЛИСТ] Параметры детекции: SSIM={ssim_val:.3f} (порог: {SHEET_SSIM_THRESHOLD}), MAD={mad_val:.2f} (порог: {SHEET_MAD_THRESHOLD})")
                                
                                if sheet_detection_timer is not None:
                                    logging.info("🟢 [ЛИСТ] Отменяем предыдущий таймер")
                                    sheet_detection_timer.cancel()
                                
                                def enable_defect_detection():
                                    nonlocal defect_detection_enabled
                                    defect_detection_enabled = True
                                    logging.info("🟢 [ЛИСТ] Детекция дефектов ВКЛЮЧЕНА (через 60 секунд после появления листа)")
                                    logging.info("🟢 [ЛИСТ] Система готова к обнаружению дефектов на металлическом листе")
                                
                                sheet_detection_timer = threading.Timer(SHEET_DETECTION_DELAY, enable_defect_detection)
                                sheet_detection_timer.daemon = True
                                sheet_detection_timer.start()
                                logging.info(f"🟢 [ЛИСТ] Таймер запущен, детекция дефектов будет включена через {SHEET_DETECTION_DELAY} секунд")
                            else:
                                # Лист исчез - отключаем детекцию дефектов
                                logging.info("🔴 [ЛИСТ] Лист исчез! Отключение детекции дефектов")
                                logging.info(f"🔴 [ЛИСТ] Параметры детекции: SSIM={ssim_val:.3f} (порог: {SHEET_SSIM_THRESHOLD}), MAD={mad_val:.2f} (порог: {SHEET_MAD_THRESHOLD})")
                                defect_detection_enabled = False
                                if sheet_detection_timer is not None:
                                    logging.info("🔴 [ЛИСТ] Отменяем таймер включения детекции дефектов")
                                    sheet_detection_timer.cancel()
                                    sheet_detection_timer = None
                                logging.info("🔴 [ЛИСТ] Детекция дефектов отключена - листа нет в станке")
                        
                        last_sheet_check_time = current_time
            
            # Периодический вывод SSIM и MAD значений раз в 30 секунд
            if sheet_detector is not None and (current_time - last_periodic_log_time) >= 30.0:
                status_text = "ЛИСТ ЕСТЬ" if last_sheet_state else "ЛИСТА НЕТ" if last_sheet_state is not None else "ОПРЕДЕЛЯЕТСЯ"
                logging.info(f"📊 [ЛИСТ] Периодический отчет (каждые 30 сек): Статус={status_text} | SSIM={last_ssim_value:.3f} (порог: <{SHEET_SSIM_THRESHOLD}) | MAD={last_mad_value:.2f} (порог: >{SHEET_MAD_THRESHOLD})")
                last_periodic_log_time = current_time
            
            # Проверяем, включена ли детекция дефектов
            if not defect_detection_enabled:
                # Логируем состояние каждые 30 секунд, чтобы не засорять логи
                if frame_count % 300 == 0:  # Примерно каждые 30 секунд при 10 FPS
                    if last_sheet_state is None:
                        logging.info("🟡 [ЛИСТ] Ожидание первого определения состояния листа...")
                    elif last_sheet_state:
                        if sheet_detection_timer is not None:
                            remaining_time = max(0, SHEET_DETECTION_DELAY - (current_time - last_sheet_check_time))
                            logging.info(f"🟡 [ЛИСТ] Лист обнаружен, ожидание включения детекции дефектов через {remaining_time:.1f} секунд")
                        else:
                            logging.info("🟡 [ЛИСТ] Лист обнаружен, но таймер не активен")
                    else:
                        logging.info("🟡 [ЛИСТ] Листа нет в станке - детекция дефектов отключена")
                continue
            
            # Оптимизированное предсказание с кешированием
            results = optimize_predict(model, frame, CONFIDENCE_THRESHOLD, cached_results)
            
            # Логируем результаты детекции для диагностики
            if results and len(results) > 0:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    # Считаем количество объектов в ROI
                    boxes_in_roi = 0
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy().astype(int)
                        cls_id = int(boxes.cls[i])
                        conf = float(boxes.conf[i])
                        class_name = model.names[cls_id]
                        
                        is_in_roi = is_box_in_roi((box[0], box[1], box[2], box[3]), ROI_POINTS)
                        if is_in_roi:
                            boxes_in_roi += 1
                            # Детальное логирование каждого обнаруженного объекта в ROI
                            logging.info(f"ДЕТЕКЦИЯ В ROI: class={class_name}, id={cls_id}, conf={conf:.3f}, box={box}")
                        else:
                            logging.debug(f"Детекция вне ROI: class={class_name}, id={cls_id}, conf={conf:.3f}")
                    
                    if boxes_in_roi > 0:
                        logging.info(f"[ROI] ДЕТЕКЦИЯ: обнаружено {boxes_in_roi} объектов из {len(boxes)} всего")
                    else:
                        logging.info(f"ДЕТЕКЦИЯ: обнаружено {len(boxes)} объектов вне ROI")
                else:
                    logging.info("Нет детекций в текущем кадре")
            else:
                logging.info("Модель не вернула результаты для текущего кадра")
            
            # Сброс счетчиков для текущего кадра
            class_counts.clear()
            
            # Инициализируем переменные для безопасности
            confirmed_defects = set()
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                
                # Используем векторизованные операции numpy вместо циклов
                if len(boxes) > 0:
                    # Получаем все данные одновременно для векторизации операций
                    all_xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
                    all_conf = boxes.conf.cpu().numpy()
                    all_cls = boxes.cls.cpu().numpy().astype(np.int32)
                    
                    # Подсчет количества детекций для каждого класса, только если они в ROI
                    for i, cls_id in enumerate(all_cls):
                        x1, y1, x2, y2 = all_xyxy[i]
                        # Проверяем, находится ли обнаруженный объект в ROI
                        if is_box_in_roi((x1, y1, x2, y2), ROI_POINTS):
                            class_counts[int(cls_id)] += 1
                            logging.info(f"Подсчет детекций: класс {cls_id} в ROI, всего {class_counts[int(cls_id)]}")
                    
                    # Обновление трекера дефектов
                    confirmed_defects = defect_tracker.update(class_counts)
                    logging.info(f"Подтвержденные дефекты: {confirmed_defects}")
                    
                    # Отображение только подтвержденных дефектов, быстрый триггер MQTT (edge) и сохранение кадров
                    roi_confirmed_classes = set()
                    for i, (x1, y1, x2, y2) in enumerate(all_xyxy):
                        cls_id = int(all_cls[i])
                        
                        box_coords = (x1, y1, x2, y2)
                        
                        # Проверяем, находится ли объект в ROI и является ли дефект подтвержденным
                        if is_box_in_roi(box_coords, ROI_POINTS) and cls_id in confirmed_defects:
                            conf = float(all_conf[i])
                            class_name = model.names[cls_id]
                            
                            logging.info(f"Подтвержденный дефект: class={class_name}, id={cls_id}, conf={conf:.3f}")

                            # Отмечаем, что класс подтвержден и в ROI на этом кадре
                            roi_confirmed_classes.add(cls_id)

                            is_new_defect = saved_defects_tracker.is_new_defect(box_coords)
                            tracker_updated = False

                            # Edge-триггер MQTT: только при переходе из "не было" -> "появился"
                            if is_new_defect and cls_id not in active_signaled_classes:
                                now_ts = time.time()
                                if mqtt_publisher is not None and (now_ts - last_mqtt_sent_time[cls_id]) >= MQTT_DEBOUNCE_SECONDS:
                                    last_mqtt_sent_time[cls_id] = now_ts
                                    active_signaled_classes.add(cls_id)
                                    logging.info("Обнаружен дефект! Немедленная отправка сигнала в Wirenboard (fast path, edge)")
                                    send_defect_event(event_publisher, uuid_publisher, 1, mqtt_publisher=mqtt_publisher)
                                    # Завершение события через 1 секунду (только AMQP, MQTT не требуется для reset)
                                    send_defect_event(event_publisher, uuid_publisher, 0, delay_seconds=1, mqtt_publisher=None)
                                    saved_defects_tracker.add_defect(box_coords)
                                    tracker_updated = True
                            
                            # Проверяем, нужно ли сохранять этот кадр (независимо от MQTT)
                            if defect_tracker.should_save(cls_id):
                                # Проверяем рабочее время
                                if is_work_time():
                                    # Проверяем, является ли этот дефект новым (по IoU)
                                    if is_new_defect:
                                        # Собираем все боксы этого класса, которые в ROI и подтверждены
                                        save_boxes = []
                                        for j, (bx1, by1, bx2, by2) in enumerate(all_xyxy):
                                            if int(all_cls[j]) == cls_id and is_box_in_roi((bx1, by1, bx2, by2), ROI_POINTS) and cls_id in confirmed_defects:
                                                save_boxes.append((bx1, by1, bx2, by2))
                                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        def save_in_thread(original_frame, class_name, cls_id, save_dir, save_boxes, log_file, timestamp):
                                            saved_path = save_frame(original_frame, class_name, cls_id, save_dir, boxes=save_boxes)
                                            log_detection(log_file, saved_path, class_name, timestamp)
                                            logging.info(f"[ROI] Сохранен кадр с дефектом '{class_name}' (conf: {conf:.2f}): {saved_path}")

                                        threading.Thread(target=save_in_thread, args=(original_frame, class_name, cls_id, save_dir, save_boxes, log_file, timestamp)).start()
                                        saved_count += 1
                                        
                                        # Добавляем сохраненный дефект в трекер
                                        if not tracker_updated:
                                            saved_defects_tracker.add_defect(box_coords)
                                        
                                        # Сигнал уже отправлен в fast path выше; здесь только сохранение/лог
                                    else:
                                        logging.info(f"[ПОВТОР] Дефект '{class_name}' похож на ранее сохраненный, пропускаем (conf: {conf:.2f})")
                                else:
                                    logging.info(f"[НЕРАБОЧЕЕ ВРЕМЯ] Дефект '{class_name}' обнаружен, но НЕ сохранен (conf: {conf:.2f})")
            
            # Обновляем набор активных классов: оставляем только те, что все еще подтверждены и в ROI
            try:
                active_signaled_classes = active_signaled_classes.intersection(confirmed_defects)
                # Если вычисляли roi_confirmed_classes в этом кадре — ограничим активные присутствием в ROI
                if 'roi_confirmed_classes' in locals():
                    active_signaled_classes = active_signaled_classes.intersection(roi_confirmed_classes)
            except Exception:
                pass

            # Сохраняем статистику в логах
            roi_defects = sum(1 for cls_id in confirmed_defects)
            logging.info(f"Итого в кадре: дефектов в ROI={roi_defects}, всего сохранено={saved_count}")
    
    except KeyboardInterrupt:
        logging.info("Получен сигнал прерывания, завершаем работу...")
    except Exception as e:
        logging.exception(f"Необработанное исключение: {e}")
    finally:
        # Освобождаем ресурсы и завершаем работу
        if cap is not None:
            cap.release()
            
        # Останавливаем AMQP Publisher
        if event_publisher is not None:
            logging.info("Останавливаем AMQP Publisher...")
            event_publisher.stop()
        
        # Останавливаем MQTT Publisher
        if mqtt_publisher is not None:
            logging.info("Останавливаем MQTT Publisher...")
            mqtt_publisher.disconnect()
            
        # Останавливаем таймер детекции листа
        if 'sheet_detection_timer' in locals() and sheet_detection_timer is not None:
            logging.info("🔴 [ЛИСТ] Останавливаем таймер детекции листа...")
            sheet_detection_timer.cancel()
            
        logging.info("🔴 [ЛИСТ] Система детекции листа остановлена")
        logging.info("Работа программы завершена")


if __name__ == "__main__":
    main() 
    