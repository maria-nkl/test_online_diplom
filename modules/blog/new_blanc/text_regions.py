# text_regions.py
import cv2
import numpy as np
from pathlib import Path
from utils import save_image


def find_text_boxes_direct(image):
    """Прямой поиск прямоугольных областей"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Адаптивная бинаризация для выделения темных границ
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 15)

    # Морфологические операции для соединения разрывов в границах
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    height, width = image.shape[:2]
    min_area = width * height * 0.02  # минимум 2% от площади

    for contour in contours:
        # Аппроксимируем контур
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Если это многоугольник с 4-8 сторонами (не обязательно идеальный прямоугольник)
        if len(approx) >= 4:  # принимаем любые многоугольники с 4+ сторонами
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h

            # Критерии для больших областей с границами
            if (area > min_area and
                    w > width * 0.15 and h > height * 0.03 and
                    w < width * 0.95 and h < height * 0.95):
                regions.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center_y': y + h / 2
                })

    # Сортируем по вертикальной позиции
    regions.sort(key=lambda r: r['center_y'])

    return regions

def classify_regions(regions, image_shape):
    """Классифицирует найденные регионы по их расположению и размеру"""
    height, width = image_shape[:2]
    classified = {}

    if not regions:
        return classified

    print(f"Regions for classification: {len(regions)}")

    # Сортируем по вертикали
    regions_sorted = sorted(regions, key=lambda r: r['center_y'])

    # Теперь ищем только 6 регионов (пропускаем самый верхний необрамленный)
    # Берем первые 6 регионов после сортировки (самые верхние, но header уже исключен)
    valid_regions = regions_sorted[:6]  # берем ровно 6 регионов

    # Классифицируем 6 регионов
    if len(valid_regions) >= 1:
        classified['student_name'] = valid_regions[0]  # первый - имя студента
        print(f"Region 1: STUDENT NAME")

    if len(valid_regions) >= 2:
        classified['test_title'] = valid_regions[1]  # второй - тестовая надпись
        print(f"Region 2: TEST TITLE")

    if len(valid_regions) >= 3:
        classified['tasks'] = valid_regions[2]  # третий - задания (будет слева)
        print(f"Region 3: TASKS")

    if len(valid_regions) >= 4:
        classified['short_answers'] = valid_regions[3]  # четвертый - краткие ответы (будет справа)
        print(f"Region 4: SHORT ANSWERS")

    if len(valid_regions) >= 5:
        classified['printed_text'] = valid_regions[4]  # пятый - печатные буквы
        print(f"Region 5: PRINTED TEXT")

    if len(valid_regions) >= 6:
        classified['cursive_text'] = valid_regions[5]  # шестой - прописной текст
        print(f"Region 6: CURSIVE TEXT")

    # Корректируем расположение для 3 и 4 регионов (должны быть рядом)
    if 'tasks' in classified and 'short_answers' in classified:
        tasks_x = classified['tasks']['bbox'][0]
        answers_x = classified['short_answers']['bbox'][0]

        # Если tasks справа, а answers слева - меняем местами
        if tasks_x > answers_x:
            classified['tasks'], classified['short_answers'] = classified['short_answers'], classified['tasks']
            print("Swapped TASKS and SHORT_ANSWERS positions")

    return classified


def draw_regions(image, regions, classified_regions):
    """Рисует найденные регионы на изображении"""
    result_image = image.copy()
    height, width = image.shape[:2]

    # Цвета для 6 регионов (header убран)
    colors = {
        'student_name': (0, 255, 0),  # Зеленый - имя студента
        'test_title': (0, 0, 255),  # Красный - тестовая надпись
        'tasks': (255, 255, 0),  # Голубой - задания
        'short_answers': (255, 0, 255),  # Фиолетовый - краткие ответы
        'printed_text': (0, 255, 255),  # Желтый - печатные буквы
        'cursive_text': (128, 128, 128)  # Серый - прописной текст
    }

    # Английские labels
    labels = {
        'student_name': "NAME",
        'test_title': "TITLE",
        'tasks': "TASKS",
        'short_answers': "ANSWERS",
        'printed_text': "PRINTED",
        'cursive_text': "CURSIVE"
    }

    # Рисуем ВСЕ найденные регионы серым цветом с номерами
    for i, region in enumerate(regions):
        x, y, w, h = region['bbox']
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (100, 100, 100), 3)
        cv2.putText(result_image, f"#{i + 1}", (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # Рисуем только 6 классифицированных регионов цветами
    for region_type, region in classified_regions.items():
        if region:
            x, y, w, h = region['bbox']
            color = colors[region_type]
            label = labels[region_type]

            # Рисуем толстую цветную рамку
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 6)

            # Добавляем подпись
            cv2.putText(result_image, label, (x + 10, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    # Добавляем общую информацию
    info_text = f"Regions found: {len(regions)} | Classified: {len(classified_regions)}/6"
    cv2.putText(result_image, info_text, (20, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)

    return result_image


def analyze_text_regions(processed_image, debug=True, output_path=None, original_filename=None):
    """Анализирует и визуализирует текстовые регионы"""

    print("\n=== TEXT REGIONS DETECTION ===")
    print("Looking for 6 bordered regions (excluding top header)...")

    # Используем только эффективный метод
    regions = find_text_boxes_direct(processed_image)

    print(f"Total bordered regions found: {len(regions)}")

    # Если нашли больше 6 регионов, берем 6 самых больших
    if len(regions) > 6:
        print(f"Found {len(regions)} regions, taking 6 largest...")
        regions.sort(key=lambda r: r['area'], reverse=True)
        regions = regions[:6]
        # Снова сортируем по вертикали для классификации
        regions.sort(key=lambda r: r['center_y'])

    # Классифицируем регионы (только первые 6)
    classified = classify_regions(regions, processed_image.shape)

    # Выводим подробную информацию
    print("\n=== REGIONS DETAILS ===")
    for i, region in enumerate(regions[:6]):
        x, y, w, h = region['bbox']
        area_percent = (region['area'] / (processed_image.shape[0] * processed_image.shape[1])) * 100
        print(f"{i + 1}. Position: ({x}, {y}), Size: {w}x{h}, Area: {region['area']} ({area_percent:.1f}%)")

    # Создаем визуализацию
    visualization = draw_regions(processed_image, regions, classified)

    # Сохраняем найденные области в отдельные папки
    if output_path and original_filename:
        save_detected_regions(processed_image, classified, output_path, original_filename)

    # Сохраняем изображение с разметкой если debug=True
    if debug and output_path and original_filename and regions:
        regions_image_path = output_path / f"{original_filename}_regions.jpg"
        save_image(visualization, regions_image_path)
        print(f"Image with regions saved: {regions_image_path}")

    if debug and regions:
        from utils import resize_for_display
        display_img = resize_for_display(visualization, max_width=1000, max_height=800)
        cv2.imshow("Detected Text Regions (6 bordered areas)", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif debug:
        print("No regions to display!")

    return visualization, regions, classified


def save_detected_regions(image, classified_regions, output_path, original_filename):
    """Сохраняет найденные области в отдельные папки"""

    # Создаем структуру папок
    folders = {
        'student_name': 'name',
        'test_title': 'test_text',
        'tasks': 'test_task',
        'short_answers': 'text_answer',
        'printed_text': 'printed_text',
        'cursive_text': 'cursive_text'
    }

    # Создаем все папки
    for folder_name in folders.values():
        folder_path = output_path / folder_name
        folder_path.mkdir(exist_ok=True)

    # Сохраняем каждую область
    saved_count = 0
    for region_type, region in classified_regions.items():
        if region and region_type in folders:
            x, y, w, h = region['bbox']

            # Вырезаем область из изображения
            region_image = image[y:y + h, x:x + w]

            # Определяем путь для сохранения
            folder_name = folders[region_type]
            filename = f"{original_filename}_{region_type}.jpg"
            save_path = output_path / folder_name / filename

            # Сохраняем изображение
            save_image(region_image, save_path)
            print(f"Saved {region_type}: {save_path}")
            saved_count += 1

    print(f"Successfully saved {saved_count} regions to separate folders")


def save_image(image, path):
    """Сохраняет изображение по указанному пути"""
    cv2.imwrite(str(path), image)