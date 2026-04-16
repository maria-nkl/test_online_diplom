import cv2
import numpy as np
from pathlib import Path


def find_dark_squares(image, area_threshold=1000):
    """Находит темные квадраты на изображении"""
    # Конвертируем в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применяем бинаризацию для выделения темных областей
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Морфологические операции для улучшения качества
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []

    for contour in contours:
        # Фильтруем по площади
        area = cv2.contourArea(contour)
        if area < area_threshold:
            continue

        # Аппроксимируем контур
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Проверяем, что это четырехугольник (квадрат/прямоугольник)
        if len(approx) == 4:
            # Вычисляем соотношение сторон
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Проверяем, что это примерно квадрат
            if 0.7 <= aspect_ratio <= 1.3:
                squares.append({
                    'contour': approx,
                    'bbox': (x, y, w, h),
                    'center': (x + w // 2, y + h // 2),
                    'area': area
                })

    return squares


def analyze_squares_layout(squares, image_shape):
    """Анализирует расположение квадратов для определения ориентации"""
    height, width = image_shape[:2]

    if len(squares) < 5:
        print(f"Предупреждение: найдено только {len(squares)} квадратов")
        return None, None

    # Группируем квадраты по линиям
    tolerance = height * 0.05  # допуск для определения линии

    # Группируем по горизонтальным линиям
    horizontal_lines = {}
    for square in squares:
        y = square['center'][1]
        found_line = False
        for line_y in horizontal_lines.keys():
            if abs(y - line_y) < tolerance:
                horizontal_lines[line_y].append(square)
                found_line = True
                break
        if not found_line:
            horizontal_lines[y] = [square]

    # Группируем по вертикальным линиям
    vertical_lines = {}
    for square in squares:
        x = square['center'][0]
        found_line = False
        for line_x in vertical_lines.keys():
            if abs(x - line_x) < tolerance:
                vertical_lines[line_x].append(square)
                found_line = True
                break
        if not found_line:
            vertical_lines[x] = [square]

    # Ищем линии с тремя квадратами (это будет нижняя или верхняя сторона)
    three_square_line = None
    line_type = None  # 'horizontal' или 'vertical'
    line_key = None

    for y, line_squares in horizontal_lines.items():
        if len(line_squares) == 3:
            three_square_line = line_squares
            line_type = 'horizontal'
            line_key = y
            break

    for x, line_squares in vertical_lines.items():
        if len(line_squares) == 3:
            three_square_line = line_squares
            line_type = 'vertical'
            line_key = x
            break

    if not three_square_line:
        print("Не найдено линии с тремя квадратами")
        return None, None

    print(f"Найдена линия с тремя квадратами: тип {line_type}")

    # Создаем список ID квадратов в линии с тремя квадратами для сравнения
    three_square_centers = [tuple(s['center']) for s in three_square_line]

    # Определяем угловые квадраты
    corners = {}

    if line_type == 'horizontal':
        # Горизонтальная линия с тремя квадратами - это либо верх, либо низ
        line_y = three_square_line[0]['center'][1]

        # Сортируем три квадрата по X координате
        three_sorted = sorted(three_square_line, key=lambda s: s['center'][0])

        # Определяем это верх или низ
        if line_y < height / 2:
            # Линия в верхней части - значит это верхние квадраты
            print("Три квадрата сверху - изображение перевернуто на 180°")

            # Находим оставшиеся 2 квадрата - это будут нижние угловые
            remaining_squares = [s for s in squares if tuple(s['center']) not in three_square_centers]
            if len(remaining_squares) >= 2:
                remaining_sorted = sorted(remaining_squares, key=lambda s: s['center'][0])
                corners['bottom_left'] = remaining_sorted[0]
                corners['bottom_right'] = remaining_sorted[-1]

            # Три верхних квадрата
            corners['top_left'] = three_sorted[0]
            corners['top_center'] = three_sorted[1]  # центральный верхний
            corners['top_right'] = three_sorted[2]

        else:
            # Линия в нижней части - значит это нижние квадраты (правильная ориентация)
            print("Три квадрата снизу - правильная ориентация")

            # Находим оставшиеся 2 квадрата - это будут верхние угловые
            remaining_squares = [s for s in squares if tuple(s['center']) not in three_square_centers]
            if len(remaining_squares) >= 2:
                remaining_sorted = sorted(remaining_squares, key=lambda s: s['center'][0])
                corners['top_left'] = remaining_sorted[0]
                corners['top_right'] = remaining_sorted[-1]

            # Три нижних квадрата
            corners['bottom_left'] = three_sorted[0]
            corners['bottom_center'] = three_sorted[1]  # центральный нижний
            corners['bottom_right'] = three_sorted[2]

    else:  # vertical
        # Вертикальная линия с тремя квадратами - это либо лево, либо право
        line_x = three_square_line[0]['center'][0]

        # Сортируем три квадрата по Y координате
        three_sorted = sorted(three_square_line, key=lambda s: s['center'][1])

        # Определяем это лево или право
        if line_x < width / 2:
            # Линия в левой части - изображение повернуто на 90° против часовой
            print("Три квадрата слева - изображение повернуто на 90° против часовой")

            # Находим оставшиеся 2 квадрата - это будут правые угловые
            remaining_squares = [s for s in squares if tuple(s['center']) not in three_square_centers]
            if len(remaining_squares) >= 2:
                remaining_sorted = sorted(remaining_squares, key=lambda s: s['center'][1])
                # После поворота эти станут нижними угловыми
                corners['bottom_left'] = remaining_sorted[0]  # станет bottom_left
                corners['bottom_right'] = remaining_sorted[-1]  # станет top_left

            # Три левых квадрата
            # После поворота эти станут верхними и центральным
            corners['top_left'] = three_sorted[0]  # станет top_right
            corners['center_left'] = three_sorted[1]  # центральный левый
            corners['bottom_left_extra'] = three_sorted[2]  # станет bottom_right

        else:
            # Линия в правой части - изображение повернуто на 90° по часовой
            print("Три квадрата справа - изображение повернуто на 90° по часовой")

            # Находим оставшиеся 2 квадрата - это будут левые угловые
            remaining_squares = [s for s in squares if tuple(s['center']) not in three_square_centers]
            if len(remaining_squares) >= 2:
                remaining_sorted = sorted(remaining_squares, key=lambda s: s['center'][1])
                # После поворота эти станут нижними угловые
                corners['bottom_left'] = remaining_sorted[0]  # станет bottom_left
                corners['bottom_right'] = remaining_sorted[-1]  # станет top_left

            # Три правых квадрата
            # После поворота эти станут верхними и центральным
            corners['top_right'] = three_sorted[0]  # станет bottom_right
            corners['center_right'] = three_sorted[1]  # центральный правый
            corners['bottom_right_extra'] = three_sorted[2]  # станет top_right

    return corners, line_type


def correct_orientation(image, corners, line_type):
    """Корректирует ориентацию изображения на основе анализа квадратов"""
    rotation = 0

    if line_type == 'horizontal':
        # Проверяем, где находятся три квадрата
        if any(k.startswith('top_') for k in corners.keys() if 'center' in k):
            # Три квадрата сверху - переворачиваем на 180°
            print("Поворот на 180°")
            image = cv2.rotate(image, cv2.ROTATE_180)
            rotation = 180
        else:
            # Три квадрата снизу - правильная ориентация
            print("Правильная ориентация")
            rotation = 0

    else:  # vertical
        if any(k.startswith('left') for k in corners.keys() if 'center' in k):
            # Три квадрата слева - поворот на 90° против часовой
            print("Поворот на 90° против часовой")
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotation = 90
        else:
            # Три квадрата справа - поворот на 90° по часовой
            print("Поворот на 90° по часовой")
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rotation = -90

    return image, rotation


def align_image(image, corners):
    """ТОЛЬКО выравнивает изображение с перспективой БЕЗ обрезки по бокам"""
    # Находим четыре угловых квадрата
    corner_points = []
    corner_positions = ['top_left', 'top_right', 'bottom_right', 'bottom_left']

    for position in corner_positions:
        if position in corners:
            corner_points.append(corners[position]['contour'][0][0])

    if len(corner_points) != 4:
        print(f"Не удалось найти все 4 угловых квадрата (найдено {len(corner_points)})")
        return image

    # Определяем исходные точки для перспективного преобразования
    src_points = np.array(corner_points, dtype=np.float32)

    # Вместо bounding box используем фиксированные пропорции
    # Сохраняем оригинальное соотношение сторон
    original_height, original_width = image.shape[:2]

    # Вычисляем приблизительную высоту области документа
    y_coords = src_points[:, 1]
    doc_height = np.max(y_coords) - np.min(y_coords)

    # Сохраняем оригинальную ширину, но корректируем высоту
    target_width = original_width
    target_height = int(doc_height * 1.1)  # +10% чтобы точно все поместилось

    # Смещаем точки чтобы документ был в центре по горизонтали
    x_offset = (original_width - (np.max(src_points[:, 0]) - np.min(src_points[:, 0]))) // 2

    dst_points = np.array([
        [x_offset, 0],
        [original_width - x_offset, 0],
        [original_width - x_offset, target_height],
        [x_offset, target_height]
    ], dtype=np.float32)

    # Применяем перспективное преобразование
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    aligned = cv2.warpPerspective(image, matrix, (target_width, target_height))

    print(f"Выравнивание выполнено:")
    print(f"  Оригинальный размер: {image.shape}")
    print(f"  Размер после выравнивания: {aligned.shape}")
    print(f"  Сохранена оригинальная ширина: {target_width}")

    return aligned


def draw_original_squares(image, squares):
    """Рисует все найденные квадраты на оригинальном изображении"""
    debug_image = image.copy()

    # Рисуем все найденные квадраты
    for i, square in enumerate(squares):
        # Рисуем контур зеленым
        cv2.drawContours(debug_image, [square['contour']], -1, (0, 255, 0), 3)

        # Рисуем bounding box синим
        x, y, w, h = square['bbox']
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Рисуем центр красной точкой
        center_x, center_y = square['center']
        cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)

        # Добавляем номер квадрата
        cv2.putText(debug_image, str(i + 1), (center_x - 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Добавляем информацию о количестве квадратов
    cv2.putText(debug_image, f"Found {len(squares)} squares", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return debug_image


def draw_debug_info(image, squares, corners, line_type):
    """Рисует отладочную информацию на изображении с анализом позиций"""
    debug_image = image.copy()

    # Рисуем все найденные квадраты
    for square in squares:
        cv2.drawContours(debug_image, [square['contour']], -1, (0, 255, 0), 3)
        x, y, w, h = square['bbox']
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Рисуем угловые квадраты разными цветами
    colors = {
        'top_left': (0, 0, 255),  # красный
        'top_right': (255, 0, 0),  # синий
        'bottom_left': (0, 255, 0),  # зеленый
        'bottom_right': (255, 255, 0),  # голубой
        'top_center': (255, 0, 255),  # фиолетовый
        'bottom_center': (255, 0, 255),  # фиолетовый
        'center_left': (255, 0, 255),  # фиолетовый
        'center_right': (255, 0, 255),  # фиолетовый
    }

    for position, square in corners.items():
        color = colors.get(position, (255, 255, 255))
        cv2.drawContours(debug_image, [square['contour']], -1, color, 4)
        cv2.putText(debug_image, position, square['center'],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Рисуем информацию о типе линии
    cv2.putText(debug_image, f"Line type: {line_type}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Добавляем информацию о количестве квадратов
    cv2.putText(debug_image, f"Total squares: {len(squares)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return debug_image


def process_image(image, debug=False):
    """Основная функция обработки изображения - поворот, выравнивание горизонта и вертикальная обрезка"""
    original_image = image.copy()

    # Находим темные квадраты
    squares = find_dark_squares(image)
    print(f"Найдено квадратов: {len(squares)}")

    if len(squares) < 5:
        print(f"Ошибка: недостаточно квадратов для обработки")
        return None, None, None

    # Анализируем расположение квадратов
    corners, line_type = analyze_squares_layout(squares, image.shape)

    if not corners:
        print("Не удалось определить расположение квадратов")
        return None, None, None

    print(f"Определена ориентация: {line_type}")

    # Корректируем ориентацию
    image, rotation = correct_orientation(image, corners, line_type)

    # После поворота заново анализируем квадраты для корректного определения позиций
    if rotation != 0:
        print("Переанализируем квадраты после поворота...")
        squares = find_dark_squares(image)
        corners, line_type = analyze_squares_layout(squares, image.shape)

    # Выравниваем горизонт по нижним квадратам
    image, horizon_angle = align_horizon(image, corners)

    # После выравнивания горизонта снова находим квадраты
    if abs(horizon_angle) > 0.5:
        print("Переанализируем квадраты после выравнивания горизонта...")
        squares = find_dark_squares(image)
        corners, line_type = analyze_squares_layout(squares, image.shape)

    # Обрезаем по вертикали используя границы квадратов
    processed_image = crop_vertical_by_squares(image, corners)

    # Подготавливаем отладочные изображения если нужно
    debug_images = {}
    if debug:
        debug_images['original_with_squares'] = draw_original_squares(original_image, squares)
        debug_images['debug'] = draw_debug_info(image, squares, corners, line_type)
        # Сохраняем изображение после поворота и выравнивания горизонта (до обрезки)
        debug_images['aligned_before_crop'] = image.copy()

    return processed_image, debug_images, squares


def crop_vertical_by_squares(image, corners):
    """Обрезает изображение по вертикали: сверху по нижней границе верхних квадратов, снизу по верхней границе нижних квадратов"""

    height, width = image.shape[:2]

    # Находим верхние и нижние квадраты
    top_squares = []
    bottom_squares = []

    for position, square in corners.items():
        if 'top' in position:
            top_squares.append(square)
        elif 'bottom' in position:
            bottom_squares.append(square)

    print(f"Найдено верхних квадратов: {len(top_squares)}, нижних: {len(bottom_squares)}")

    # Если не нашли квадраты, возвращаем оригинальное изображение
    if not top_squares and not bottom_squares:
        print("Не найдены квадраты для обрезки")
        return image

    # Определяем границы обрезки
    top_crop = 0
    bottom_crop = height

    # Для верхних квадратов - берем самую нижнюю точку (Y + height)
    if top_squares:
        top_crop = max(square['bbox'][1] + square['bbox'][3] for square in top_squares)
        print(f"Самая нижняя точка верхних квадратов: {top_crop}")

    # Для нижних квадратов - берем самую верхнюю точку (Y)
    if bottom_squares:
        bottom_crop = min(square['bbox'][1] for square in bottom_squares)
        print(f"Самая верхняя точка нижних квадратов: {bottom_crop}")

    # Добавляем отступы
    top_padding = 20
    bottom_padding = 00

    top_crop = min(height, top_crop + top_padding)
    bottom_crop = max(0, bottom_crop - bottom_padding)

    # Проверяем корректность координат
    if top_crop >= bottom_crop:
        print(f"Некорректные координаты обрезки: верх {top_crop} >= низ {bottom_crop}")
        return image

    # Проверяем, что обрезка имеет смысл (минимум 50% высоты остается)
    if (bottom_crop - top_crop) < height * 0.5:
        print(f"Обрезка слишком агрессивная, оставшаяся высота: {bottom_crop - top_crop}")
        return image

    # Обрезаем по вертикали (горизонталь остается без изменений)
    cropped_image = image[top_crop:bottom_crop, 0:width]

    print(f"Вертикальная обрезка выполнена:")
    print(f"  Область обрезки: Y={top_crop} до Y={bottom_crop}")
    print(f"  Размер до обрезки: {image.shape}")
    print(f"  Размер после обрезки: {cropped_image.shape}")

    return cropped_image


def align_horizon(image, corners):
    """Выравнивает горизонт по нижним квадратам (небольшой поворот)"""
    # Находим нижние квадраты
    bottom_squares = []
    for position, square in corners.items():
        if 'bottom' in position:
            bottom_squares.append(square)

    if len(bottom_squares) < 2:
        print("Недостаточно нижних квадратов для выравнивания горизонта")
        return image, 0

    # Сортируем нижние квадраты по X координате
    bottom_squares_sorted = sorted(bottom_squares, key=lambda s: s['center'][0])

    # Берем левый и правый нижние квадраты
    left_bottom = bottom_squares_sorted[0]
    right_bottom = bottom_squares_sorted[-1]

    # Вычисляем угол наклона между ними
    y1 = left_bottom['center'][1]
    y2 = right_bottom['center'][1]
    x1 = left_bottom['center'][0]
    x2 = right_bottom['center'][0]

    # Вычисляем угол в радианах и градусах
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)

    print(f"Угол наклона горизонта: {angle_deg:.2f}°")

    # Если угол очень маленький, не поворачиваем
    if abs(angle_deg) < 0.5:
        print("Угол слишком мал, выравнивание не требуется")
        return image, 0

    # Поворачиваем изображение
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Создаем матрицу поворота
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Вычисляем новые размеры чтобы не обрезать углы
    cos_val = np.abs(rotation_matrix[0, 0])
    sin_val = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))

    # Корректируем матрицу поворота для центра
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Применяем поворот
    aligned = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))

    print(f"Выравнивание горизонта выполнено:")
    print(f"  Угол поворота: {angle_deg:.2f}°")
    print(f"  Размер до: {image.shape}")
    print(f"  Размер после: {aligned.shape}")

    return aligned, angle_deg