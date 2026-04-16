import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import logging
import glob
import os
import sys

# Импортируем необходимые модули
from utils import load_image, save_image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ---------------- НАСТРОЙКИ ----------------
@dataclass
class Config:
    CELL_MIN_W: int = 25
    CELL_MAX_W: int = 80
    CELL_MIN_H: int = 30
    CELL_MAX_H: int = 90
    ROW_THRESHOLD: int = 20
    AREA_TOLERANCE: float = 0.15
    DEBUG_SCALE: float = 0.7
    DEBUG: bool = False  # Только для отображения на экране, сохранение всегда
    
    # Базовые пути
    INPUT_BASE: str = "out"
    OUTPUT_CELLS_BASE: str = "cells"
    OUTPUT_DEBUG_BASE: str = "cells_debug"
    
    # Паттерны для поиска файлов по подпапкам
    FILE_PATTERNS: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        "name": ("name", "*_student_name.jpg"),
        "test_text": ("test_text", "*_test_title.jpg"),
        "text_answers": ("text_answer", "*_short_answers.jpg"),
        "printed_text": ("printed_text", "*_printed_text.jpg")
    })
    
    # Параметры кадрирования
    TEST_TEXT_CROP: Tuple[int, int] = (205, 350)
    TEXT_ANSWERS_CROP: Tuple[int, int] = (80, 165)


@dataclass
class Cell:
    bbox: Tuple[int, int, int, int]
    center_x: float
    center_y: float
    area: int
    
    @classmethod
    def from_contour(cls, contour: np.ndarray, config: Config) -> Optional['Cell']:
        x, y, w, h = cv2.boundingRect(contour)
        
        if (config.CELL_MIN_W < w < config.CELL_MAX_W and
            config.CELL_MIN_H < h < config.CELL_MAX_H and
            0.4 < w / h < 1.4):
            return cls(
                bbox=(x, y, w, h),
                center_x=x + w / 2,
                center_y=y + h / 2,
                area=w * h
            )
        return None


# ---------------- БАЗОВЫЕ ФУНКЦИИ ----------------
class ImageProcessor:
    def __init__(self, config: Config, debug_prefix: str = "debug"):
        self.config = config
        self.debug_prefix = debug_prefix
        self.debug_mode = config.DEBUG
        
    def find_cells(self, image: np.ndarray) -> List[Cell]:
        """Находит прямоугольные клетки"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 5
        )
        
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for contour in contours:
            if cell := Cell.from_contour(contour, self.config):
                cells.append(cell)
        
        logger.info(f"Detected cells: {len(cells)}")
        return cells


def draw_cells(image: np.ndarray, cells: List[Cell], config: Config,
               window_name: str = "Cells", draw_numbers: bool = True) -> np.ndarray:
    """Рисует найденные клетки и сохраняет в debug"""
    debug_img = image.copy()
    
    for i, cell in enumerate(cells):
        x, y, w, h = cell.bbox
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if draw_numbers:
            cv2.putText(debug_img, str(i+1), (x+2, y+18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    
    # Масштабирование для отображения (только если DEBUG=True)
    if config.DEBUG:
        h, w = debug_img.shape[:2]
        display_img = cv2.resize(debug_img, (int(w*config.DEBUG_SCALE), int(h*config.DEBUG_SCALE)))
        cv2.imshow(window_name, display_img)
        cv2.waitKey(1000)
    
    return debug_img


def save_cells(image: np.ndarray, cells: List[Cell], 
               output_folder: Path, prefix: str = "cell", margin: int = 2):
    """Сохраняет отдельные клетки"""
    output_folder.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for i, cell in enumerate(cells):
        x, y, w, h = cell.bbox
        # Проверяем, что координаты в пределах изображения
        y1, y2 = max(0, y+margin), min(image.shape[0], y+h-margin)
        x1, x2 = max(0, x+margin), min(image.shape[1], x+w-margin)
        
        if y2 > y1 and x2 > x1:
            cell_img = image[y1:y2, x1:x2]
            filename = f"{prefix}_{i+1:03d}.jpg"
            cv2.imwrite(str(output_folder / filename), cell_img)
            saved_count += 1
    
    logger.info(f"Saved {saved_count} cells to {output_folder}")


def save_debug_cells(image: np.ndarray, cells: List[Cell], path: Path):
    """Сохраняет отладочное изображение с размеченными клетками (всегда)"""
    debug_img = image.copy()
    
    for i, cell in enumerate(cells):
        x, y, w, h = cell.bbox
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(debug_img, str(i + 1), (x + 3, y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imwrite(str(path), debug_img)


# ---------------- ФУНКЦИИ СОРТИРОВКИ ----------------
def group_cells_into_rows(cells: List[Cell], expected_rows: int) -> List[List[Cell]]:
    """Группирует клетки в заданное число строк"""
    if not cells:
        return []
    
    cells.sort(key=lambda c: c.center_y)
    
    min_y = cells[0].center_y
    max_y = cells[-1].center_y
    row_height = (max_y - min_y) / expected_rows if expected_rows > 0 else 1
    
    rows = [[] for _ in range(expected_rows)]
    for cell in cells:
        index = min(int((cell.center_y - min_y) / row_height), expected_rows - 1)
        rows[index].append(cell)
    
    for row in rows:
        row.sort(key=lambda c: c.center_x)
    
    return rows


def filter_nested_cells(cells: List[Cell]) -> List[Cell]:
    """Удаляет вложенные клетки"""
    filtered = []
    for i, c1 in enumerate(cells):
        x1, y1, w1, h1 = c1.bbox
        inside = False
        
        for j, c2 in enumerate(cells):
            if i == j:
                continue
            
            x2, y2, w2, h2 = c2.bbox
            if (x1 >= x2 and y1 >= y2 and 
                x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2):
                inside = True
                break
        
        if not inside:
            filtered.append(c1)
    
    logger.info(f"After removing nested: {len(filtered)}")
    return filtered


def filter_by_area(cells: List[Cell], tolerance: float = 0.15) -> List[Cell]:
    """Фильтрует клетки по площади относительно медианы"""
    if not cells:
        return []
    
    areas = sorted([c.area for c in cells])
    median_area = areas[len(areas) // 2]
    
    filtered = [c for c in cells if abs(c.area - median_area) < median_area * tolerance]
    logger.info(f"After area filter: {len(filtered)}")
    return filtered


# ---------------- ФУНКЦИИ ПОИСКА ФАЙЛОВ ----------------
def find_files_by_pattern(base_path: str, subfolder: str, pattern: str) -> List[Path]:
    """Находит все файлы по паттерну в указанной подпапке"""
    search_path = Path(base_path) / subfolder / pattern
    return sorted(Path(p) for p in glob.glob(str(search_path)))


def extract_prefix(filename: Path, suffix: str) -> str:
    """Извлекает префикс из имени файла (часть до суффикса)"""
    name = filename.stem
    return name.replace(suffix, "").rstrip("_")


# ---------------- СПЕЦИАЛИЗИРОВАННЫЕ ОБРАБОТЧИКИ ----------------
def process_name_files(files: List[Path], config: Config):
    """Обработка файлов с именами"""
    logger.info(f"\n=== PROCESSING NAME FILES: {len(files)} files ===")
    
    for file_path in files:
        logger.info(f"\n--- Processing: {file_path.name} ---")
        
        image = cv2.imread(str(file_path))
        if image is None:
            logger.error(f"Failed to load: {file_path}")
            continue
        
        # Извлекаем префикс для создания уникальных папок
        prefix = extract_prefix(file_path, "_student_name")
        
        processor = ImageProcessor(config, f"debug_name_{prefix}")
        cells = processor.find_cells(image)
        
        # Сортировка по строкам
        rows = group_cells_into_rows(cells, 2)
        
        if len(rows) >= 2:
            # Берем по 35 крупнейших из каждой строки
            top_row = sorted(rows[0], key=lambda c: c.area, reverse=True)[:35]
            bottom_row = sorted(rows[1], key=lambda c: c.area, reverse=True)[:35]
            
            top_row.sort(key=lambda c: c.center_x)
            bottom_row.sort(key=lambda c: c.center_x)
            
            cells = top_row + bottom_row
        
        # Рисуем (и показываем только если DEBUG=True)
        draw_cells(image, cells, config, f"NAME CELLS - {prefix}")
        
        # Сохраняем отладочное изображение (всегда)
        debug_path = Path(config.OUTPUT_DEBUG_BASE) / f"{prefix}_debug" / "name_cells.jpg"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        save_debug_cells(image, cells, debug_path)
        
        # Сохраняем клетки
        cells_path = Path(config.OUTPUT_CELLS_BASE) / f"{prefix}_cells" / "name_cells"
        save_cells(image, cells, cells_path, f"name_{prefix}")


def process_test_text_files(files: List[Path], config: Config):
    """Обработка файлов с тестовым текстом"""
    logger.info(f"\n=== PROCESSING TEST TEXT FILES: {len(files)} files ===")
    
    for file_path in files:
        logger.info(f"\n--- Processing: {file_path.name} ---")
        
        image = cv2.imread(str(file_path))
        if image is None:
            logger.error(f"Failed to load: {file_path}")
            continue
        
        # Извлекаем префикс
        prefix = extract_prefix(file_path, "_test_title")
        
        # Кадрирование
        crop_y, crop_x = config.TEST_TEXT_CROP
        if image.shape[0] > crop_y and image.shape[1] > crop_x:
            image = image[crop_y:, crop_x:]
        
        processor = ImageProcessor(config, f"debug_test_text_{prefix}")
        cells = processor.find_cells(image)
        
        if not cells:
            logger.warning("No cells found")
            continue
        
        # Фильтрация
        cells = filter_nested_cells(cells)
        cells = filter_by_area(cells, config.AREA_TOLERANCE)
        
        # Группировка в 2 строки
        rows = group_cells_into_rows(cells, 2)
        
        # Коррекция и заполнение пропусков
        ordered = []
        for row in rows:
            corrected = correct_row_sequence(row, target_count=29)
            ordered.extend(corrected)
        
        logger.info(f"FINAL CELL COUNT: {len(ordered)}")
        
        draw_cells(image, ordered, config, f"TEST TEXT CELLS - {prefix}")
        
        # Сохраняем отладочное изображение (всегда)
        debug_path = Path(config.OUTPUT_DEBUG_BASE) / f"{prefix}_debug" / "test_text_cells.jpg"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        save_debug_cells(image, ordered, debug_path)
        
        # Сохраняем клетки
        cells_path = Path(config.OUTPUT_CELLS_BASE) / f"{prefix}_cells" / "test_text_cells"
        save_cells(image, ordered, cells_path, f"test_text_{prefix}")


def correct_row_sequence(row: List[Cell], target_count: int) -> List[Cell]:
    """Корректирует последовательность клеток в строке"""
    if len(row) < 2:
        return row
    
    row = sorted(row, key=lambda c: c.center_x)
    
    # Вычисляем средние параметры
    avg_w = int(sum(c.bbox[2] for c in row) / len(row))
    avg_h = int(sum(c.bbox[3] for c in row) / len(row))
    avg_step = calculate_avg_step(row)
    
    if avg_step == 0:
        return row
    
    corrected = [row[0]]
    
    for i in range(len(row) - 1):
        current = row[i]
        next_cell = row[i + 1]
        corrected.append(next_cell)
        
        dx = next_cell.center_x - current.center_x
        expected = max(1, int(round(dx / avg_step)))
        
        # Вставляем недостающие клетки
        for k in range(1, expected):
            new_x = int(current.bbox[0] + avg_step * k)
            new_bbox = (new_x, current.bbox[1], avg_w, avg_h)
            
            corrected.insert(-1, Cell(
                bbox=new_bbox,
                center_x=new_x + avg_w / 2,
                center_y=current.center_y,
                area=avg_w * avg_h
            ))
    
    # Дополняем до target_count
    while len(corrected) < target_count and avg_step > 0:
        last = corrected[-1]
        new_x = int(last.bbox[0] + avg_step)
        new_bbox = (new_x, last.bbox[1], avg_w, avg_h)
        
        corrected.append(Cell(
            bbox=new_bbox,
            center_x=new_x + avg_w / 2,
            center_y=last.center_y,
            area=avg_w * avg_h
        ))
    
    return corrected[:target_count]


def calculate_avg_step(row: List[Cell]) -> float:
    """Вычисляет средний шаг между клетками"""
    if len(row) < 2:
        return 0
    steps = [row[i+1].center_x - row[i].center_x for i in range(len(row)-1)]
    return sum(steps) / len(steps) if steps else 0


def process_text_answers_files(files: List[Path], config: Config):
    """Обработка файлов с текстовыми ответами"""
    logger.info(f"\n=== PROCESSING TEXT ANSWERS FILES: {len(files)} files ===")
    
    for file_path in files:
        logger.info(f"\n--- Processing: {file_path.name} ---")
        
        image = cv2.imread(str(file_path))
        if image is None:
            logger.error(f"Failed to load: {file_path}")
            continue
        
        # Извлекаем префикс
        prefix = extract_prefix(file_path, "_short_answers")
        
        # Кадрирование
        crop_y, crop_x = config.TEXT_ANSWERS_CROP
        if image.shape[0] > crop_y and image.shape[1] > crop_x:
            image = image[crop_y:, crop_x:]
        
        processor = ImageProcessor(config, f"debug_text_answers_{prefix}")
        cells = processor.find_cells(image)
        
        cells = filter_nested_cells(cells)
        rows = group_cells_into_rows(cells, 10)
        
        # Создаем папки для сохранения
        output_base = Path(config.OUTPUT_CELLS_BASE) / f"{prefix}_cells" / "text_answer_cells"
        debug_base = Path(config.OUTPUT_DEBUG_BASE) / f"{prefix}_debug" / "text_answer_rows"
        
        for i, row in enumerate(rows):
            row = sorted(row, key=lambda c: c.center_x)[:21]
            
            draw_cells(image, row, config, f"ANSWERS ROW {i+1} - {prefix}")
            
            # Сохраняем отладочное изображение строки (всегда)
            debug_path = debug_base / f"row_{i+1}.jpg"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            save_debug_cells(image, row, debug_path)
            
            # Сохраняем клетки
            row_folder = output_base / f"row_{i+1}"
            save_cells(image, row, row_folder, f"row{i+1}_{prefix}")


def process_printed_text_files(files: List[Path], config: Config):
    """Обработка файлов с печатным текстом"""
    logger.info(f"\n=== PROCESSING PRINTED TEXT FILES: {len(files)} files ===")
    
    for file_path in files:
        logger.info(f"\n--- Processing: {file_path.name} ---")
        
        image = cv2.imread(str(file_path))
        if image is None:
            logger.error(f"Failed to load: {file_path}")
            continue
        
        # Извлекаем префикс
        prefix = extract_prefix(file_path, "_printed_text")
        
        processor = ImageProcessor(config, f"debug_printed_text_{prefix}")
        cells = processor.find_cells(image)
        
        cells = filter_nested_cells(cells)
        cells = sorted(cells, key=lambda c: c.area, reverse=True)[:344]
        
        # Восстановление пропущенных клеток
        if cells:
            rows = group_cells_into_rows(cells, max(1, len(cells) // 30))
            cells = reconstruct_missing_cells(rows)
        
        draw_cells(image, cells, config, f"PRINTED TEXT CELLS - {prefix}")
        
        # Сохраняем отладочное изображение (всегда)
        debug_path = Path(config.OUTPUT_DEBUG_BASE) / f"{prefix}_debug" / "printed_text_cells.jpg"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        save_debug_cells(image, cells, debug_path)
        
        # Сохраняем клетки
        cells_path = Path(config.OUTPUT_CELLS_BASE) / f"{prefix}_cells" / "printed_text_cells"
        save_cells(image, cells, cells_path, f"printed_{prefix}")


def reconstruct_missing_cells(rows: List[List[Cell]]) -> List[Cell]:
    """Восстанавливает пропущенные клетки в строках"""
    reconstructed = []
    
    for row_index, row in enumerate(rows):
        if len(row) < 2:
            reconstructed.extend(row)
            continue
        
        row = sorted(row, key=lambda c: c.center_x)
        
        avg_w = int(sum(c.bbox[2] for c in row) / len(row))
        avg_h = int(sum(c.bbox[3] for c in row) / len(row))
        avg_y = int(sum(c.center_y for c in row) / len(row))
        avg_step = calculate_avg_step(row)
        
        if avg_step == 0:
            reconstructed.extend(row)
            continue
        
        new_row = [row[0]]
        
        for i in range(len(row) - 1):
            c1, c2 = row[i], row[i+1]
            new_row.append(c2)
            
            dx = c2.center_x - c1.center_x
            
            while dx > avg_step * 1.5:
                logger.info(f"Reconstructing missing cell in row {row_index+1}")
                new_center_x = c1.center_x + avg_step
                x = int(new_center_x - avg_w / 2)
                y = int(avg_y - avg_h / 2)
                
                fake_cell = Cell(
                    bbox=(x, y, avg_w, avg_h),
                    center_x=new_center_x,
                    center_y=avg_y,
                    area=avg_w * avg_h
                )
                
                new_row.insert(-1, fake_cell)
                c1 = fake_cell
                dx = c2.center_x - c1.center_x
        
        reconstructed.extend(new_row)
    
    return reconstructed


# ---------------- MAIN ----------------
def main():
    """Главная функция - для вызова из основного скрипта"""
    try:
        # Создаем экземпляр конфигурации
        config = Config()
        
        # Создаем базовые папки
        Path(config.OUTPUT_CELLS_BASE).mkdir(parents=True, exist_ok=True)
        Path(config.OUTPUT_DEBUG_BASE).mkdir(parents=True, exist_ok=True)
        
        # Обрабатываем каждую категорию файлов
        for category, (subfolder, pattern) in config.FILE_PATTERNS.items():
            files = find_files_by_pattern(config.INPUT_BASE, subfolder, pattern)
            
            if not files:
                logger.warning(f"No files found in {subfolder} for pattern: {pattern}")
                continue
            
            # Вызываем соответствующую функцию обработки
            if category == "name":
                process_name_files(files, config)
            elif category == "test_text":
                process_test_text_files(files, config)
            elif category == "text_answers":
                process_text_answers_files(files, config)
            elif category == "printed_text":
                process_printed_text_files(files, config)
        
        logger.info("\n✅ Разбиение на клетки завершено")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()