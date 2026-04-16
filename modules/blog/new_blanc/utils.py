import cv2
import numpy as np
from pathlib import Path
from typing import List


def load_image(file_path):
    """Загружает изображение из файла (PDF или JPG/PNG)"""
    file_path = Path(file_path)

    if file_path.suffix.lower() == '.pdf':
        # Для PDF берем только первую страницу
        images = pdf_to_images(file_path, dpi=300)
        if images:
            return images[0]
        else:
            raise ValueError(f"Не удалось извлечь изображение из PDF: {file_path}")
    else:
        # Загружаем обычное изображение
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {file_path}")
        return image


def pdf_to_images(pdf_path, dpi=300) -> List[np.ndarray]:
    """Конвертирует все страницы PDF в список изображений"""
    import fitz  # PyMuPDF
    
    images = []
    doc = fitz.open(pdf_path)
    
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # увеличиваем DPI для лучшего качества
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        
        # Конвертируем в формат OpenCV
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if pix.n == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        images.append(img_array)
    
    doc.close()
    return images


def save_image(image, path):
    """Сохраняет изображение по указанному пути"""
    cv2.imwrite(str(path), image)


def resize_for_display(image, max_width=800, max_height=600):
    """Изменяет размер изображения для отображения на экране"""
    height, width = image.shape[:2]

    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))

    return image


def get_supported_extensions():
    """Возвращает список поддерживаемых расширений файлов"""
    return ['.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']