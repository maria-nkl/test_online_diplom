import os
import re
import json
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import shutil
import random
import sys
from collections import Counter
from pathlib import Path
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = Path(__file__).parent  # C:\Users\nikol\backend\modules\blog\new_blanc
SYMBOL_DIR = BASE_DIR / 'symbol'   # C:\Users\nikol\backend\modules\blog\new_blanc\symbol
MODEL_DIR = BASE_DIR / 'model'     # C:\Users\nikol\backend\modules\blog\new_blanc\model

def convert_to_serializable(obj):
    """Рекурсивно конвертирует numpy типы в стандартные Python типы для JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


class TextPostProcessor:
    def __init__(self):
        self.ambiguous_map = {'0': 'О', '4': 'Ч', '3':'З', '2':'Д', '9':'У', '6':'С'}
        self.letter_like_digits = {'0', '2', '3', '4', '6', '9'}
    
    def correct_ambiguous_symbols(self, text):
        if not text:
            return text
        text_list = list(text)
        for i, char in enumerate(text_list):
            if char in self.letter_like_digits:
                is_part_of_word = (
                    (i > 0 and text_list[i-1].isalpha()) or
                    (i < len(text_list)-1 and text_list[i+1].isalpha())
                )
                if is_part_of_word:
                    text_list[i] = self.ambiguous_map[char]
        return ''.join(text_list)
    
    def fix_hyphenation(self, text):
        text = re.sub(r'(\w+)-(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)\s+-\s+(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)\s+-(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)-\s+', r'\1 ', text)
        text = re.sub(r'(\w+)-\s*$', r'\1', text)
        return text
    
    def normalize_spaces(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,!?:;])', r'\1', text)
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        text = text.strip()
        return text
    
    def process(self, text):
        if not text:
            return text
        text = self.correct_ambiguous_symbols(text)
        text = self.fix_hyphenation(text)
        text = self.normalize_spaces(text)
        return text


class FineTuningDataset(Dataset):
    def __init__(self, images, labels, img_size=(64, 64)):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class RussianHandwritingRecognizerInference:
    def __init__(self, model_path='russian_handwriting_model_complete.pth', 
                 encoder_path='label_encoder.pkl', img_size=(64, 64)):
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.post_processor = TextPostProcessor()
        
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.img_size = checkpoint['img_size']
        num_classes = len(self.label_encoder.classes_)
        
        self.model = self._create_model(num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Модель загружена успешно!")
        print(f"Количество распознаваемых символов: {len(self.label_encoder.classes_)}")
    
    def _create_model(self, num_classes):
        class CNNModel(nn.Module):
            def __init__(self, num_classes, img_size=64):
                super(CNNModel, self).__init__()
                
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(0.25),
                )
                
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 4 * 4, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x
        
        return CNNModel(num_classes, self.img_size[0])
    
    def predict_single(self, image_path):
        img = Image.open(image_path).convert('L')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_symbol = self.label_encoder.inverse_transform([predicted.cpu().numpy()[0]])[0]
        return predicted_symbol, confidence.cpu().numpy()[0]
    
    def predict_folder(self, folder_path, sort_by_name=True, apply_postprocessing=True):
        if not os.path.exists(folder_path):
            return "", [], []
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(image_extensions)]
        
        if not image_files:
            return "", [], []
        
        if sort_by_name:
            def extract_number(filename):
                numbers = re.findall(r'\d+', filename)
                return int(numbers[0]) if numbers else 0
            image_files.sort(key=extract_number)
        
        image_paths = [os.path.join(folder_path, f) for f in image_files]
        
        raw_text = ""
        confidences = []
        for path in image_paths:
            symbol, conf = self.predict_single(path)
            raw_text += symbol
            confidences.append(conf)
        
        if apply_postprocessing:
            return self.post_processor.process(raw_text), raw_text, confidences
        return raw_text, raw_text, confidences
    
    def predict_text_answers(self, text_answer_folder, sort_by_name=True, apply_postprocessing=True):
        if not os.path.exists(text_answer_folder):
            return {}
        
        answers = {}
        row_pattern = re.compile(r'^row_(\d+)$', re.IGNORECASE)
        row_folders = []
        
        for item in os.listdir(text_answer_folder):
            item_path = os.path.join(text_answer_folder, item)
            if os.path.isdir(item_path):
                match = row_pattern.match(item)
                if match:
                    row_num = int(match.group(1))
                    row_folders.append((row_num, item_path))
        
        row_folders.sort(key=lambda x: x[0])
        
        for row_num, row_path in row_folders:
            text, _, _ = self.predict_folder(row_path, sort_by_name, apply_postprocessing)
            answers[f'row_{row_num}'] = text
        
        return answers
    
    def fine_tune(self, finetune_dataset_path, epochs=4, learning_rate=0.0001):
        """Дообучение модели на датасете new_data"""
        print(f"\n{'='*70}")
        print("🎯 НАЧАЛО ДООБУЧЕНИЯ МОДЕЛИ")
        print(f"{'='*70}")
        
        # Маппинг имен папок в символы для encoder
        folder_to_char = {
            '0пробел': ' ',
            '0тире': '-',
            '0точка': '.',
            '0восклзнак': '!',
            '0кавычки': '"',
            '0левскобка': '(',
            '0правскобка': ')',
            '0двоеточие': ':',
            '0точкасзапятой': ';',
            '0запятая': ','
        }
        
        images = []
        labels = []
        
        symbol_folders = [f for f in os.listdir(finetune_dataset_path) 
                        if os.path.isdir(os.path.join(finetune_dataset_path, f))]
        
        print(f"📂 Загрузка данных из {finetune_dataset_path}")
        print(f"Найдено папок с символами: {len(symbol_folders)}")
        
        skipped_chars = []
        
        for folder_name in symbol_folders:
            folder_path = os.path.join(finetune_dataset_path, folder_name)
            
            # Преобразуем имя папки в символ для encoder
            if folder_name in folder_to_char:
                char = folder_to_char[folder_name]
            else:
                char = folder_name
            
            # Проверяем, есть ли такой символ в encoder
            if char not in self.label_encoder.classes_:
                skipped_chars.append(f"{folder_name} -> '{char}'")
                continue
            
            # Получаем все изображения в папке
            image_files = [f for f in os.listdir(folder_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                print(f"⚠️ Нет изображений для папки '{folder_name}'")
                continue
            
            label_id = self.label_encoder.transform([char])[0]
            
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(label_id)
                except Exception as e:
                    print(f"Ошибка загрузки {img_path}: {e}")
            
            print(f"   ✅ '{folder_name}' -> '{char}': {len(image_files)} изображений")
            sys.stdout.flush()  # Принудительный вывод
        
        if skipped_chars:
            print(f"\n   ⚠️ Пропущены папки (нет в encoder): {skipped_chars}")
        
        if not images:
            print("❌ Нет данных для дообучения")
            return False
        
        print(f"\n📊 Всего загружено {len(images)} изображений")
        print(f"📊 Размер одного изображения: {images[0].shape}")
        print(f"📊 Количество классов: {len(set(labels))}")
        sys.stdout.flush()
        
        # Создание датасета
        print("\n🔄 Создание тензоров...")
        sys.stdout.flush()
        
        X = torch.FloatTensor(np.array(images)).reshape(-1, 1, self.img_size[0], self.img_size[1])
        y = torch.LongTensor(labels)
        
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        sys.stdout.flush()
        
        dataset = FineTuningDataset(X, y, self.img_size)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        print(f"   Создано {len(dataloader)} батчей по 32 изображения")
        sys.stdout.flush()
        
        # ========== НОВОЕ: ЗАМОРАЖИВАЕМ РАННИЕ СЛОИ ==========
        print("\n🔒 Настройка слоев для дообучения...")
        sys.stdout.flush()
        
        # 1. Сначала замораживаем ВСЕ параметры
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 2. Размораживаем только нужные слои
        
        # Размораживаем последний сверточный блок (последние 4 операции в conv_layers)
        # conv_layers содержит: Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout2d
        # Размораживаем последние 5 операций (последний сверточный блок)
        conv_children = list(self.model.conv_layers.children())
        for child in conv_children[-5:]:  # Последний сверточный блок
            for param in child.parameters():
                param.requires_grad = True
        
        # Размораживаем ВСЕ полносвязные слои
        for param in self.model.fc_layers.parameters():
            param.requires_grad = True
        
        # 3. Считаем статистику
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"   📊 Всего параметров: {total_params:,}")
        print(f"   🔓 Обучаемых параметров: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"   🔒 Замороженных параметров: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        sys.stdout.flush()
        
        # Настройка для дообучения
        print("\n🔄 Настройка оптимизатора...")
        sys.stdout.flush()
        
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        
        # Оптимизатор ТОЛЬКО для параметров с requires_grad=True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        print(f"   ✅ Оптимизатор настроен на {trainable_params:,} параметров")
        sys.stdout.flush()
        
        best_loss = float('inf')
        
        print(f"\n🚀 НАЧАЛО ОБУЧЕНИЯ ({epochs} эпох)")
        print("="*70)
        sys.stdout.flush()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                # Выводим прогресс каждые 10 батчей
                if (batch_idx + 1) % 10 == 0:
                    print(f"   Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
                    sys.stdout.flush()
            
            avg_loss = total_loss / len(dataloader)
            accuracy = 100 * correct / total
            
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            sys.stdout.flush()
        
        print(f"\n✅ Дообучение завершено!")
        return True
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'img_size': self.img_size
        }, filepath)
        print(f"Модель сохранена: {filepath}")


class StudentWorkProcessor:
    def __init__(self, cells_root_folder='cells', output_folder='recognition_results', 
                 test_task_folder=None):
        self.cells_root_folder = cells_root_folder
        self.output_folder = output_folder
        self.test_task_folder = test_task_folder
        self.recognizer = None
        self.target_sentence = "СЪЕШЬ ЕЩЁ ЭТИХ МЯГКИХ ФРАНЦУЗСКИХ БУЛОК ДА ВЫПЕЙ ЖЕ ЧАЮ."
        self.full_sentence_chars = list(self.target_sentence)
        
        self.special_chars_map = {
            ' ': '0пробел',
            '.': '0точка',
            '-': '0тире',
            '!': '0восклзнак',
            '"': '0кавычки',
            '(': '0левскобка',
            ')': '0правскобка',
            ':': '0двоеточие',
            ';': '0точкасзапятой',
            ',': '0запятая'
        }
        
        os.makedirs(output_folder, exist_ok=True)

    
    # ========== МЕТОДЫ ДЛЯ ОБРАБОТКИ ТЕСТОВЫХ ЗАДАНИЙ ==========
    
    def find_test_frames(self, image: np.ndarray):
        """Находит 5 вертикальных рамок с тестовыми заданиями"""
        h, w = image.shape[:2]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 7
        )
        
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_info = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, rect_w, rect_h = cv2.boundingRect(contour)
            
            if area > (w * h) * 0.5:
                continue
            
            rect_area = rect_w * rect_h
            fill_ratio = area / rect_area if rect_area > 0 else 0
            
            if (area > 80000 and area < 130000 and fill_ratio > 0.5):
                contour_info.append({
                    'bbox': (x, y, rect_w, rect_h),
                    'area': area,
                    'center_y': y + rect_h/2,
                })
        
        if len(contour_info) < 5:
            return []
        
        contour_info.sort(key=lambda c: c['center_y'])
        
        frames = []
        for i in range(0, len(contour_info), 2):
            if len(frames) < 5:
                frames.append(contour_info[i])
        
        if len(frames) != 5:
            frames = contour_info[:5]
        
        return [f['bbox'] for f in frames]
    
    def find_cells_in_test_frame(self, frame_image: np.ndarray):
        """Находит 4 клетки внутри тестовой рамки"""
        gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        best_cells = []
        
        for threshold in [30, 40, 50, 60, 70, 80, 90, 100]:
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            
            kernel = np.ones((2, 2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cells = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                if (w > 50 and w < 100 and
                    h > 50 and h < 90 and
                    area > 3000 and area < 8000):
                    
                    cells.append({
                        'bbox': (x, y, w, h),
                        'center_x': x + w/2,
                        'left_x': x,
                        'right_x': x + w,
                    })
            
            if len(cells) >= 4:
                cells.sort(key=lambda c: c['center_x'])
                cells = cells[:4]
                
                no_overlap = True
                for i in range(3):
                    if cells[i]['right_x'] >= cells[i+1]['left_x']:
                        no_overlap = False
                        break
                
                if no_overlap:
                    best_cells = cells
                    break
        
        if not best_cells:
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cells = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                if (w > 40 and w < 120 and
                    h > 40 and h < 100 and
                    area > 2000 and area < 10000):
                    
                    cells.append({
                        'bbox': (x, y, w, h),
                        'center_x': x + w/2,
                        'left_x': x,
                        'right_x': x + w
                    })
            
            if len(cells) >= 4:
                cells.sort(key=lambda c: c['center_x'])
                best_cells = cells[:4]
        
        return [c['bbox'] for c in best_cells] if best_cells else []
    
    def predict_test_cell(self, cell_image: np.ndarray) -> str:
        """Предсказывает состояние тестовой клетки"""
        try:
            temp_path = "temp_test_cell.jpg"
            cv2.imwrite(temp_path, cell_image)
            
            predicted_symbol, _ = self.recognizer.predict_single(temp_path)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return "empty" if predicted_symbol == ' ' else "marked"
                
        except Exception as e:
            print(f"⚠️ Ошибка распознавания тестовой клетки: {e}")
            return "empty"
    
    def process_test_tasks_for_student(self, student_name: str) -> dict:
        """Обрабатывает тестовые задания для конкретного ученика"""
        if not self.test_task_folder:
            return {}
        
        # Ищем изображение ученика в папке test_task
        test_task_path = Path(self.test_task_folder)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        student_image = None
        for ext in image_extensions:
            potential_file = test_task_path / f"{student_name}{ext}"
            if potential_file.exists():
                student_image = potential_file
                break
        
        if not student_image:
            # Пробуем найти по части имени
            for file in test_task_path.iterdir():
                if file.suffix.lower() in image_extensions:
                    if student_name.lower() in file.stem.lower():
                        student_image = file
                        break
        
        if not student_image:
            return {}
        
        print(f"   📝 Обработка тестовых заданий: {student_image.name}")
        
        image = cv2.imread(str(student_image))
        if image is None:
            return {}
        
        # Находим рамки
        frames = self.find_test_frames(image)
        
        if len(frames) != 5:
            print(f"   ⚠️ Найдено {len(frames)} рамок, нужно 5")
            return {}
        
        test_results = {}
        
        for frame_idx, frame_bbox in enumerate(frames, 1):
            x, y, w, h = frame_bbox
            frame_image = image[y:y+h, x:x+w]
            
            # Находим клетки
            cells = self.find_cells_in_test_frame(frame_image)
            
            frame_results = {}
            for cell_idx, cell_bbox in enumerate(cells, 1):
                cx, cy, cw, ch = cell_bbox
                cell_image = frame_image[cy:cy+ch, cx:cx+cw]
                
                status = self.predict_test_cell(cell_image)
                frame_results[str(cell_idx)] = status
            
            test_results[str(frame_idx)] = frame_results
        
        return test_results

    
    def init_recognizer(self, model_path='model/russian_handwriting_model_complete.pth', 
                        encoder_path='model/label_encoder.pkl'):
        self.recognizer = RussianHandwritingRecognizerInference(
            model_path=model_path, encoder_path=encoder_path
        )
    
    def get_student_name(self, folder_name):
        # Убираем суффикс _cells
        folder_name = folder_name.replace('_cells', '')
        
        # Возвращаем полное имя с номером страницы, если он есть
        # Например: Bushueva_merged_page_001 -> Bushueva_merged_page_001
        match = re.match(r'^(.+?)(?:_page_\d+)?$', folder_name)
        if match:
            return match.group(0)  # Возвращаем полное имя
        
        return folder_name
    
    def compare_texts(self, recognized_text):
        recognized_clean = re.sub(r'[^\wЁ]', '', recognized_text.upper())
        target_clean = re.sub(r'[^\wЁ]', '', self.target_sentence.replace('-', '').upper())
        
        errors = []
        max_len = max(len(recognized_clean), len(target_clean))
        
        for i in range(max_len):
            if i < len(recognized_clean) and i < len(target_clean):
                if recognized_clean[i] != target_clean[i]:
                    errors.append({
                        'position': i,
                        'expected': target_clean[i],
                        'recognized': recognized_clean[i]
                    })
            elif i < len(target_clean):
                errors.append({
                    'position': i,
                    'expected': target_clean[i],
                    'recognized': None
                })
        
        return errors
    
    def get_folder_name_for_char(self, char):
        if char in self.special_chars_map:
            return self.special_chars_map[char]
        return char
    
    def create_balanced_dataset(self, student_folder):
        test_text_folder = os.path.join(student_folder, 'test_text_cells')
        
        if not os.path.exists(test_text_folder):
            print(f"   ⚠️ Папка test_text_cells не найдена")
            return None
        
        target_sentence = "СЪЕШЬ ЕЩЁ ЭТИХ МЯГКИХ ФРАНЦУ-ЗСКИХ БУЛОК ДА ВЫПЕЙ ЖЕ ЧАЮ."
        expected_chars = list(target_sentence)
        
        position_to_file = {}
        
        for filename in os.listdir(test_text_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                match = re.search(r'_(\d+)\.', filename)
                if match:
                    file_num = int(match.group(1))
                    position = file_num - 1
                    position_to_file[position] = os.path.join(test_text_folder, filename)
        
        print(f"\n   📁 Найдено {len(position_to_file)} файлов")
        
        char_counts = Counter(expected_chars)
        
        print(f"\n   🎯 Цель: ровно 70 изображений для каждого символа (48 копий из бланка + 22 из старого датасета)")
        
        new_data_path = 'new_data'
        if os.path.exists(new_data_path):
            shutil.rmtree(new_data_path)
        os.makedirs(new_data_path)
        
        # Создаем папки для всех символов
        for char in set(expected_chars):
            folder_name = self.get_folder_name_for_char(char)
            target_folder = os.path.join(new_data_path, folder_name)
            os.makedirs(target_folder, exist_ok=True)
        
        # Для каждого уникального символа
        for char in set(expected_chars):
            folder_name = self.get_folder_name_for_char(char)
            target_folder = os.path.join(new_data_path, folder_name)
            positions = [i for i, c in enumerate(expected_chars) if c == char]
            
            # Сколько раз этот символ встречается в тексте
            occurrences = len(positions)
            
            # Вычисляем сколько копий нужно с каждой позиции чтобы получить 48 всего
            if occurrences > 0:
                copies_per_position = 48 // occurrences  # Целочисленное деление
                remainder = 48 % occurrences  # Остаток распределим по первым позициям
            else:
                copies_per_position = 0
                remainder = 0
            
            current_count = 0
            
            # Копируем из бланка ученика
            for idx, pos in enumerate(positions):
                if pos in position_to_file:
                    src_path = position_to_file[pos]
                    
                    # Сколько копий для этой позиции
                    num_copies = copies_per_position
                    if idx < remainder:  # Распределяем остаток
                        num_copies += 1
                    
                    for copy_num in range(num_copies):
                        dst_filename = f"pos{pos+1}_copy{copy_num}.png"
                        dst_path = os.path.join(target_folder, dst_filename)
                        shutil.copy2(src_path, dst_path)
                        current_count += 1
            
            print(f"   📂 '{folder_name}': символ '{char}' встречается {occurrences} раз(а), создано {current_count} копий из бланка")
            
            # Добавляем изображения из старого датасета (ровно 22 штуки)
            old_char_path = SYMBOL_DIR / folder_name
            old_added = 0
            
            if os.path.exists(old_char_path):
                old_images = [f for f in os.listdir(old_char_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if old_images:
                    # Берем ровно 22 случайных изображения (или сколько есть)
                    num_old = min(22, len(old_images))
                    
                    if len(old_images) >= 22:
                        selected = random.sample(old_images, 22)
                    else:
                        selected = old_images
                        print(f"   ⚠️ '{folder_name}': в старом датасете только {len(old_images)} изображений (нужно 22)")
                    
                    for i, img_name in enumerate(selected):
                        src_path = os.path.join(old_char_path, img_name)
                        dst_filename = f"old_{i}_{img_name}"
                        dst_path = os.path.join(target_folder, dst_filename)
                        shutil.copy2(src_path, dst_path)
                        old_added += 1
                    
                    print(f"   ✅ '{folder_name}': добавлено {old_added} фото из старого датасета")
            else:
                print(f"   ⚠️ '{folder_name}': папка в старом датасете не найдена")
            
            total_images = current_count + old_added
            print(f"   📊 '{folder_name}': ВСЕГО {total_images} изображений (цель: 70)")
            
            if total_images < 70:
                print(f"   ⚠️ '{folder_name}': не хватает {70 - total_images} изображений до 70")
        
        return new_data_path

    def process_student_with_finetune(self, student_folder):
        student_name = self.get_student_name(os.path.basename(student_folder))
        print(f"\n{'='*70}")
        print(f"📚 Обработка работ ученика: {student_name}")
        print(f"{'='*70}")
        
        student_result = {
            'student_name': student_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # ========== 1. ОБРАБОТКА ТЕСТОВЫХ ЗАДАНИЙ (НОВОЕ) ==========
        if self.test_task_folder:
            test_results = self.process_test_tasks_for_student(student_name)
            if test_results:
                # Конвертируем в формат для шаблона
                option_map = {1: "A", "1": "A", 2: "B", "2": "B", 3: "C", "3": "C", 4: "D", "4": "D"}
                test_task_converted = {}
                
                for question_num, options in test_results.items():
                    marked_letters = [
                        option_map.get(opt_num) 
                        for opt_num, status in options.items() 
                        if status == "marked" and opt_num in option_map
                    ]
                    test_task_converted[str(question_num)] = {"answers_letters": marked_letters}
                
                student_result['test_task'] = test_task_converted
                print(f"   ✅ Тестовые задания обработаны: {len(test_results)} заданий")
        

        # ========== 2. ТЕСТОВЫЙ ТЕКСТ ==========
        folder = os.path.join(student_folder, 'test_text_cells')
        finetuned_recognizer = None  # Для хранения дообученного распознавателя
        
        if os.path.exists(folder):
            text, raw_text, confidences = self.recognizer.predict_folder(folder)
            student_result['test_text'] = text
            print(f"   ✅ Тестовый текст (до дообучения): {text}")
            
            errors = self.compare_texts(text)
            error_count = len(errors)
            
            # Дообучение только если ошибок 3 или больше
            if error_count >= 3:
                print(f"   ⚠️ Обнаружены ошибки в распознавании ({error_count} ошибок)!")
                print(f"   🔄 Запуск процесса дообучения для этого ученика...")
                
                new_dataset_path = self.create_balanced_dataset(student_folder)
                
                if new_dataset_path and os.path.exists(new_dataset_path):
                    original_model_path = MODEL_DIR / 'russian_handwriting_model_complete.pth'
                    original_encoder_path = MODEL_DIR / 'label_encoder.pkl'
                    
                    if not original_model_path.exists():
                        # Пробуем в текущей директории
                        original_model_path = BASE_DIR / 'russian_handwriting_model_complete.pth'
                        original_encoder_path = BASE_DIR / 'label_encoder.pkl'
                    
                    if not original_model_path.exists():
                        print(f"   ❌ Модель не найдена: {original_model_path}")
                        return student_result
                    
                    temp_model_path = 'temp_finetuned_model.pth'
                    temp_encoder_path = 'temp_finetuned_encoder.pkl'
                    
                    shutil.copy2(original_encoder_path, temp_encoder_path)
                    
                    try:
                        temp_recognizer = RussianHandwritingRecognizerInference(
                            model_path=original_model_path,
                            encoder_path=original_encoder_path
                        )
                    except Exception as e:
                        print(f"   ❌ Ошибка загрузки модели: {e}")
                        if os.path.exists(new_dataset_path):
                            shutil.rmtree(new_dataset_path)
                        return student_result
                    
                    success = temp_recognizer.fine_tune(
                        finetune_dataset_path=new_dataset_path,
                        epochs=4,
                        learning_rate=0.0001
                    )
                    
                    if success:
                        temp_recognizer.save_model(temp_model_path)
                        
                        print(f"\n   🔄 Повторное распознавание тестового текста с дообученной моделью...")
                        
                        try:
                            finetuned_recognizer = RussianHandwritingRecognizerInference(
                                model_path=temp_model_path,
                                encoder_path=temp_encoder_path
                            )
                            
                            new_text, new_raw_text, new_confidences = finetuned_recognizer.predict_folder(folder)
                            student_result['test_text'] = new_text
                            
                            new_errors_count = len(self.compare_texts(new_text))
                            print(f"   ✅ После дообучения: {new_text}")
                            print(f"   📊 Ошибок было: {error_count}, стало: {new_errors_count}")
                            
                        except Exception as e:
                            print(f"   ❌ Ошибка при повторном распознавании: {e}")
                            finetuned_recognizer = None
                    else:
                        print(f"   ❌ Ошибка при дообучении модели")
                        if os.path.exists(new_dataset_path):
                            shutil.rmtree(new_dataset_path)
                        return student_result
                    
                    if os.path.exists(new_dataset_path):
                        shutil.rmtree(new_dataset_path)
            elif error_count > 0:
                print(f"   ⚠️ Обнаружены ошибки в распознавании ({error_count} ошибок), но дообучение не требуется (нужно 3+ ошибок)")
            else:
                print(f"   ✅ Текст распознан верно! Дообучение не требуется.")
        else:
            student_result['test_text'] = ''
            print(f"   ⚠️ Папка test_text_cells не найдена")
        
        # ========== 3. ИСПОЛЬЗУЕМ ДООБУЧЕННУЮ МОДЕЛЬ (ЕСЛИ ЕСТЬ) ДЛЯ ОСТАЛЬНЫХ ПОЛЕЙ ==========
        # Выбираем какой распознаватель использовать
        current_recognizer = finetuned_recognizer if finetuned_recognizer else self.recognizer
        
        # name_cells
        folder = os.path.join(student_folder, 'name_cells')
        if os.path.exists(folder):
            text, _, _ = current_recognizer.predict_folder(folder)
            student_result['name'] = text
            print(f"   ✅ Имя: {text}")
        else:
            student_result['name'] = ''
            print(f"   ⚠️ Папка name_cells не найдена")
        
        # printed_text_cells
        folder = os.path.join(student_folder, 'printed_text_cells')
        if os.path.exists(folder):
            text, _, _ = current_recognizer.predict_folder(folder)
            student_result['printed_text'] = text
            print(f"   ✅ Печатный текст: {text[:100]}{'...' if len(text) > 100 else ''}")
        else:
            student_result['printed_text'] = ''
            print(f"   ⚠️ Папка printed_text_cells не найдена")
        
        # text_answer_cells
        folder = os.path.join(student_folder, 'text_answer_cells')
        if os.path.exists(folder):
            answers = current_recognizer.predict_text_answers(folder)
            student_result['text_answers'] = answers
            answered = [k for k, v in answers.items() if v]
            print(f"   ✅ Ответы: {len(answered)} из 10")
            for row, text in answers.items():
                if text:
                    print(f"      {row}: {text[:50]}{'...' if len(text) > 50 else ''}")
        else:
            student_result['text_answers'] = {}
            print(f"   ⚠️ Папка text_answer_cells не найдена")
        
        # ========== 4. УДАЛЯЕМ ВРЕМЕННЫЕ ФАЙЛЫ ==========
        if finetuned_recognizer:
            temp_model_path = 'temp_finetuned_model.pth'
            temp_encoder_path = 'temp_finetuned_encoder.pkl'
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                print(f"   🗑️ Временная модель удалена")
            if os.path.exists(temp_encoder_path):
                os.remove(temp_encoder_path)
                print(f"   🗑️ Временный encoder удален")
        
        return student_result
    
    def process_all_students(self, model_path='russian_handwriting_model_complete.pth',
                            encoder_path='label_encoder.pkl'):
        model_path = Path(model_path)
        encoder_path = Path(encoder_path)
        
        if not model_path.is_absolute():
            # Сначала ищем в MODEL_DIR
            if (MODEL_DIR / model_path).exists():
                model_path = MODEL_DIR / model_path
                encoder_path = MODEL_DIR / encoder_path
            # Потом в BASE_DIR
            elif (BASE_DIR / model_path).exists():
                model_path = BASE_DIR / model_path
                encoder_path = BASE_DIR / encoder_path
            else:
                print(f"❌ Модель не найдена: {model_path}")
                return []
        
        if not os.path.exists(encoder_path):
            print(f"❌ Encoder не найден: {encoder_path}")
            return []
        
        self.init_recognizer(model_path, encoder_path)
        
        if not os.path.exists(self.cells_root_folder):
            print(f"❌ Папка {self.cells_root_folder} не найдена!")
            return []
        
        student_folders = [f for f in os.listdir(self.cells_root_folder) 
                          if os.path.isdir(os.path.join(self.cells_root_folder, f))
                          and '_cells' in f]
        
        if not student_folders:
            print(f"❌ В папке {self.cells_root_folder} не найдено папок учеников")
            return []
        
        print(f"\n{'='*70}")
        print(f"🔍 НАЙДЕНО УЧЕНИКОВ: {len(student_folders)}")
        print(f"{'='*70}")
        
        all_results = []
        
        for student_folder in student_folders:
            student_path = os.path.join(self.cells_root_folder, student_folder)
            result = self.process_student_with_finetune(student_path)
            all_results.append(result)
        
        master_json = os.path.join(self.output_folder, 'master_report.json')
        with open(master_json, 'w', encoding='utf-8') as f:
            json.dump(convert_to_serializable(all_results), f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 Отчет сохранен: {master_json}")
        
        return all_results
    
    def print_summary(self, all_results):
        print(f"\n{'='*70}")
        print("📊 КРАТКАЯ СВОДКА РЕЗУЛЬТАТОВ")
        print(f"{'='*70}")
        
        for result in all_results:
            student_name = result['student_name']
            print(f"\n👤 {student_name}:")
            print(f"   📝 Имя: {result.get('name', '')}")
            print(f"   📝 Печатный текст: {result.get('printed_text', '')[:60]}...")
            print(f"   📝 Тестовый текст: {result.get('test_text', '')[:60]}...")
            if 'test_task' in result:
                print(f"   📝 Тестовых заданий: {len(result['test_task'])}")


def main():
    processor = StudentWorkProcessor(
        cells_root_folder='cells',
        output_folder='recognition_results'
    )
    
    all_results = processor.process_all_students(
        model_path='russian_handwriting_model_complete.pth',
        encoder_path='label_encoder.pkl'
    )
    
    if all_results:
        processor.print_summary(all_results)
        print(f"\n{'='*70}")
        print("✅ ОБРАБОТКА ЗАВЕРШЕНА!")
        print(f"{'='*70}")
        print(f"Результат: recognition_results/master_report.json")
    else:
        print("\n❌ Обработка не выполнена.")


if __name__ == "__main__":
    main()