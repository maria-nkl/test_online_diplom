import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import warnings
warnings.filterwarnings('ignore')

# Добавляем LabelEncoder в безопасные глобальные объекты PyTorch
try:
    torch.serialization.add_safe_globals([LabelEncoder])
except AttributeError:
    pass  # Для старых версий PyTorch

# Установка seed для воспроизводимости
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class RussianHandwritingDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class CNNModel(nn.Module):
    def __init__(self, num_classes, img_size=64):
        super(CNNModel, self).__init__()
        
        # Сверточная часть
        self.conv_layers = nn.Sequential(
            # Блок 1: 64x64 -> 32x32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Блок 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Блок 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Блок 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        
        # Полносвязная часть
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

class RussianHandwritingRecognizer:
    def __init__(self, data_path='symbol', img_size=(64, 64)):
        self.data_path = data_path
        self.img_size = img_size
        self.model = None
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")
        
        # Маппинг имен папок на символы
        self.folder_to_symbol = {
            '0восклзнак': '!',
            '0выпрзнак': '"',
            '0двоеточие': ':',
            '0запятая': ',',
            '0кавычки': '"',
            '0левскобка': '(',
            '0правскобка': ')',
            '0тире': '-',
            '0точка': '.',
            '0точкасзапятой': ';',
            '0пробел': ' '
        }
    
    def load_data(self):
        """Загрузка данных из папок"""
        images = []
        labels = []
        
        print("Загрузка данных...")
        print(f"Поиск данных в папке: {os.path.abspath(self.data_path)}")
        
        if not os.path.exists(self.data_path):
            print(f"Ошибка: Папка {self.data_path} не найдена!")
            return np.array([]), np.array([])
        
        folders = [f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]
        print(f"Найдено папок: {len(folders)}")
        
        for folder_name in folders:
            folder_path = os.path.join(self.data_path, folder_name)
            
            # Определяем метку (символ)
            if folder_name in self.folder_to_symbol:
                label = self.folder_to_symbol[folder_name]
            else:
                label = folder_name
            
            image_count = 0
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(folder_path, img_file)
                    try:
                        # Загружаем и обрабатываем изображение
                        img = Image.open(img_path).convert('L')  # Оттенки серого
                        img = img.resize(self.img_size)
                        img_array = np.array(img) / 255.0  # Нормализация
                        images.append(img_array)
                        labels.append(label)
                        image_count += 1
                    except Exception as e:
                        print(f"Ошибка загрузки {img_path}: {e}")
            
            if image_count > 0:
                print(f"Загружено {image_count} изображений для символа '{label}'")
        
        if len(images) == 0:
            print("Ошибка: Не найдено ни одного изображения!")
            return np.array([]), np.array([])
        
        # Преобразуем в numpy массивы (формат: N, 1, H, W)
        X = np.array(images).reshape(-1, 1, self.img_size[0], self.img_size[1])
        y = np.array(labels)
        
        print(f"\nВсего загружено {len(X)} изображений")
        print(f"Уникальных символов: {len(np.unique(y))}")
        print(f"Символы: {sorted(np.unique(y))}")
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2, val_size=0.1, batch_size=32):
        """Подготовка данных для обучения"""
        # Кодируем метки
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Разделяем на train, validation и test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        val_relative = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative, random_state=42, stratify=y_temp
        )
        
        # Создаем датасеты
        train_dataset = RussianHandwritingDataset(X_train, y_train)
        val_dataset = RussianHandwritingDataset(X_val, y_val)
        test_dataset = RussianHandwritingDataset(X_test, y_test)
        
        # Создаем DataLoader'ы
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\nРазмер обучающей выборки: {len(X_train)}")
        print(f"Размер валидационной выборки: {len(X_val)}")
        print(f"Размер тестовой выборки: {len(X_test)}")
        print(f"Количество классов: {len(self.label_encoder.classes_)}")
        
        return train_loader, val_loader, test_loader
    
    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001):
        """Обучение модели"""
        num_classes = len(self.label_encoder.classes_)
        self.model = CNNModel(num_classes, self.img_size[0]).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': []
        }
        
        best_val_accuracy = 0
        
        print("\nНачало обучения модели...")
        
        for epoch in range(epochs):
            # Обучение
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_accuracy = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Валидация
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # Сохраняем историю
            history['train_losses'].append(avg_train_loss)
            history['val_losses'].append(avg_val_loss)
            history['train_accuracies'].append(train_accuracy)
            history['val_accuracies'].append(val_accuracy)
            
            scheduler.step(avg_val_loss)
            
            # Сохраняем лучшую модель (с весами только для совместимости)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                # Сохраняем только веса модели, а не весь словарь с label_encoder
                torch.save(self.model.state_dict(), 'best_model_weights.pth')
                print(f"Epoch {epoch+1}: Сохранена лучшая модель с точностью: {val_accuracy:.2f}%")
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Загружаем лучшие веса
        self.model.load_state_dict(torch.load('best_model_weights.pth', map_location=self.device))
        
        return history
    
    def evaluate(self, test_loader):
        """Оценка модели на тестовой выборке"""
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        print(f"\nТестовая точность: {accuracy:.2f}%")
        
        return accuracy, all_predictions, all_labels
    
    def predict(self, image_path):
        """Предсказание символа на одном изображении"""
        self.model.eval()
        
        # Загружаем и обрабатываем изображение
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
    
    def save_model(self, filepath='russian_handwriting_model_complete.pth'):
        """Сохранение модели с метаданными"""
        # Сохраняем label_encoder отдельно как pickle
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Сохраняем только веса модели
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'img_size': self.img_size
        }, filepath)
        
        print(f"Модель сохранена в {filepath}")
        print(f"Label encoder сохранен в label_encoder.pkl")
    
    def load_model(self, model_path='russian_handwriting_model_complete.pth', encoder_path='label_encoder.pkl'):
        """Загрузка модели и label encoder"""
        # Загружаем label_encoder
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Загружаем веса модели
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.img_size = checkpoint['img_size']
        num_classes = len(self.label_encoder.classes_)
        
        self.model = CNNModel(num_classes, self.img_size[0]).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Модель загружена из {model_path}")
        print(f"Загружено {len(self.label_encoder.classes_)} классов")
    
    def plot_training_history(self, history):
        """Визуализация обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history['train_accuracies'], label='Train Accuracy')
        ax1.plot(history['val_accuracies'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(history['train_losses'], label='Train Loss')
        ax2.plot(history['val_losses'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """Основная функция"""
    # Создаем экземпляр распознавателя
    recognizer = RussianHandwritingRecognizer(data_path='symbol', img_size=(64, 64))
    
    # Загружаем данные
    X, y = recognizer.load_data()
    
    if len(X) == 0:
        print("Ошибка: Не найдено изображений в папке 'symbol'")
        print("Убедитесь, что:")
        print("1. Папка 'symbol' существует в текущей директории")
        print(f"2. Текущая директория: {os.getcwd()}")
        print("3. В папке 'symbol' есть подпапки с изображениями")
        return
    
    # Подготовка данных
    train_loader, val_loader, test_loader = recognizer.preprocess_data(X, y, batch_size=32)
    
    # Обучение
    history = recognizer.train(train_loader, val_loader, epochs=50)
    
    # Визуализация
    recognizer.plot_training_history(history)
    
    # Оценка
    test_accuracy, _, _ = recognizer.evaluate(test_loader)
    
    # Сохранение модели
    recognizer.save_model()
    
    print("\nОбучение завершено успешно!")
    print(f"Финальная точность на тестовой выборке: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()