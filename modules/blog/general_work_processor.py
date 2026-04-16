"""
Обработчик для новых бланков (общая работа)
"""
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Путь к папке с обработчиком
NEW_BLANC_PATH = Path(__file__).parent / 'new_blanc'

def convert_test_results_to_task_format(test_results: dict) -> dict:
    """
    Конвертирует формат тестовых результатов:
    Из: {"1": {"1": "marked", "2": "empty", "3": "marked", "4": "empty"}, "3": {}}
    В:  {"1": {"answers_letters": ["A", "C"]}, "3": {"answers_letters": []}}
    """
    if not test_results or not isinstance(test_results, dict):
        return {}
    
    # Поддерживаем и строковые, и числовые ключи
    option_map = {1: "A", "1": "A", 2: "B", "2": "B", 3: "C", "3": "C", 4: "D", "4": "D"}
    converted = {}
    
    for question_num, options in test_results.items():
        if not isinstance(options, dict):
            continue
            
        # Собираем отмеченные варианты
        marked_letters = [
            option_map.get(opt_num) 
            for opt_num, status in options.items() 
            if status == "marked" and opt_num in option_map
        ]
        
        # ✅ ВСЕГДА сохраняем задание, даже если список ответов пуст
        converted[str(question_num)] = {"answers_letters": marked_letters}
    
    return converted

def process_general_work(file_path: str, original_filename: str = None) -> dict:
    import sys
    import os
    import shutil
    import tempfile
    import uuid
    import json
    from pathlib import Path
    from datetime import datetime
    import re

    # --- ПУТИ ---
    BASE_DIR = Path("media/processing_results")
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    if str(NEW_BLANC_PATH) not in sys.path:
        sys.path.insert(0, str(NEW_BLANC_PATH))

    from utils import load_image, save_image
    from image_normalized import process_image
    from text_regions import analyze_text_regions
    from use_model import StudentWorkProcessor

    # --- ИМЯ ФАЙЛА ---
    if original_filename:
        base_name = Path(original_filename).stem
        base_name = re.sub(r'[^\w\-_]', '_', base_name)
    else:
        base_name = Path(file_path).stem

    unique_id = uuid.uuid4().hex[:6]
    temp_dir = Path(tempfile.gettempdir()) / f"work_{base_name}_{unique_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    persistent_dir = BASE_DIR / f"{base_name}_{unique_id}"
    persistent_dir.mkdir(parents=True, exist_ok=True)

    in_folder = temp_dir / 'in'
    out_folder = temp_dir / 'out'
    cells_folder = temp_dir / 'cells'
    result_folder = temp_dir / 'recognition_results'

    for folder in [in_folder, out_folder, cells_folder, result_folder]:
        folder.mkdir(parents=True, exist_ok=True)

    clean_filename = original_filename or f"{base_name}.pdf"
    clean_filename = clean_filename.replace(' ', '_')
    input_file = in_folder / clean_filename
    shutil.copy2(file_path, input_file)

    try:
        # --- ШАГ 1: Предобработка ---
        image = load_image(input_file)
        processed_image, _, _ = process_image(image, debug=False)
        if processed_image is None:
            raise ValueError("Ошибка обработки изображения")

        processed_path = out_folder / f"{input_file.stem}_processed.jpg"
        save_image(processed_image, processed_path)

        analyze_text_regions(processed_image, debug=False, output_path=out_folder, original_filename=input_file.stem)

        # --- ШАГ 2: Разбиение на клетки ---
        import find_cells
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        find_cells.main()
        os.chdir(original_cwd)

        # --- ШАГ 3: ЕДИНАЯ ОБРАБОТКА ВСЕГО (включая тесты) ---
        model_path = NEW_BLANC_PATH / 'model' / 'russian_handwriting_model_complete.pth'
        encoder_path = NEW_BLANC_PATH / 'model' / 'label_encoder.pkl'
        test_task_folder = out_folder / 'test_task'

        if not model_path.exists() or not encoder_path.exists():
            raise ValueError(f"Модель не найдена: {model_path}")

        # Создаем процессор с указанием папки тестов
        processor = StudentWorkProcessor(
            cells_root_folder=str(cells_folder),
            output_folder=str(result_folder),
            test_task_folder=str(test_task_folder) if test_task_folder.exists() else None
        )
        
        # Запускаем единую обработку
        processor.process_all_students(
            model_path=str(model_path),
            encoder_path=str(encoder_path)
        )

        # --- ЧТЕНИЕ РЕЗУЛЬТАТА ИЗ ФАЙЛА (так как process_all_students сохраняет его) ---
        master_report_path = result_folder / 'master_report.json'
        if not master_report_path.exists():
            return {"students": {}}

        with open(master_report_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)

        # --- НОРМАЛИЗАЦИЯ ДАННЫХ ---
        # Если result_data - это список (старый формат)
        if isinstance(result_data, list):
            students_dict = {}
            for student in result_data:
                student_name = student.get("student_name", "unknown")
                students_dict[student_name] = student
            result_data = {"students": students_dict}
        
        # Если нет ключа students
        elif "students" not in result_data:
            result_data = {"students": result_data}


        # --- СОХРАНЕНИЕ В ПОСТОЯННУЮ ПАПКУ ---
        final_report_path = persistent_dir / "master_report.json"
        
        # Добавляем мета-информацию
        if "meta" not in result_data:
            result_data["meta"] = {}
        
        result_data["meta"].update({
            "saved_path": str(final_report_path),
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_students": len(result_data.get("students", {})),
            "original_filename": original_filename
        })
        
        with open(final_report_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        return result_data

    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Ошибка удаления temp: {e}")