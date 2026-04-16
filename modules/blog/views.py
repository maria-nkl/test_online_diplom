from django.http import JsonResponse
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.messages.views import SuccessMessageMixin
from django.urls import reverse_lazy
from django.shortcuts import redirect
import random
from django.db.models import Count
from taggit.models import Tag
from django.contrib.postgres.search import SearchVector, SearchQuery, SearchRank
from django.contrib import messages

from .models import Article, Category, Comment, ArticleFile
from .forms import ArticleCreateForm, ArticleUpdateForm, CommentCreateForm
from ..services.mixins import AuthorRequiredMixin
from .image_processor import ImageProcessor
from .general_work_processor import process_general_work


class ArticleListView(ListView):
    model = Article
    template_name = 'blog/articles_list.html'
    context_object_name = 'articles'
    paginate_by = 10

    def get_queryset(self):
        if not self.request.user.is_authenticated:
            return Article.objects.none()
        
        queryset = Article.objects.filter(author=self.request.user)\
                                 .select_related('author', 'category')\
                                 .prefetch_related('tags')\
                                 .order_by('-fixed', '-time_create')
        
        # Фильтр по типу работы (может быть несколько)
        work_type_filter = self.request.GET.get('work_type')
        if work_type_filter:
            work_types = work_type_filter.split(',')
            queryset = queryset.filter(work_type__in=work_types)
        
        # Фильтр по категориям (может быть несколько)
        category_filter = self.request.GET.get('category')
        if category_filter:
            categories = category_filter.split(',')
            queryset = queryset.filter(category__slug__in=categories)
        
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Мои работы' if self.request.user.is_authenticated else 'Доступ к работам'
        context['user_authenticated'] = self.request.user.is_authenticated
        context['current_work_type'] = self.request.GET.get('work_type', '')
        context['current_categories'] = self.request.GET.get('category', '')
        return context


class ArticleDetailView(DetailView):
    model = Article
    template_name = 'blog/articles_detail.html'
    context_object_name = 'article'
    queryset = model.objects.detail()

    def get_similar_articles(self, obj):
        article_tags_ids = obj.tags.values_list('id', flat=True)
        similar_articles = Article.objects.filter(tags__in=article_tags_ids).exclude(id=obj.id)
        similar_articles = similar_articles.annotate(related_tags=Count('tags')).order_by('-related_tags')
        similar_articles_list = list(similar_articles.all())
        random.shuffle(similar_articles_list)
        return similar_articles_list[:6]

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = self.object.title
        context['form'] = CommentCreateForm
        context['similar_articles'] = self.get_similar_articles(self.object)
        return context


class ArticleByCategoryListView(ListView):
    model = Article
    template_name = 'blog/articles_list.html'
    context_object_name = 'articles'
    category = None

    def get_queryset(self):
        self.category = Category.objects.get(slug=self.kwargs['slug'])
        queryset = Article.objects.all().filter(category__slug=self.category.slug)
        
        # Фильтр по типу работы
        work_type_filter = self.request.GET.get('work_type')
        if work_type_filter:
            work_types = work_type_filter.split(',')
            queryset = queryset.filter(work_type__in=work_types)
        
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = f'Статьи из категории: {self.category.title}'
        context['current_work_type'] = self.request.GET.get('work_type', '')
        return context


class ArticleByTagListView(ListView):
    model = Article
    template_name = 'blog/articles_list.html'
    context_object_name = 'articles'
    paginate_by = 10
    tag = None

    def get_queryset(self):
        self.tag = Tag.objects.get(slug=self.kwargs['tag'])
        queryset = Article.objects.all().filter(tags__slug=self.tag.slug)
        
        # Фильтр по типу работы
        work_type_filter = self.request.GET.get('work_type')
        if work_type_filter:
            work_types = work_type_filter.split(',')
            queryset = queryset.filter(work_type__in=work_types)
        
        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = f'Статьи по тегу: {self.tag.name}'
        context['current_work_type'] = self.request.GET.get('work_type', '')
        return context


class ArticleUpdateView(AuthorRequiredMixin, SuccessMessageMixin, UpdateView):
    model = Article
    template_name = 'blog/articles_update.html'
    form_class = ArticleUpdateForm
    success_message = 'Материал был успешно обновлен'

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['instance'] = self.get_object()
        if 'updater' in self.get_form_class().Meta.fields:
            kwargs['initial'] = {'updater': self.request.user}
        return kwargs

    def form_valid(self, form):
        # Handle file deletion
        if 'delete_files' in self.request.POST:
            ArticleFile.objects.filter(
                id__in=self.request.POST.getlist('delete_files'),
                article=self.object
            ).delete()

        # Handle new file uploads
        for file in self.request.FILES.getlist('files'):
            ArticleFile.objects.create(article=self.object, file=file)

        # Update updater field if it exists
        if 'updater' in form.fields:
            form.instance.updater = self.request.user

        return super().form_valid(form)


class ArticleDeleteView(AuthorRequiredMixin, DeleteView):
    model = Article
    success_url = reverse_lazy('home')
    context_object_name = 'article'
    template_name = 'blog/articles_delete.html'

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = f'Удаление статьи: {self.object.title}'
        return context


class CommentCreateView(LoginRequiredMixin, CreateView):
    model = Comment
    form_class = CommentCreateForm

    def is_ajax(self):
        return self.request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    def form_invalid(self, form):
        if self.is_ajax():
            return JsonResponse({'error': form.errors}, status=400)
        return super().form_invalid(form)

    def form_valid(self, form):
        comment = form.save(commit=False)
        comment.article_id = self.kwargs.get('pk')
        comment.author = self.request.user
        comment.parent_id = form.cleaned_data.get('parent')
        comment.save()

        if self.is_ajax():
            return JsonResponse({
                'is_child': comment.is_child_node(),
                'id': comment.id,
                'author': comment.author.username,
                'parent_id': comment.parent_id,
                'time_create': comment.time_create.strftime('%Y-%b-%d %H:%M:%S'),
                'avatar': comment.author.profile.avatar.url,
                'content': comment.content,
                'get_absolute_url': comment.author.profile.get_absolute_url()
            }, status=200)

        return redirect(comment.article.get_absolute_url())

    def handle_no_permission(self):
        return JsonResponse({'error': 'Необходимо авторизоваться для добавления комментариев'}, status=400)


class ArticleSearchResultView(ListView):
    model = Article
    context_object_name = 'articles'
    paginate_by = 10
    allow_empty = True
    template_name = 'blog/articles_list.html'

    def get_queryset(self):
        query = self.request.GET.get('do')
        if not query:
            return Article.objects.none()
        
        # Поиск по заголовку, описанию и типу работы
        from django.db.models import Q
        
        queryset = Article.objects.filter(
            Q(title__icontains=query) |
            Q(short_description__icontains=query) |
            Q(full_description__icontains=query) |
            Q(work_type__icontains=query)
        ).filter(author=self.request.user).distinct()
        
        # Фильтр по типу работы если есть
        work_type_filter = self.request.GET.get('work_type')
        if work_type_filter:
            work_types = work_type_filter.split(',')
            queryset = queryset.filter(work_type__in=work_types)
        
        return queryset.order_by('-fixed', '-time_create')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = f'Результаты поиска: {self.request.GET.get("do")}'
        context['current_work_type'] = self.request.GET.get('work_type', '')
        return context
    

import logging
logger = logging.getLogger(__name__)

# views.py - исправленная часть ArticleCreateView

class ArticleCreateView(LoginRequiredMixin, CreateView):
    model = Article
    form_class = ArticleCreateForm
    template_name = 'blog/articles_create.html'
    login_url = 'home'

    def form_valid(self, form):
        form.instance.author = self.request.user
        response = super().form_valid(form)
        
        work_type = form.cleaned_data.get('work_type')
        files = self.request.FILES.getlist('files')
        reference_file = self.request.FILES.get('reference_file')
        
        if not files:
            messages.warning(self.request, 'Не выбраны файлы для проверки')
            return response
        
        if work_type == 'test':
            # Для тестов - используем существующий ImageProcessor
            for file in files:
                ArticleFile.objects.create(article=self.object, file=file)
            messages.success(self.request, f'Загружено {len(files)} файлов. Результаты появятся после обработки.')
        
        else:
            # Для общих работ - используем general_work_processor
            import tempfile
            import os
            import fitz  # PyMuPDF
            
            try:
                # ИСПРАВЛЕНО: Сначала обрабатываем ЭТАЛОН и делаем его превью
                reference_results = None
                reference_file = self.request.FILES.get('reference_file')
                
                if reference_file:
                    # Сохраняем эталон временно
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_ref:
                        for chunk in reference_file.chunks():
                            tmp_ref.write(chunk)
                        tmp_ref_path = tmp_ref.name
                    
                    # Создаем превью из ЭТАЛОНА
                    try:
                        if reference_file.name.lower().endswith('.pdf'):
                            pdf_document = fitz.open(tmp_ref_path)
                            first_page = pdf_document[0]
                            pix = first_page.get_pixmap(dpi=150)
                            img_data = pix.tobytes("jpeg")
                            
                            from django.core.files.base import ContentFile
                            self.object.thumbnail.save(
                                f'preview_reference_{self.object.slug}.jpg', 
                                ContentFile(img_data), 
                                save=True
                            )
                            pdf_document.close()
                        else:
                            # Если эталон - изображение
                            from django.core.files.base import ContentFile
                            self.object.thumbnail.save(
                                f'preview_reference_{self.object.slug}.jpg', 
                                ContentFile(reference_file.read()), 
                                save=True
                            )
                            # Возвращаем указатель файла в начало
                            reference_file.seek(0)
                        
                        logger.info(f"✅ Превью создано из эталона: {reference_file.name}")
                    except Exception as e:
                        logger.warning(f"Не удалось создать превью из эталона: {e}")
                        # Если не удалось создать превью из эталона, пробуем из первого файла студента
                        if files:
                            first_file = files[0]
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_preview:
                                for chunk in first_file.chunks():
                                    tmp_preview.write(chunk)
                                tmp_preview_path = tmp_preview.name
                            
                            try:
                                if first_file.name.lower().endswith('.pdf'):
                                    pdf_document = fitz.open(tmp_preview_path)
                                    first_page = pdf_document[0]
                                    pix = first_page.get_pixmap(dpi=150)
                                    img_data = pix.tobytes("jpeg")
                                    
                                    from django.core.files.base import ContentFile
                                    self.object.thumbnail.save(
                                        f'preview_{self.object.slug}.jpg', 
                                        ContentFile(img_data), 
                                        save=True
                                    )
                                    pdf_document.close()
                                else:
                                    from django.core.files.base import ContentFile
                                    self.object.thumbnail.save(
                                        f'preview_{self.object.slug}.jpg', 
                                        ContentFile(first_file.read()), 
                                        save=True
                                    )
                                os.unlink(tmp_preview_path)
                            except Exception as e2:
                                logger.warning(f"Не удалось создать превью из файла студента: {e2}")
                    
                    # Обрабатываем эталон для получения правильных ответов
                    reference_results = process_general_work(
                        file_path=tmp_ref_path,
                        original_filename=f"REFERENCE_{reference_file.name}"  # ← ВАЖНО!
                    )
                    os.unlink(tmp_ref_path)
                    
                    # Сохраняем эталон в базе данных
                    ArticleFile.objects.create(
                        article=self.object, 
                        file=reference_file,
                        title=f"📌 ЭТАЛОН - {reference_file.name}"
                    )
                else:
                    # Если эталон не загружен, делаем превью из первого файла студента
                    if files:
                        first_file = files[0]
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_preview:
                            for chunk in first_file.chunks():
                                tmp_preview.write(chunk)
                            tmp_preview_path = tmp_preview.name
                        
                        try:
                            if first_file.name.lower().endswith('.pdf'):
                                pdf_document = fitz.open(tmp_preview_path)
                                first_page = pdf_document[0]
                                pix = first_page.get_pixmap(dpi=150)
                                img_data = pix.tobytes("jpeg")
                                
                                from django.core.files.base import ContentFile
                                self.object.thumbnail.save(
                                    f'preview_{self.object.slug}.jpg', 
                                    ContentFile(img_data), 
                                    save=True
                                )
                                pdf_document.close()
                            else:
                                from django.core.files.base import ContentFile
                                self.object.thumbnail.save(
                                    f'preview_{self.object.slug}.jpg', 
                                    ContentFile(first_file.read()), 
                                    save=True
                                )
                            os.unlink(tmp_preview_path)
                        except Exception as e:
                            logger.warning(f"Не удалось создать превью: {e}")
                            
                # ИСПРАВЛЕНО: Обрабатываем каждый загруженный файл и сохраняем ВСЕ результаты
                all_students_data = {}
                
                for file in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        for chunk in file.chunks():
                            tmp_file.write(chunk)
                        tmp_path = tmp_file.name

                    # ✅ Передаем оригинальное имя файла
                    results = process_general_work(
                        file_path=tmp_path,
                        original_filename=file.name  # ← ВАЖНО!
                    )
                    os.unlink(tmp_path)
                    
                    # Сохраняем файл в базе данных
                    ArticleFile.objects.create(article=self.object, file=file)
                    
                    # ИСПРАВЛЕНО: Объединяем всех студентов из всех файлов
                    if results and 'students' in results:
                        for student_name, student_data in results['students'].items():
                            # Добавляем информацию о файле-источнике
                            student_data['source_file'] = file.name
                            if student_name in all_students_data:
                                i = 2
                                new_name = f"{student_name}_{i}"
                                while new_name in all_students_data:
                                    i += 1
                                    new_name = f"{student_name}_{i}"
                                student_name = new_name

                            all_students_data[student_name] = student_data
                
                # ИСПРАВЛЕНО: Сохраняем ПОЛНЫЙ master_report.json
                if all_students_data:
                    import json
                    from pathlib import Path
                    from datetime import datetime
                    
                    # Создаем директорию для результатов
                    results_dir = Path('media/recognition_results')
                    results_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Сохраняем полный отчет
                    master_report = {
                        "students": all_students_data,
                        "metadata": {
                            "article_id": self.object.id,
                            "article_slug": self.object.slug,
                            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "total_students": len(all_students_data),
                            "files_processed": len(files)
                        }
                    }
                    
                    report_path = results_dir / f'master_report_{self.object.slug}.json'
                    with open(report_path, 'w', encoding='utf-8') as f:
                        json.dump(master_report, f, ensure_ascii=False, indent=2)
                    
                    # ИСПРАВЛЕНО: Формируем HTML с правильным сравнением
                    if reference_results:
                        # Извлекаем эталонные данные
                        ref_students = reference_results.get('students', {})
                        ref_data = list(ref_students.values())[0] if ref_students else None
                        
                        if ref_data:
                            html_results = format_general_comparison_with_full_data(
                                reference_data=ref_data,
                                students_data=all_students_data
                            )
                        else:
                            html_results = format_general_work_results_full(all_students_data)
                    else:
                        html_results = format_general_work_results_full(all_students_data)
                    
                    # Сохраняем результаты в full_description
                    current_content = self.object.full_description or ''
                    self.object.full_description = f"{current_content}\n\n{html_results}"
                    self.object.save()
                    
                    messages.success(
                        self.request, 
                        f'✅ Обработано {len(files)} файлов, найдено {len(all_students_data)} студентов.'
                    )
                else:
                    messages.warning(self.request, 'Не удалось распознать данные студентов')
                
            except Exception as e:
                messages.error(self.request, f'Ошибка обработки: {str(e)}')
                import traceback
                traceback.print_exc()
        
        return response
    
    def get_initial(self):
        initial = super().get_initial()
        if self.request.GET.get('type') == 'general':
            initial['work_type'] = 'general'
        return initial


# ИСПРАВЛЕНО: Новая функция для полного форматирования результатов
def format_general_work_results_full(students_data: dict) -> str:
    """Форматирует ВСЕ результаты обработки общей работы в HTML"""
    if not students_data:
        return '<p>Нет результатов</p>'
    
    html = '''
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 100%; margin-top: 20px;">
        <h2 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">
            📊 Результаты обработки бланков
        </h2>
        <p style="color: #6c757d; margin-bottom: 20px;">
            <i class="fas fa-users"></i> Всего студентов: <strong>{total_students}</strong>
        </p>
    '''
    
    html = html.format(total_students=len(students_data))
    
    for student_name, student_data in students_data.items():
        html += f'''
        <div style="margin-bottom: 30px; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px; background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: #3498db; margin-top: 0; display: flex; align-items: center;">
                <span style="background-color: #3498db; color: white; width: 40px; height: 40px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin-right: 15px;">
                    👤
                </span>
                {student_data.get("name", student_name)}
            </h3>
        '''
        
        # Информация о файле-источнике
        if student_data.get('source_file'):
            html += f'''
            <p style="color: #6c757d; margin-bottom: 15px; font-size: 0.9em;">
                <i class="fas fa-file"></i> Файл: {student_data['source_file']}
            </p>
            '''
        
        # Время обработки
        if student_data.get('timestamp'):
            html += f'''
            <p style="color: #6c757d; margin-bottom: 15px; font-size: 0.9em;">
                <i class="fas fa-clock"></i> Обработано: {student_data['timestamp']}
            </p>
            '''
        
        # Печатный текст (полный)
        if student_data.get('printed_text'):
            html += f'''
            <div style="margin-bottom: 20px; background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
                <h4 style="color: #28a745; margin-top: 0; margin-bottom: 10px;">
                    <i class="fas fa-pencil-alt"></i> Письменное задание
                </h4>
                <p style="margin: 0; line-height: 1.6; font-size: 1.1em;">{student_data["printed_text"]}</p>
            </div>
            '''
        
        # Тестовый текст (для проверки почерка)
        if student_data.get('test_text'):
            html += f'''
            <div style="margin-bottom: 20px; background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8;">
                <h4 style="color: #17a2b8; margin-top: 0; margin-bottom: 10px;">
                    <i class="fas fa-font"></i> Тестовый текст (проверка почерка)
                </h4>
                <p style="margin: 0; line-height: 1.6;">{student_data["test_text"]}</p>
            </div>
            '''
        
        # Текстовые ответы
        if student_data.get('text_answers'):
            html += '''
            <div style="margin-bottom: 20px;">
                <h4 style="color: #fd7e14; margin-bottom: 10px;">
                    <i class="fas fa-tasks"></i> Задания с кратким ответом
                </h4>
                <table style="width: 100%; border-collapse: collapse; background-color: white;">
                    <thead>
                        <tr style="background-color: #e9ecef;">
                            <th style="padding: 10px; border: 1px solid #dee2e6; text-align: left;">№</th>
                            <th style="padding: 10px; border: 1px solid #dee2e6; text-align: left;">Ответ</th>
                        </tr>
                    </thead>
                    <tbody>
            '''
            for i in range(1, 11):
                row_key = f'row_{i}'
                answer = student_data['text_answers'].get(row_key, '—')
                html += f'''
                    <tr>
                        <td style="padding: 8px; border: 1px solid #dee2e6; width: 60px; font-weight: bold;">{i}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">{answer if answer else "—"}</td>
                    </tr>
                '''
            html += '</tbody></table></div>'
        
        # Тестовые задания
        if student_data.get('test_task'):
            html += '''
            <div style="margin-bottom: 20px;">
                <h4 style="color: #6f42c1; margin-bottom: 10px;">
                    <i class="fas fa-check-square"></i> Тестовые задания
                </h4>
                <table style="width: 100%; border-collapse: collapse; background-color: white;">
                    <thead>
                        <tr style="background-color: #e9ecef;">
                            <th style="padding: 10px; border: 1px solid #dee2e6; text-align: left;">Задание</th>
                            <th style="padding: 10px; border: 1px solid #dee2e6; text-align: left;">Выбранные варианты</th>
                        </tr>
                    </thead>
                    <tbody>
            '''
            for task_num in sorted(student_data['test_task'].keys(), key=int):
                task_data = student_data['test_task'][task_num]
                answers = task_data.get('answers_letters', [])
                answers_str = ', '.join(answers) if answers else '—'
                
                html += f'''
                    <tr>
                        <td style="padding: 8px; border: 1px solid #dee2e6; width: 80px; font-weight: bold;">{task_num}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">{answers_str}</td>
                    </tr>
                '''
            html += '</tbody></table></div>'
        
        html += '</div>'
    
    html += '</div>'
    return html

# Добавьте эту функцию для расчета общей статистики
def calculate_group_statistics(students_data: dict, reference_data: dict = None) -> dict:
    """Рассчитывает общую статистику по группе"""
    stats = {
        'total_students': len(students_data),
        'text_answers_stats': {
            'total_questions': 0,
            'total_correct': 0,
            'by_question': {i: {'correct': 0, 'total': 0} for i in range(1, 11)}
        },
        'test_tasks_stats': {
            'total_questions': 0,
            'total_correct': 0,
            'by_task': {i: {'correct': 0, 'total': 0} for i in range(1, 6)}
        },
        'student_scores': [],  # баллы каждого студента
        'best_student': None,
        'worst_student': None,
        'average_score': 0,
        'difficult_questions': [],  # самые сложные вопросы
        'easy_questions': []  # самые легкие вопросы
    }
    
    if not students_data or not reference_data:
        return stats
    
    student_scores = []
    
    for student_name, student_data in students_data.items():
        student_score = {'name': student_data.get('name', student_name), 'correct': 0, 'total': 0}
        
        # Анализ текстовых ответов
        if student_data.get('text_answers') and reference_data.get('text_answers'):
            for i in range(1, 11):
                row_key = f'row_{i}'
                student_answer = student_data['text_answers'].get(row_key, '').strip().upper()
                ref_answer = reference_data['text_answers'].get(row_key, '').strip().upper()
                
                if ref_answer:  # только если есть эталонный ответ
                    stats['text_answers_stats']['total_questions'] += 1
                    stats['text_answers_stats']['by_question'][i]['total'] += 1
                    student_score['total'] += 1
                    
                    if student_answer == ref_answer:
                        stats['text_answers_stats']['total_correct'] += 1
                        stats['text_answers_stats']['by_question'][i]['correct'] += 1
                        student_score['correct'] += 1
        
        # Анализ тестовых заданий
        if student_data.get('test_task') and reference_data.get('test_task'):
            for i in range(1, 6):
                task_key = str(i)
                student_task = student_data['test_task'].get(task_key, {})
                ref_task = reference_data.get('test_task', {}).get(task_key, {})
                
                student_answers = set(student_task.get('answers_letters', []))
                ref_answers = set(ref_task.get('answers_letters', []))
                
                if ref_answers:  # только если есть эталонный ответ
                    stats['test_tasks_stats']['total_questions'] += 1
                    stats['test_tasks_stats']['by_task'][i]['total'] += 1
                    student_score['total'] += 1
                    
                    if student_answers == ref_answers:
                        stats['test_tasks_stats']['total_correct'] += 1
                        stats['test_tasks_stats']['by_task'][i]['correct'] += 1
                        student_score['correct'] += 1
        
        student_scores.append(student_score)
    
    # Сортируем студентов по баллам
    if student_scores:
        student_scores.sort(key=lambda x: x['correct'], reverse=True)
        stats['student_scores'] = student_scores
        stats['best_student'] = student_scores[0]
        stats['worst_student'] = student_scores[-1]
        
        total_correct = sum(s['correct'] for s in student_scores)
        total_questions = sum(s['total'] for s in student_scores)
        stats['average_score'] = (total_correct / len(student_scores)) if student_scores else 0
        
        # Находим самые сложные вопросы (меньше всего правильных ответов)
        all_questions = []
        for i in range(1, 11):
            q_data = stats['text_answers_stats']['by_question'][i]
            if q_data['total'] > 0:
                accuracy = (q_data['correct'] / q_data['total']) * 100
                all_questions.append({
                    'type': 'text',
                    'number': i,
                    'accuracy': accuracy,
                    'correct': q_data['correct'],
                    'total': q_data['total']
                })
        
        for i in range(1, 6):
            q_data = stats['test_tasks_stats']['by_task'][i]
            if q_data['total'] > 0:
                accuracy = (q_data['correct'] / q_data['total']) * 100
                all_questions.append({
                    'type': 'test',
                    'number': i,
                    'accuracy': accuracy,
                    'correct': q_data['correct'],
                    'total': q_data['total']
                })
        
        # Сортируем по точности
        all_questions.sort(key=lambda x: x['accuracy'])
        stats['difficult_questions'] = all_questions[:3]  # 3 самых сложных
        stats['easy_questions'] = all_questions[-3:]  # 3 самых легких
    
    return stats


# ИСПРАВЛЕНО: Функция сравнения с эталоном (сохраняет ВСЕ данные)
def format_general_comparison_with_full_data(reference_data: dict, students_data: dict) -> str:
    """Форматирует сравнение результатов с эталоном, показывая ВСЕ данные"""
    if not students_data:
        return '<p>Нет данных для сравнения</p>'
    
    html = '''
    <div style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 100%; margin-top: 20px;">
        <h2 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">
            📊 Сравнение с эталоном
        </h2>
        
        <div style="background-color: #e7f3ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <h3 style="color: #0056b3; margin-top: 0;">
                <i class="fas fa-star"></i> Эталонные ответы
            </h3>
            <p><strong>Эталон:</strong> {reference_name}</p>
    '''.format(reference_name=reference_data.get('name', 'Не указано'))
    
    # Показываем эталонные ответы
    if reference_data.get('text_answers'):
        html += '<div style="margin-top: 10px;"><strong>Правильные ответы:</strong><br>'
        for i in range(1, 11):
            row_key = f'row_{i}'
            answer = reference_data['text_answers'].get(row_key, '—')
            html += f'{i}: {answer}; '
        html += '</div>'
    
    if reference_data.get('test_task'):
        html += '<div style="margin-top: 10px;"><strong>Правильные тестовые задания:</strong><br>'
        for task_num in sorted(reference_data['test_task'].keys(), key=int):
            task_data = reference_data['test_task'][task_num]
            answers = task_data.get('answers_letters', [])
            html += f'Задание {task_num}: {", ".join(answers) if answers else "—"}<br>'
        html += '</div>'
    
    html += '</div>'
    
    # Статистика
    total_students = len(students_data)
    html += f'<p style="color: #6c757d; margin-bottom: 20px;"><i class="fas fa-users"></i> Всего студентов: <strong>{total_students}</strong></p>'
    
    # Данные каждого студента с сравнением
    for student_name, student_data in students_data.items():
        html += f'''
        <div style="margin-bottom: 30px; border: 1px solid #dee2e6; border-radius: 10px; padding: 20px; background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="color: #3498db; margin-top: 0; display: flex; align-items: center;">
                <span style="background-color: #3498db; color: white; width: 40px; height: 40px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin-right: 15px;">
                    👤
                </span>
                {student_data.get("name", student_name)}
            </h3>
        '''
        
        # Информация о файле
        if student_data.get('source_file'):
            html += f'''
            <p style="color: #6c757d; margin-bottom: 15px;">
                <i class="fas fa-file"></i> Файл: {student_data['source_file']}
            </p>
            '''
        
        # Счетчики для статистики
        correct_text_answers = 0
        total_text_answers = 0
        correct_test_tasks = 0
        total_test_tasks = 0
        
        # Сравнение текстовых ответов
        if student_data.get('text_answers') and reference_data.get('text_answers'):
            html += '''
            <div style="margin-bottom: 20px;">
                <h4 style="color: #fd7e14; margin-bottom: 10px;">
                    <i class="fas fa-tasks"></i> Задания с кратким ответом
                </h4>
                <table style="width: 100%; border-collapse: collapse; background-color: white;">
                    <thead>
                        <tr style="background-color: #e9ecef;">
                            <th style="padding: 10px; border: 1px solid #dee2e6;">№</th>
                            <th style="padding: 10px; border: 1px solid #dee2e6;">Ответ студента</th>
                            <th style="padding: 10px; border: 1px solid #dee2e6;">Правильный ответ</th>
                            <th style="padding: 10px; border: 1px solid #dee2e6; width: 60px;">Результат</th>
                        </tr>
                    </thead>
                    <tbody>
            '''
            
            for i in range(1, 11):
                row_key = f'row_{i}'
                student_answer = student_data['text_answers'].get(row_key, '').strip().upper()
                ref_answer = reference_data['text_answers'].get(row_key, '').strip().upper()
                
                is_correct = student_answer == ref_answer if ref_answer else None
                
                if ref_answer:  # Считаем только если есть эталонный ответ
                    total_text_answers += 1
                    if is_correct:
                        correct_text_answers += 1
                
                status_icon = '✓' if is_correct else ('✗' if is_correct is False else '—')
                status_color = '#28a745' if is_correct else ('#dc3545' if is_correct is False else '#6c757d')
                
                html += f'''
                    <tr>
                        <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">{i}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">{student_answer if student_answer else "—"}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">{ref_answer if ref_answer else "—"}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6; text-align: center;">
                            <span style="color: {status_color}; font-weight: bold; font-size: 18px;">{status_icon}</span>
                        </td>
                    </tr>
                '''
            html += '</tbody></table></div>'
        
        # Сравнение тестовых заданий
        if student_data.get('test_task') and reference_data.get('test_task'):
            html += '''
            <div style="margin-bottom: 20px;">
                <h4 style="color: #6f42c1; margin-bottom: 10px;">
                    <i class="fas fa-check-square"></i> Тестовые задания
                </h4>
                <table style="width: 100%; border-collapse: collapse; background-color: white;">
                    <thead>
                        <tr style="background-color: #e9ecef;">
                            <th style="padding: 10px; border: 1px solid #dee2e6;">Задание</th>
                            <th style="padding: 10px; border: 1px solid #dee2e6;">Ответ студента</th>
                            <th style="padding: 10px; border: 1px solid #dee2e6;">Правильный ответ</th>
                            <th style="padding: 10px; border: 1px solid #dee2e6; width: 60px;">Результат</th>
                        </tr>
                    </thead>
                    <tbody>
            '''
            
            for task_num in sorted(student_data['test_task'].keys(), key=int):
                student_task = student_data['test_task'].get(task_num, {})
                ref_task = reference_data.get('test_task', {}).get(task_num, {})
                
                student_answers = set(student_task.get('answers_letters', []))
                ref_answers = set(ref_task.get('answers_letters', []))
                
                is_correct = student_answers == ref_answers if ref_answers else None
                
                if ref_answers:  # Считаем только если есть эталонный ответ
                    total_test_tasks += 1
                    if is_correct:
                        correct_test_tasks += 1
                
                status_icon = '✓' if is_correct else ('✗' if is_correct is False else '—')
                status_color = '#28a745' if is_correct else ('#dc3545' if is_correct is False else '#6c757d')
                
                student_str = ', '.join(sorted(student_answers)) if student_answers else '—'
                ref_str = ', '.join(sorted(ref_answers)) if ref_answers else '—'
                
                html += f'''
                    <tr>
                        <td style="padding: 8px; border: 1px solid #dee2e6; font-weight: bold;">{task_num}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">{student_str}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6;">{ref_str}</td>
                        <td style="padding: 8px; border: 1px solid #dee2e6; text-align: center;">
                            <span style="color: {status_color}; font-weight: bold; font-size: 18px;">{status_icon}</span>
                        </td>
                    </tr>
                '''
            html += '</tbody></table></div>'
        
        # Печатный текст (без сравнения, просто показываем)
        if student_data.get('printed_text'):
            html += f'''
            <div style="margin-bottom: 20px; background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
                <h4 style="color: #28a745; margin-top: 0;">
                    <i class="fas fa-pencil-alt"></i> Письменное задание
                </h4>
                <p style="margin: 0;">{student_data["printed_text"]}</p>
            </div>
            '''
        
        # Статистика по студенту
        total_correct = correct_text_answers + correct_test_tasks
        total_questions = total_text_answers + total_test_tasks
        percent = (total_correct / total_questions * 100) if total_questions > 0 else 0
        
        html += f'''
        <div style="margin-top: 20px; padding-top: 15px; border-top: 2px solid #dee2e6; background-color: #f8f9fa; padding: 15px; border-radius: 8px;">
            <h4 style="margin-top: 0; color: #495057;">📈 Статистика</h4>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div>
                    <span style="color: #28a745;">✓ Правильно: <strong>{total_correct}</strong></span>
                </div>
                <div>
                    <span style="color: #dc3545;">✗ Ошибки: <strong>{total_questions - total_correct}</strong></span>
                </div>
                <div>
                    <span>📊 Всего вопросов: <strong>{total_questions}</strong></span>
                </div>
                <div>
                    <span style="font-size: 1.2em;">🎯 Точность: <strong style="color: { '#28a745' if percent >= 70 else '#ffc107' if percent >= 50 else '#dc3545' };">{percent:.1f}%</strong></span>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <div style="background-color: #e9ecef; height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="background-color: #28a745; height: 100%; width: {percent}%; transition: width 0.3s;"></div>
                </div>
            </div>
        </div>
        '''.replace('{percent}', str(percent))
        
        html += '</div>'
    
    return html