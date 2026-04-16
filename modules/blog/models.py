from django.db import models
from django.core.validators import FileExtensionValidator
from django.contrib.auth import get_user_model
from django.urls import reverse
from taggit.managers import TaggableManager
from django_ckeditor_5.fields import CKEditor5Field

from mptt.models import MPTTModel, TreeForeignKey

from modules.services.utils import unique_slugify
import os
from django.core.exceptions import ValidationError

User = get_user_model()

import os
from django.db import models
from django.core.validators import FileExtensionValidator
from django.core.exceptions import ValidationError
from .image_processor import ImageProcessor
import logging

logger = logging.getLogger(__name__)

def validate_file_size(value):
    filesize = value.size
    if filesize > 10 * 1024 * 1024:
        raise ValidationError("Максимальный размер файла 10MB")

class Article(models.Model):
    """
    Модель постов для сайта
    """    

    class ArticleManager(models.Manager):
        """
        Кастомный менеджер для модели статей
        """

        def all(self):
            """
            Список статей (SQL запрос с фильтрацией для страницы списка статей)
            """
            return self.get_queryset().select_related('author', 'category').filter(status='published')

        def detail(self):
            """
            Детальная статья (SQL запрос с фильтрацией для страницы со статьёй)
            """
            return self.get_queryset()\
                .select_related('author', 'category')\
                .prefetch_related('comments', 'comments__author', 'comments__author__profile', 'tags')\
                .filter(status='published')

    STATUS_OPTIONS = (
        ('published', 'Опубликовано'), 
        ('draft', 'Черновик')
    )
    
    WORK_TYPE_OPTIONS = (
        ('test', 'Тестовая работа'),
        ('general', 'Общая работа'),
    )

    title = models.CharField(verbose_name='Заголовок', max_length=255)
    slug = models.SlugField(verbose_name='URL', max_length=255, blank=True, unique=True)
    short_description = CKEditor5Field(max_length=500, verbose_name='Краткое описание', config_name='extends')
    full_description = CKEditor5Field(verbose_name='Полное описание', config_name='extends')
    thumbnail = models.ImageField(
        verbose_name='Эталон (для тестов)', 
        blank=True, 
        upload_to='images/thumbnails/%Y/%m/%d/', 
        validators=[FileExtensionValidator(allowed_extensions=('png', 'jpg', 'webp', 'jpeg', 'gif'))]
    )
    status = models.CharField(choices=STATUS_OPTIONS, default='published', verbose_name='Статус поста', max_length=10)
    work_type = models.CharField(choices=WORK_TYPE_OPTIONS, default='test', verbose_name='Тип работы', max_length=10)
    time_create = models.DateTimeField(auto_now_add=True, verbose_name='Время добавления')
    time_update = models.DateTimeField(auto_now=True, verbose_name='Время обновления')
    author = models.ForeignKey(to=User, verbose_name='Автор', on_delete=models.SET_DEFAULT, related_name='author_posts', default=1)
    updater = models.ForeignKey(to=User, verbose_name='Обновил', on_delete=models.SET_NULL, null=True, related_name='updater_posts', blank=True)
    fixed = models.BooleanField(verbose_name='Зафиксировано', default=False)
    category = TreeForeignKey('Category', on_delete=models.PROTECT, related_name='articles', verbose_name='Категория')
    objects = ArticleManager()
    tags = TaggableManager()

    class Meta:
        db_table = 'app_articles'
        ordering = ['-fixed', '-time_create']
        indexes = [models.Index(fields=['-fixed', '-time_create', 'status'])]
        verbose_name = 'Статья'
        verbose_name_plural = 'Статьи'

    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse('articles_detail', kwargs={'slug': self.slug})

    def save(self, *args, **kwargs):
        """
        Сохранение полей модели при их отсутствии заполнения
        """
        if not self.slug:
            self.slug = unique_slugify(self, self.title)
        super().save(*args, **kwargs)


class Category(MPTTModel):
    """
    Модель категорий с вложенностью
    """
    title = models.CharField(max_length=255, verbose_name='Название категории')
    slug = models.SlugField(max_length=255, verbose_name='URL категории', blank=True)
    description = models.TextField(verbose_name='Описание категории', max_length=300)
    parent = TreeForeignKey(
        'self',
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        db_index=True,
        related_name='children',
        verbose_name='Родительская категория'
    )

    class MPTTMeta:
        """
        Сортировка по вложенности
        """
        order_insertion_by = ('title',)

    class Meta:
        """
        Сортировка, название модели в админ панели, таблица в данными
        """
        verbose_name = 'Категория'
        verbose_name_plural = 'Категории'
        db_table = 'app_categories'

    def __str__(self):
        """
        Возвращение заголовка статьи
        """
        return self.title
    
    def get_absolute_url(self):
        return reverse('articles_by_category', kwargs={'slug': self.slug})


class Comment(MPTTModel):
    """
    Модель древовидных комментариев
    """

    STATUS_OPTIONS = (
        ('published', 'Опубликовано'),
        ('draft', 'Черновик')
    )

    article = models.ForeignKey(Article, on_delete=models.CASCADE, verbose_name='Статья', related_name='comments')
    author = models.ForeignKey(User, verbose_name='Автор комментария', on_delete=models.CASCADE, related_name='comments_author')
    content = models.TextField(verbose_name='Текст комментария', max_length=3000)
    time_create = models.DateTimeField(verbose_name='Время добавления', auto_now_add=True)
    time_update = models.DateTimeField(verbose_name='Время обновления', auto_now=True)
    status = models.CharField(choices=STATUS_OPTIONS, default='published', verbose_name='Статус поста', max_length=10)
    parent = TreeForeignKey('self', verbose_name='Родительский комментарий', null=True, blank=True, related_name='children', on_delete=models.CASCADE)

    class MTTMeta:
        order_insertion_by = ('-time_create',)

    class Meta:
        db_table = 'app_comments'
        indexes = [models.Index(fields=['-time_create', 'time_update', 'status', 'parent'])]
        ordering = ['-time_create']
        verbose_name = 'Комментарий'
        verbose_name_plural = 'Комментарии'

    def __str__(self):
        return f'{self.author}:{self.content}'


class ArticleFile(models.Model):
    article = models.ForeignKey('Article', on_delete=models.CASCADE, related_name='files')
    file = models.FileField(
        verbose_name='Файл',
        upload_to='articles/files/%Y/%m/%d/',
        validators=[
            FileExtensionValidator(allowed_extensions=('jpg', 'jpeg', 'pdf')),
            validate_file_size
        ]
    )
    title = models.CharField(verbose_name='Название файла', max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Дата загрузки')
    is_active = models.BooleanField(default=True, verbose_name='Активный')

    def save(self, *args, **kwargs):
        if not self.title:
            self.title = os.path.basename(self.file.name)
        super().save(*args, **kwargs)
        
        # Обработка изображения после сохранения
        if self.file.name.lower().endswith(('.jpg', '.jpeg')):
            self.process_image()


    def process_image(self):
        try:
            processor = ImageProcessor()
            
            # 1. Обрабатываем текущее изображение
            current_results = processor.process_uploaded_image(self.file.path)
            if 'error' in current_results:
                raise ValueError(current_results['error'])
            
            # 2. Получаем и обрабатываем эталон (превью статьи)
            reference_results = None
            if self.article.thumbnail:
                try:

                    reference_results = processor.process_uploaded_image(self.article.thumbnail.path)
                    print("!!!!!!!!", reference_results['raw_data'])
                    if 'error' in reference_results:
                        logger.warning(f"Ошибка обработки эталонного изображения: {reference_results['error']}")
                        reference_results = None
                except Exception as e:
                    logger.warning(f"Не удалось обработать эталонное изображение: {str(e)}")
            
            # 3. Формируем результаты
            if reference_results:
                comparison_data = processor.compare_with_reference(
                    reference_results, 
                    current_results
                )
                results_text = processor.format_comparison_results(comparison_data)
            else:
                results_text = current_results.get('formatted_html', '')
            
            # 4. Обновляем описание статьи
            content = self.article.full_description
            if "## 📊 Результаты проверки" in content or "## 📊 Результаты обработки шаблонов" in content:
                # Удаляем старые результаты если есть
                content = content.split("## 📊")[0].strip()
            
            self.article.full_description = f"{content}\n\n{results_text}"
            self.article.save()
            
        except Exception as e:
            logger.error(f"🛑 Ошибка обработки: {str(e)}")
            raise
    
    
    
    def get_file_type(self):
        ext = os.path.splitext(self.file.name)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            return 'Изображение'
        elif ext == '.pdf':
            return 'PDF'
        return 'Другой'

    def __str__(self):
        return self.title or os.path.basename(self.file.name)
    
