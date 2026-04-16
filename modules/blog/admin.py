from django import forms
from django.contrib import admin
from mptt.admin import DraggableMPTTAdmin
from django.utils.html import format_html
from .models import Category, Article, Comment, ArticleFile


class ArticleFileForm(forms.ModelForm):
    class Meta:
        model = ArticleFile
        fields = '__all__'


class ArticleFileInline(admin.TabularInline):
    model = ArticleFile
    extra = 1
    fields = ('file', 'title', 'file_link', 'created_at', 'is_active')
    readonly_fields = ('file_link', 'created_at')

    def file_link(self, obj):
        if obj.file:
            return format_html(
                '<a href="{}" target="_blank"><i class="fas {}"></i> Скачать</a>',
                obj.file.url,
                obj.get_file_icon()
            )
        return "-"

    file_link.short_description = "Файл"


@admin.register(Category)
class CategoryAdmin(DraggableMPTTAdmin):
    list_display = ('tree_actions', 'indented_title', 'id', 'title', 'slug')
    list_display_links = ('title', 'slug')
    prepopulated_fields = {'slug': ('title',)}


@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    inlines = [ArticleFileInline]
    list_display = ('title', 'author', 'category', 'work_type', 'status', 'files_count')
    list_filter = ('work_type', 'status', 'category')
    search_fields = ('title', 'author__username')

    def files_count(self, obj):
        return obj.files.count()
    files_count.short_description = 'Файлов'


@admin.register(Comment)
class CommentAdminPage(DraggableMPTTAdmin):
    list_display = ('tree_actions', 'indented_title', 'article', 'author', 'time_create', 'status')
    mptt_level_indent = 2
    list_display_links = ('article',)
    list_filter = ('time_create', 'time_update', 'author')
    list_editable = ('status',)


@admin.register(ArticleFile)
class ArticleFileAdmin(admin.ModelAdmin):
    list_display = ('title', 'article_link', 'file_type', 'created_at')
    list_filter = ('created_at', 'is_active')
    
    def article_link(self, obj):
        return format_html(
            '<a href="{}">{}</a>',
            obj.article.get_absolute_url(),
            obj.article.title
        )
    article_link.short_description = 'Статья'
    
    def file_type(self, obj):
        return obj.get_file_type()
    file_type.short_description = 'Тип файла'

