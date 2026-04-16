from django import forms
from django.core.validators import FileExtensionValidator
from .models import Article, Comment, ArticleFile, validate_file_size


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput(attrs={'multiple': True}))
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        if data is None:
            return []
        if isinstance(data, (list, tuple)):
            return [super(MultipleFileField, self).clean(d) for d in data]
        return [super(MultipleFileField, self).clean(data)]


class ArticleCreateForm(forms.ModelForm):
    # ОДИНАКОВЫЕ ПОЛЯ ДЛЯ ОБОИХ ТИПОВ
    files = MultipleFileField(
        label='Файлы с бланками (JPG/PDF)',
        required=False,
        validators=[FileExtensionValidator(allowed_extensions=('jpg', 'jpeg', 'pdf'))]
    )
    
    reference_file = forms.FileField(
        label='Эталон (правильные ответы)',
        required=False,
        validators=[FileExtensionValidator(allowed_extensions=('jpg', 'jpeg', 'pdf')), validate_file_size]
    )

    class Meta:
        model = Article
        fields = ('title', 'slug', 'category', 'short_description', 'full_description', 
                  'thumbnail', 'status', 'work_type')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Общие атрибуты для всех полей
        for field in self.fields:
            if field not in ('files', 'reference_file'):
                self.fields[field].widget.attrs.update({
                    'class': 'form-control',
                    'autocomplete': 'off'
                })

        # CKEditor поля
        self.fields['short_description'].widget.attrs.update({'class': 'form-control django_ckeditor_5'})
        self.fields['full_description'].widget.attrs.update({'class': 'form-control django_ckeditor_5'})
        
        # Файловые поля
        self.fields['files'].widget.attrs.update({'class': 'form-control-file'})
        self.fields['reference_file'].widget.attrs.update({'class': 'form-control-file'})
        self.fields['thumbnail'].widget.attrs.update({'class': 'form-control-file'})
        self.fields['work_type'].widget.attrs.update({'class': 'form-control'})


class ArticleUpdateForm(ArticleCreateForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'fixed' in self.fields:
            self.fields['fixed'].widget.attrs.update({'class': 'form-check-input'})

    class Meta(ArticleCreateForm.Meta):
        fields = ArticleCreateForm.Meta.fields + ('fixed', 'updater')


class CommentCreateForm(forms.ModelForm):
    parent = forms.IntegerField(widget=forms.HiddenInput, required=False)
    content = forms.CharField(label='', widget=forms.Textarea(attrs={
        'cols': 30,
        'rows': 5,
        'placeholder': 'Комментарий',
        'class': 'form-control'
    }))

    class Meta:
        model = Comment
        fields = ('content',)