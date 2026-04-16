from django.contrib.auth.mixins import AccessMixin, UserPassesTestMixin
from django.contrib import messages
from django.shortcuts import redirect
from django.core.exceptions import PermissionDenied


class AuthorRequiredMixin(AccessMixin):

    def dispatch(self, request, *args, **kwargs):
        if not request.user.is_authenticated:
            return self.handle_no_permission()
        
        obj = self.get_object()
        
        # Разрешаем доступ если:
        # 1. Пользователь - автор статьи
        # 2. ИЛИ пользователь - администратор (is_staff или is_superuser)
        if request.user == obj.author or request.user.is_staff or request.user.is_superuser:
            return super().dispatch(request, *args, **kwargs)
        
        # Иначе - отказываем
        messages.info(request, 'Изменение и удаление статьи доступно только автору')
        return redirect('home')


class UserIsNotAuthenticated(UserPassesTestMixin):
    def test_func(self):
        if self.request.user.is_authenticated:
            messages.info(self.request, 'Вы уже авторизованы. Вы не можете посетить эту страницу.')
            raise PermissionDenied
        return True
        
    def handle_no_permission(self):
        return redirect('home')