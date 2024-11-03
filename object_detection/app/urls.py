from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('add_image/', views.add_image, name='add_image'),
    path('process_image/<int:image_id>/', views.process_image, name='process_image'),
    path('delete_image/<int:image_id>/', views.delete_image, name='delete_image'),
]
