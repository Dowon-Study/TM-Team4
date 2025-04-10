from django.urls import path
from . import main

urlpatterns = [
    path('news/', main.get_news_by_date, name='get_news_by_date'),
]
