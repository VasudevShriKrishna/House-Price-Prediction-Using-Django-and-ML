from django.contrib import admin
from django.urls import path 
from HousePricePrediction import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.home, name="home"),
    path("predict/", views.predict, name="predict"),
    path("result/", views.result, name="result"),
    path("history/", views.prediction_history, name="prediction_history"),
]
