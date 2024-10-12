from django.urls import path
from . import views
from cxr_app import views

urlpatterns = [
    path('', views.upload_xray, name='upload_xray'),
    path('results/', views.results, name='results'),  # Add this if you create a separate results view
    path('home/', views.home, name='home'),  # Adjust as necessary
]
