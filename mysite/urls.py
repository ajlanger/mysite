"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from pages import views as page_views
from dl_extractor import views as dle_views
from crypto_price_predictor.views import HomeView, get_data, ChartData, crypto_pp_view

urlpatterns = [
    path('admin/', admin.site.urls),

    path('', page_views.home_view, name='Home'),
    path('CV-html.html', page_views.cv_view, name='CV'),

    path('dl_extractor.html', dle_views.dle_view, name='dle_app'),

    path('crypto_pp', crypto_pp_view, name='cpp_app'),
    path(r'crypto_pp/api/data/', get_data, name='api-data'),
    path(r'crypto_pp/api/chart/data/', ChartData.as_view()),
]
