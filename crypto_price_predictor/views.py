from django.http import JsonResponse
from django.shortcuts import render
from django.views.generic import View

from rest_framework.views import APIView
from rest_framework.response import Response

from .crypto_bot.crypto_predict import Predict
from .models import cryptoData

import random

# Create your views here.
def crypto_pp_view(request, *args, **kwargs):
    interval   = ""
    coin_choice = ""

    input_objects = cryptoData(request.POST or None)

    if input_objects.is_valid():
        interval    = input_objects.cleaned_data.get("interval")
        coin_choice = input_objects.cleaned_data.get("coinchoice")

        # Predict
        predi = Predict(coin_choice, interval)
        prediction = round(predi.get_prediction(), 2)
        advice = predi.advice
        x_axis, y_axis_prediction, y_axis_past_prices = predi.get_plot_objects()

        context = {'input_objects': input_objects, 'x_axis': x_axis, 'y_axis_prediction': y_axis_prediction, 'y_axis_past_prices': y_axis_past_prices, 'coin_choice':coin_choice, 'prediction': prediction}


        # Send data to chart view
        return render(request, 'crypto_pp.html', context)
    else:
        context = {'input_objects': input_objects}
        return render(request, 'crypto_pp.html', context)


class HomeView(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'crypto_pp.html', {})


def get_data(request, *args, **kwargs):
    data = {
        'sales': 100,
        'customers': 10,
    }
    return JsonResponse(data)


class ChartData(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request, format=None, coin='EOS', interval='hourly'):
        predi = Predict(coin, interval)
        prediction = predi.get_prediction()
        advice = predi.advice
        x_axis, y_axis_prediction, y_axis_past_prices = predi.get_plot_objects()

        data = {
                    'x_axis': x_axis,
                    'y_past_prices': y_axis_past_prices,
                    'y_prediction' : y_axis_prediction,
        }
        return Response(data)
