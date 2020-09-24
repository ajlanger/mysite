from .fi import get_coin_data
from .evaluator import predict_price, get_x_axis, get_prediction_and_input_data
from .lstm_model_1 import load_model_efficient, get_price_series
import numpy as np
import datetime

class Predict():
    def __init__(self, coin, inter):
        self.lag   = 24
        self.seq   = 1
        self.coin  = coin
        self.inter = inter

    def get_prediction(self):
        # The coin model has to be one for which there actually is a model
        if self.coin.upper() not in ['EOS', 'LINK', 'XXBTZ', 'XXRPZ']:
            coin_model = 'XXBTZ'
        else:
            coin_model = self.coin

        self.load_model()
        prediction, self.advice, self.past_data = predict_price(self.model, self.lag, self.seq, self.inter, self.coin)
        prediction = prediction.tolist()[0][0]
        self.prediction = round(prediction, 4)
        return self.prediction

    def get_plot_objects(self):

        coin_data   = get_coin_data(coin=self.coin, interval=self.inter)
        series      = get_price_series(coin_data)
        prediction, X_set = get_prediction_and_input_data(series, self.model, self.coin, self.inter, self.lag, self.seq)

        x_axis = get_x_axis(series, self.lag, self.seq, self.inter)
        x_axis = [datetime.datetime.strftime(date, '%d-%m %H:%M') for date in x_axis]
        y_axis_prediction = ['null' for i in range(len(x_axis)-1)]
        x_set_list = X_set.tolist()
        y_axis_past_prices = list(map(lambda x: round(x, 4), x_set_list))

        y_axis_prediction.append(self.prediction)
        y_axis_past_prices.extend(['null'])

        return x_axis, y_axis_prediction, y_axis_past_prices

    def load_model(self):
        self.model = load_model_efficient(self.coin, self.lag, self.seq, self.inter)
