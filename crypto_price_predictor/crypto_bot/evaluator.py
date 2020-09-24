import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from fc import *
from .fi import get_coin_data
from .lstm_model_1 import get_price_series, load_model_efficient, get_actual_b_s_labels, get_b_s_preds, evaluate_b_s_preds
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras import initializers
import matplotlib as mpl
import matplotlib
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K
from matplotlib.dates import DateFormatter
import seaborn as sns
import datetime

#  █████   ██████ ████████ ██    ██  █████  ██          ██████  ██████  ███████ ██████
# ██   ██ ██         ██    ██    ██ ██   ██ ██          ██   ██ ██   ██ ██      ██   ██
# ███████ ██         ██    ██    ██ ███████ ██          ██████  ██████  █████   ██   ██
# ██   ██ ██         ██    ██    ██ ██   ██ ██          ██      ██   ██ ██      ██   ██
# ██   ██  ██████    ██     ██████  ██   ██ ███████     ██      ██   ██ ███████ ██████

def get_prediction_and_input_data(series, model, coin, interval, lag, seq):
    # Transform data
    X_set = np.array(series[-lag:], dtype='float32')
    scaler = MinMaxScaler(feature_range=(0,1))
    X_set_scaled = scaler.fit_transform(X_set.reshape(lag,1))
    X_set_reshaped = X_set_scaled.reshape(1, lag, 1)
    tensorfile = tf.convert_to_tensor(X_set_reshaped, dtype='float32', dtype_hint=None, name=None)

    # Predict
    prediction = model.predict(tensorfile)
    prediction = scaler.inverse_transform(prediction.reshape(1, seq))

    return prediction, X_set

# Get buy or sell advice
def get_advice(prediction, X_set):
    return 'b' if max(prediction)[0] > X_set[-1] else 's'


def get_x_axis(series, lag, seq, interval):
    interval_minutes = {'weekly':datetime.timedelta(weeks=1),
                        'daily':datetime.timedelta(days=1),
                        'hourly':datetime.timedelta(hours=1),
                        'half-hourly':datetime.timedelta(minutes=30),
                        'minutely':datetime.timedelta(minutes=1)}
    time = series.index[-lag:].tolist()
    future_times = []
    for n in [i+1 for i in range(seq)]:
        future_times.append(time[-1] + interval_minutes[interval]*n)
    time.extend(future_times)
    return time


def plot_final(model, interval, coin, lag, seq):
    coin_data = get_coin_data(coin=coin, interval=interval)
    series = lstm_model_1.get_price_series(coin_data)
    prediction, X_set = get_prediction_and_input_data(series, model, coin, interval, lag, seq)
    y_axis = np.append(X_set, prediction)
    x_axis = get_x_axis(series, lag, seq, interval)

    df = pd.DataFrame({'Time':x_axis[:30], 'Price':y_axis[:30]})
    df_p = pd.DataFrame({'Time':x_axis[-1:], 'Price':y_axis[-1:]})
    plt.figure(figsize=(12,5))
    plt.figtext(.5,.9, coin, fontsize=18, ha='center')
    plt.figtext(.5,.85, f'PredictionDate: {x_axis[-1]} Prediction: {round(y_axis[-1], 2)}',fontsize=13,ha='center')
    # plt.title(f'Startdate: {x_axis[0]} & PredictionDate: {x_axis[-1]} & Interval: {interval}')
    sns.lineplot(x='Time', y='Price', data=df, label='Input', color='blue', marker='o')
    sns.scatterplot(x='Time', y='Price', data=df_p, label='Predicted', marker='X', color='red')
    plt.xticks(x_axis, rotation=45)
    plt.legend()
    if interval in ['weekly', 'daily']:
        xformatter = DateFormatter('%d-%m')
    else:
        xformatter = DateFormatter('%H:%M')
    plt.gcf().axes[0].xaxis.set_major_formatter(xformatter)
    plt.show()


def predict_price(model, lag, seq, interval, coin):
    coin_data = get_coin_data(coin=coin, interval=interval)
    series = get_price_series(coin_data)

    # Make y_axis
    prediction, X_set = get_prediction_and_input_data(series, model, coin, interval, lag, seq)
    advice = get_advice(prediction, X_set)
    return prediction, advice, X_set

####################################################################################################

def load_models(interval):
    if interval == 'daily':
        m_link = lstm_model_1.load_model_efficient('LINK', 24, 1, 'daily')
        m_xxrpz = lstm_model_1.load_model_efficient('XXRPZ', 24, 1, 'daily')
        models = [(m_link, 24, 'daily'), (m_xxrpz, 24, 'daily')]
        model_names = ['m_link', 'm_xxrpz']
        return models, model_names

    if interval == 'hourly':
        m_eos = lstm_model_1.load_model_efficient('EOS', 30, 1, 'hourly')
        m_xxbtz = lstm_model_1.load_model_efficient('XXBTZ', 30, 1, 'hourly')
        models = [(m_eos, 30, 'hourly'), (m_xxbtz, 30, 'hourly')]
        model_names = ['m_eos', 'm_xxbtz']
        return models, model_names


def get_advices(coin_list, number=1, min_acc=0.6, interval='hourly', advice_kind='b'):
    """
    number = number of desired buy advices
    """
    models, model_names = load_models(interval)
    accuracies = {'model':[], 'coin':[], 'acc':[], 'lag':[], 'inter':[]}
    for coin in coin_list:
        for model, lag, inter in models:
            coin_data = get_coin_data(coin=coin, interval=inter)
            series = lstm_model_1.get_price_series(coin_data)
            n_test = int(0.15 * len(series))
            accuracy_list = []
            if len(series) > 150:
                # Get test data
                scaler_train, scaler_test, X_train, X_test, y_train, y_test = lstm_model_1.prepare_data(series, n_test, lag, 1)
                # X_test_dates, y_test_dates = lstm_model_1.get_dates(X_test, y_test, n_test, lag, 1)

                # Test model
                b_s_labels = lstm_model_1.get_actual_b_s_labels(y_test, X_test)
                predictions_bs = lstm_model_1.get_b_s_preds(model, X_test)

                # Calculate accuracy
                model_accuracy = lstm_model_1.evaluate_b_s_preds(predictions_bs, b_s_labels)

                accuracies['model'].append(model_names[models.index((model, lag, inter))])
                accuracies['coin'].append(coin)
                accuracies['acc'].append(model_accuracy)
                accuracies['lag'].append(lag)
                accuracies['inter'].append(inter)


    models_vs_coins = pd.DataFrame(accuracies)
    if len(models_vs_coins) < number:
        number = len(models_vs_coins)
    seq = 1
    sorted_frame = models_vs_coins.sort_values('acc', ascending=False).loc[models_vs_coins['inter']==interval].reset_index().drop(columns='index')
    advices = {}
    for i in range(number):
        model_name = sorted_frame.iloc[i]['model']
        coin = sorted_frame.iloc[i]['coin']
        interval = sorted_frame.iloc[i]['inter']
        lag = sorted_frame.iloc[i]['lag']
        acc = sorted_frame.iloc[i]['acc']
        predictor = models[model_names.index(model_name)][0]

        # Get buy/sell advice:
        prediction, advice = predict_price(predictor, lag, seq, interval, coin)
        rounded_pred = round(prediction[0][0], 2)
        rounded_acc = round(acc, 2)
        if advice == advice_kind:
            if coin in list(advices.keys()):
                if advices[coin]['acc'] < acc:
                    advices[coin] = {'predicted_price':rounded_pred, 'model':model_name, 'acc':rounded_acc, 'advice':advice}
            else:
                advices[coin] = {'predicted_price':rounded_pred, 'model':model_name, 'acc':rounded_acc, 'advice':advice}

    return advices

# For selling
# get_advices(['EOS'], number=5, min_acc=0.61, interval='hourly', advice_kind='s')

# For buying
# coins = get_all_possible_coins()
# get_advices(coins, number=5, min_acc=0.61, interval='hourly', advice_kind='b')

# get_processed_portfolio(16)
