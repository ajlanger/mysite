import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fc import *
from fi import *
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras import initializers
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib
import random
import sys
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K
from matplotlib.dates import DateFormatter
import seaborn as sns
import lstm_model_1

# ███    ███  ██████  ██████  ███████ ██          ███████ ███████  █████  ██████   ██████ ██   ██
# ████  ████ ██    ██ ██   ██ ██      ██          ██      ██      ██   ██ ██   ██ ██      ██   ██
# ██ ████ ██ ██    ██ ██   ██ █████   ██          ███████ █████   ███████ ██████  ██      ███████
# ██  ██  ██ ██    ██ ██   ██ ██      ██               ██ ██      ██   ██ ██   ██ ██      ██   ██
# ██      ██  ██████  ██████  ███████ ███████     ███████ ███████ ██   ██ ██   ██  ██████ ██   ██

interesting_coins = ['XXBTZ', 'ETH', 'EOS', 'XXRPZ', 'BCH', 'LINK']
intervals = ['hourly', 'daily']

for coin in interesting_coins:
    for lag in n_lag_list:
        for inter in intervals:
            # Initialize
            units_1 = 150
            units_2 = 75

            epochs = 800
            n_lag = lag
            n_seq = 1
            coin = coin
            coin_data = get_coin_data(coin=coin, interval=inter)
            series = lstm_model_1.get_price_series(coin_data)

            print(f'{units_1} units_1; {units_2} units_2; {epochs} epochs')
            # 2. Prepare data
            print('Preparing data...')
            scaler_train, scaler_test, X_train, X_test, y_train, y_test = lstm_model_1.prepare_data(series, n_test, n_lag, n_seq)

            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # 3. Train model
            print('Training model...')
            model = lstm_model_1.get_model(n_seq=n_seq, units_1=units_1, units_2=units_2, epochs=epochs, X_train=X_train, y_train=y_train)


            # 4. Test model
            print('Testing model...')
            X_test_dates, y_test_dates = lstm_model_1.get_dates(X_test, y_test, n_test, n_lag, n_seq)
            predicted_test = model.predict(X_test)

            X_test_converted = scaler_test.inverse_transform(X_test.reshape(X_test.shape[0], X_test.shape[1]))
            y_test_converted = scaler_test.inverse_transform(y_test)
            predicted_test_converted = scaler_test.inverse_transform(predicted_test)

            # 5. Visualisation
            print('Visualisation...')
            model_name = f'LSTM_{coin}_l{n_lag}s_{n_seq}_i{inter}'
            print(f'Model Name: {model_name}')
            plot(1, X_test_converted, y_test_converted, predicted_test_converted, X_test, X_test_dates,
            y_test_dates)

            # Save model
            print('Saving model...')
            model.save(f'keras_models/{model_name}')
            del model

#  ██████  ██████  ██ ███    ██     ███████ ██    ██  █████  ██
# ██      ██    ██ ██ ████   ██     ██      ██    ██ ██   ██ ██
# ██      ██    ██ ██ ██ ██  ██     █████   ██    ██ ███████ ██
# ██      ██    ██ ██ ██  ██ ██     ██       ██  ██  ██   ██ ██
#  ██████  ██████  ██ ██   ████     ███████   ████   ██   ██ ███████

"""
Conclusion:
    - best coins to take are ['XXBTZ', 'ETH', 'EOS', 'XXRPZ', 'BCH', 'LINK']
    - Best intervals: Daily and hourly
    - Best epochs: 800
    - Best lag: [15, 24, 30]
    - Best seq: 1
"""

coin_list = get_all_possible_coins()
m_eos = lstm_model_1.load_model_efficient('EOS', 30, 1, 'hourly')
m_link = lstm_model_1.load_model_efficient('LINK', 24, 1, 'daily')
m_xxbtz = lstm_model_1.load_model_efficient('XXBTZ', 30, 1, 'hourly')
m_xxrpz = lstm_model_1.load_model_efficient('XXRPZ', 24, 1, 'daily')
models = [(m_eos, 30, 'hourly'), (m_link, 24, 'daily'), (m_xxbtz, 30, 'hourly'), (m_xxrpz, 24, 'daily')]
model_names = ['m_eos', 'm_link', 'm_xxbtz', 'm_xxrpz']

accuracies = {'model':[], 'coin':[], 'acc':[], 'lag':[], 'inter':[]}
for coin in coin_list:
    for model, lag, inter in models:
        print('*------*------*------*-------*')
        print(coin)
        coin_data = get_coin_data(coin=coin, interval=inter)
        series = lstm_model_1.get_price_series(coin_data)
        n_test = int(0.15 * len(series))
        accuracy_list = []
        print(len(series))
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

models_vs_coins.pivot_table(values=['acc'],index=['lag', 'inter'],aggfunc='mean').plot.bar(figsize=(12, 4))

models_vs_coins.pivot_table(values=['acc'],index=['lag', 'inter'],aggfunc='mean')
max_acc = models_vs_coins['acc'].loc[models_vs_coins['inter']=='hourly'].argmax()

model_name = models_vs_coins.iloc[max_acc]['model']
coin = models_vs_coins.iloc[max_acc]['coin']
interval = models_vs_coins.iloc[max_acc]['inter']
lag = models_vs_coins.iloc[max_acc]['lag']
predictor = models[model_names.index(model_name)]
model_name
coin
interval
lag


coindata = get_coin_data(coin=coin, interval=interval)
coinprices = [price[1] for price in coindata[coin]]
sns.lineplot(x=range(len(coinprices)), y=coinprices)


# ███    ███  ██████  ██████  ███████ ██      ███████     ███████ ██    ██  █████  ██
# ████  ████ ██    ██ ██   ██ ██      ██      ██          ██      ██    ██ ██   ██ ██
# ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████     █████   ██    ██ ███████ ██
# ██  ██  ██ ██    ██ ██   ██ ██      ██           ██     ██       ██  ██  ██   ██ ██
# ██      ██  ██████  ██████  ███████ ███████ ███████     ███████   ████   ██   ██ ███████

"""
Best: XXBTZ, INTER=HOURLY, LAG=30
"""

# Getting the scores
coin_scores = {'coin': [], 'lag': [], 'inter': [], 'score': []}
for lag in n_lag_list:
    for coin in interesting_coins:
        for inter in intervals:
            coin_data = get_coin_data(coin=coin, interval=inter)
            series = lstm_model_1.get_price_series(coin_data)
            scaler_train, scaler_test, X_train, X_test, y_train, y_test = lstm_model_1.prepare_data(series, n_test, lag, n_seq)

            X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            model_name = f'keras_models/LSTM_{coin}_l{lag}s_{n_seq}_i{inter}'
            K.clear_session()
            model = lstm_model_1.load_model_efficient(coin, lag, n_seq, inter)

            b_s_labels = lstm_model_1.get_actual_b_s_labels(y_test, X_test)

            tensorfile = tf.convert_to_tensor(X_test_reshaped, dtype=None, dtype_hint=None, name=None)
            predictions = lstm_model_1.get_b_s_preds(model, tensorfile)

            score = lstm_model_1.evaluate_b_s_preds(predictions, b_s_labels)

            coin_scores['coin'].append(coin)
            coin_scores['lag'].append(lag)
            coin_scores['inter'].append(inter)
            coin_scores['score'].append(score)

score_df = pd.DataFrame(coin_scores)

score_df.pivot_table(values=['score'],index=['coin'],aggfunc='mean').plot.bar()
score_df.pivot_table(values=['score'],index=['inter'],aggfunc='mean').plot.bar()
score_df.pivot_table(values=['score'],index=['lag'],aggfunc='mean').plot.bar()
score_df.pivot_table(values=['score'],index=['coin','inter'],aggfunc='mean').plot.bar()
