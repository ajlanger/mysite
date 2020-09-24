# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .fi import get_coin_data
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
from tensorflow.keras.models import model_from_json
import tensorflow as tf
from keras import backend as K
from matplotlib.dates import DateFormatter
import seaborn as sns
import datetime


# ██████  ██████  ███████ ██████   █████  ██████  ███████     ██████   █████  ████████  █████
# ██   ██ ██   ██ ██      ██   ██ ██   ██ ██   ██ ██          ██   ██ ██   ██    ██    ██   ██
# ██████  ██████  █████   ██████  ███████ ██████  █████       ██   ██ ███████    ██    ███████
# ██      ██   ██ ██      ██      ██   ██ ██   ██ ██          ██   ██ ██   ██    ██    ██   ██
# ██      ██   ██ ███████ ██      ██   ██ ██   ██ ███████     ██████  ██   ██    ██    ██   ██

# Get a Series object with coin data
def get_price_series(coin_data):
	price_list = []
	datelist = []
	for el in coin_data[list(coin_data.keys())[0]]:
		datelist.append(datetime.datetime.strptime(el[0], '%d-%m-%y %H:%M'))
		price_list.append(el[1])
	return pd.Series(price_list, index=datelist, name='price')


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values.reshape(series.values.shape[0],1)
    train, test = raw_values[0:-n_test], raw_values[-n_test:]
    # rescale values to -1, 1
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    scaler_test = MinMaxScaler(feature_range=(0, 1))

    scaled_values_train = scaler_train.fit_transform(train)
    scaled_values_test = scaler_test.fit_transform(test)

    scaled_values_train = scaled_values_train.reshape(len(scaled_values_train), 1)
    scaled_values_test = scaled_values_test.reshape(len(scaled_values_test), 1)

    # transform into supervised learning problem X, y
    supervised_train = series_to_supervised(scaled_values_train, n_lag, n_seq)
    supervised_test = series_to_supervised(scaled_values_test, n_lag, n_seq)

    supervised_values_train = supervised_train.values
    supervised_values_test = supervised_test.values
    # split into train and test sets

    x_train = supervised_values_train[:, :n_lag]
    x_test = supervised_values_test[:, :n_lag]
    y_train = supervised_values_train[:, n_lag:]
    y_test = supervised_values_test[:, n_lag:]
    return scaler_train, scaler_test, x_train, x_test, y_train, y_test


#  ██████ ██████  ███████  █████  ████████ ███████     ███    ███  ██████  ██████  ███████ ██
# ██      ██   ██ ██      ██   ██    ██    ██          ████  ████ ██    ██ ██   ██ ██      ██
# ██      ██████  █████   ███████    ██    █████       ██ ████ ██ ██    ██ ██   ██ █████   ██
# ██      ██   ██ ██      ██   ██    ██    ██          ██  ██  ██ ██    ██ ██   ██ ██      ██
#  ██████ ██   ██ ███████ ██   ██    ██    ███████     ██      ██  ██████  ██████  ███████ ███████


def get_model(X_train, y_train, n_seq=1, units_1=100, units_2=40, epochs=100, verbose=0, x_val=None, y_val=None):
    model = keras.Sequential(
        [
            keras.layers.LSTM(units = units_1, return_sequences = True, input_shape = (X_train.shape[1], 1), kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()),
            keras.layers.Dropout(0.3),
            keras.layers.LSTM(units = units_1, return_sequences = True, kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(units = units_1, return_sequences = True, kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(units = units_2, kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units = n_seq, kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros())
        ]
    )

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    if x_val != None and y_val != None:
        model.fit(X_train, y_train, epochs = epochs, batch_size = 32, validation_data=(x_val, y_val), verbose=verbose)
    elif x_val == None and y_val == None:
        model.fit(X_train, y_train, epochs = epochs, batch_size = 32, verbose=verbose)
    return model


def save_model(model, model_name):
    """
    Save model
    """
    model.save_weights(f'{model_name}.h5')
    model_json = model.to_json()
    with open(f'{model_name}.json', "w") as json_file:
        json_file.write(model_json)
    json_file.close()


def load_model_efficient(coin, lag, seq, inter):
	# Loading model
	try:
		if coin in ['EOS', 'LINK', 'XXBTZ', 'XXRPZ']:
			model_coin = coin
		else:
			model_coin = 'XXBTZ'
		model_name = f'crypto_price_predictor/crypto_bot/keras_models/LSTM_{model_coin}_L{lag}_S{seq}_I{inter}'
		json_file = open(f'{model_name}.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		model.load_weights(f'{model_name}.h5')
		return model
	except:
	    sys.exit('No model with these options could be found.')


# ███████ ██    ██  █████  ██      ██    ██  █████  ████████ ███████
# ██      ██    ██ ██   ██ ██      ██    ██ ██   ██    ██    ██
# █████   ██    ██ ███████ ██      ██    ██ ███████    ██    █████
# ██       ██  ██  ██   ██ ██      ██    ██ ██   ██    ██    ██
# ███████   ████   ██   ██ ███████  ██████  ██   ██    ██    ███████

def get_dates(X_test, y_test, n_test, n_lag, n_seq, series):
    X_test_dates = []
    y_test_dates = []
    y_test_reshaped = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    for i in range(len(X_test_reshaped)):
        # For X
        original_values = list(np.round(series[-n_test+i:-n_test+i+n_lag].values, 2))
        processed_values = [round(element[0], 2) for element in scaler_test.inverse_transform(X_test_reshaped[i])]

        if len(original_values) == len(processed_values):
            dates = series[-n_test+i:-n_test+i+n_lag].index
            X_test_dates.append(dates)
        else:
            print(original_values)
            print('***')
            print(processed_values)
            print('----#-----#------#-')
            sys.exit('Error')

        # For y
        end = -n_test+i+n_lag+n_seq
        if end == 0:
            end = len(series)
        original_values = list(np.round(series[-n_test+i+n_lag:end].values,2))
        processed_values = [round(element[0], 2) for element in scaler_test.inverse_transform(y_test_reshaped[i])]

        if len(original_values) == len(processed_values):
            dates = series[-n_test+i+n_lag:end].index
            y_test_dates.append(dates)
    return X_test_dates, y_test_dates


def get_actual_b_s_labels(y_test, X_test):
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    b_s_labels = []
    for i in range(len(y_test)):
        if y_test[i] > X_test_reshaped[i][-1]:
            b_s_labels.append('b')
        else:
            b_s_labels.append('s')
    return b_s_labels


def plot(n_plots, X_test_converted, y_test_converted, predicted_test_converted, X_test, X_test_dates, y_test_dates):
    times = n_plots
    # random_numbers = [random.randint(0, len(X_test)) for i in range(times)]
    random_numbers = [-3, -2, -1]
    for n in range(n_plots):
        plt.figure(figsize=(12, 4))
        y_values = np.append(X_test_converted[n], y_test_converted[n])
        x_values = np.append(X_test_dates[n], y_test_dates[n])
        plt.plot(X_test_dates[n], X_test_converted[n], label='Inputs', marker='o')
        plt.plot(y_test_dates[n], y_test_converted[n], label='Expected', marker='*', c='#2ca02c')
        plt.plot(y_test_dates[n], predicted_test_converted[n], label='Predicted', marker='X', c='#2ca02c')
        plt.xticks(x_values, rotation='vertical')
        plt.legend()
        plt.show()
        IPython.display.clear_output()


def get_b_s_preds(model, X):
    # Returns the predicted list of desired buy or sell actions

    # 0. Reshape inputs
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    tensorfile = tf.convert_to_tensor(X, dtype=None, dtype_hint=None, name=None)

    # 1. Make prediction of prices
    predicted_prices = model.predict(X)

    # 2. Generate the labels b/s
    labels = []
    for i in range(len(predicted_prices)):
        if predicted_prices[i] > X[i][-1]:
            labels.append('b')
        else:
            labels.append('s')
    # 3. Return labels
    return labels


def evaluate_b_s_preds(predictions, actuals):
    correct = 0
    incorrect = 0
    for i in range(len(predictions)):
        if predictions[i] == actuals[i]:
            correct += 1
        else:
            incorrect += 1

    return correct/(correct+incorrect)
