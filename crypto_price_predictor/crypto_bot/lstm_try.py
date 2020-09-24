# Imports
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics import mean_squared_error

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

from fc import *
from fi import *
import importlib
import fi
importlib.reload(fi)


# ███████ ██    ██ ███    ██  ██████ ████████ ██  ██████  ███    ██ ███████
# ██      ██    ██ ████   ██ ██         ██    ██ ██    ██ ████   ██ ██
# █████   ██    ██ ██ ██  ██ ██         ██    ██ ██    ██ ██ ██  ██ ███████
# ██      ██    ██ ██  ██ ██ ██         ██    ██ ██    ██ ██  ██ ██      ██
# ██       ██████  ██   ████  ██████    ██    ██  ██████  ██   ████ ███████

# Get a Series object with coin data
def get_price_series(coin_data):
	price_list = []
	datelist = []
	for el in coin_data[list(coin_data.keys())[0]]:
		datelist.append(datetime.datetime.strptime(el[0], '%d-%m-%y %H:%M'))
		price_list.append(el[1])
	return pd.Series(price_list, index=datelist, name='price')


# Scale data (normalize_data)
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
coin = 'XBT'
coin_data = get_coin_data(coin=coin, interval='hourly')
series = get_price_series(coin_data)
series_frame = series.to_frame()

plot_features = series_frame['price']
plot_features.plot(subplots=True)

plot_features = series_frame['price'][500:]
plot_features.plot(subplots=True)

series_frame.describe()
series.plot()

# Normalize the data
data_normalized, scaler = normalize_data(series_frame)
series_normalized_df = pd.DataFrame(data_normalized.reshape(720,1), columns=['price'])
series_normalized_df.plot(subplots=True)

column_indices = {name: i for i, name in enumerate(series_normalized_df.columns)}

# Split the data
n = len(series_normalized_df)
train_df = series_normalized_df[0:int(n*0.7)]
val_df = series_normalized_df[int(n*0.7):int(n*0.9)]
test_df = series_normalized_df[int(n*0.9):]

num_features = series_normalized_df.shape[1]

# Window generator class to make window
# https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing

# ██████   █████  ████████  █████
# ██   ██ ██   ██    ██    ██   ██
# ██   ██ ███████    ██    ███████
# ██   ██ ██   ██    ██    ██   ██
# ██████  ██   ██    ██    ██   ██

# ██     ██ ██ ███    ██ ██████   ██████  ██     ██ ██ ███    ██  ██████
# ██     ██ ██ ████   ██ ██   ██ ██    ██ ██     ██ ██ ████   ██ ██
# ██  █  ██ ██ ██ ██  ██ ██   ██ ██    ██ ██  █  ██ ██ ██ ██  ██ ██   ███
# ██ ███ ██ ██ ██  ██ ██ ██   ██ ██    ██ ██ ███ ██ ██ ██  ██ ██ ██    ██
#  ███ ███  ██ ██   ████ ██████   ██████   ███ ███  ██ ██   ████  ██████

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

	#  ██        ██ ███    ██ ██████  ███████ ██   ██ ███████ ███████      █████  ███    ██ ██████
	# ███        ██ ████   ██ ██   ██ ██       ██ ██  ██      ██          ██   ██ ████   ██ ██   ██
	#  ██        ██ ██ ██  ██ ██   ██ █████     ███   █████   ███████     ███████ ██ ██  ██ ██   ██
	#  ██        ██ ██  ██ ██ ██   ██ ██       ██ ██  ██           ██     ██   ██ ██  ██ ██ ██   ██
	#  ██ ██     ██ ██   ████ ██████  ███████ ██   ██ ███████ ███████     ██   ██ ██   ████ ██████

	#  ██████  ███████ ███████ ███████ ███████ ████████ ███████
	# ██    ██ ██      ██      ██      ██         ██    ██
	# ██    ██ █████   █████   ███████ █████      ██    ███████
	# ██    ██ ██      ██           ██ ██         ██         ██
	#  ██████  ██      ██      ███████ ███████    ██    ███████

class WindowGenerator():
    def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([f'Total window size: {self.total_window_size}', f'Input indices: {self.input_indices}', f'Label indices: {self.label_indices}', f'Label column name(s): {self.label_columns}'])

w1 = WindowGenerator(input_width=24, label_width=1, shift=24, label_columns=['price'])
w2 = WindowGenerator(input_width=6, label_width=1, shift=1, label_columns=['price'])

	# ██████         ███████ ██████  ██      ██ ████████
	#      ██        ██      ██   ██ ██      ██    ██
	#  █████         ███████ ██████  ██      ██    ██
	# ██                  ██ ██      ██      ██    ██
	# ███████ ██     ███████ ██      ███████ ██    ██

def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

WindowGenerator.split_window = split_window

# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),np.array(train_df[100:100+w2.total_window_size]), np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')

	# ██████         ██████  ██       ██████  ████████
	#      ██        ██   ██ ██      ██    ██    ██
	#  █████         ██████  ██      ██    ██    ██
	#      ██        ██      ██      ██    ██    ██
	# ██████  ██     ██      ███████  ██████     ██

w2.example = example_inputs, example_labels

def plot(self, model=None, plot_col='price', max_subplots=3):
	try:
		inputs, labels = self.example
	except:
		inputs, labels = self.example()

	plt.figure(figsize=(12, 8))
	plot_col_index = self.column_indices[plot_col]
	max_n = min(max_subplots, len(inputs))
	for n in range(max_n):
		plt.subplot(3, 1, n+1)
		plt.ylabel(f'{plot_col} [normed]')
		plt.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)

		if self.label_columns:
			label_col_index = self.label_columns_indices.get(plot_col, None)
		else:
			label_col_index = plot_col_index

		if label_col_index is None:
			continue

		plt.scatter(self.label_indices, labels[n, :, label_col_index],
				edgecolors='k', label='Labels', c='#2ca02c', s=64)

		if model is not None:
			predictions = model(inputs)
			plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)

		if n == 0:
			plt.legend()

	plt.xlabel('Time [h]')

WindowGenerator.plot = plot

w2.plot()
w2.plot(plot_col='price')

	# ██   ██         ██████ ██████  ███████  █████  ████████ ███████
	# ██   ██        ██      ██   ██ ██      ██   ██    ██    ██
	# ███████        ██      ██████  █████   ███████    ██    █████
	#      ██        ██      ██   ██ ██      ██   ██    ██    ██
	#      ██ ██      ██████ ██   ██ ███████ ██   ██    ██    ███████

	# ██████   █████  ████████  █████  ███████ ███████ ████████ ███████
	# ██   ██ ██   ██    ██    ██   ██ ██      ██         ██    ██
	# ██   ██ ███████    ██    ███████ ███████ █████      ██    ███████
	# ██   ██ ██   ██    ██    ██   ██      ██ ██         ██         ██
	# ██████  ██   ██    ██    ██   ██ ███████ ███████    ██    ███████

def make_dataset(self, data):
	data = np.array(data, dtype=np.float32)
	ds = tf.keras.preprocessing.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size, sequence_stride=1, shuffle=True, batch_size=32,)

	ds = ds.map(self.split_window)
	return ds

WindowGenerator.make_dataset = make_dataset

def train(self):
	return self.make_dataset(self.train_df)

def val(self):
	return self.make_dataset(self.val_df)

def test(self):
	return self.make_dataset(self.test_df)

def example(self):
	"""Get and cache an example batch of `inputs, labels` for plotting."""
	result = getattr(self, '_example', None)
	if result is None:
		# No example batch was found, so get one from the `.train` dataset
		result = next(iter(self.train()))
		# And cache it for next time
		self._example = result
	return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# Each element is an (inputs, label) pair
w2.train().element_spec

for example_inputs, example_labels in w2.train().take(1):
	print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
	print(f'Labels shape (batch, time, features): {example_labels.shape}')

# Inspect training set in w2:
print([l for l in list(w2.test().as_numpy_iterator())])

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

# ███████ ██ ███    ██  ██████  ██      ███████     ███████ ████████ ███████ ██████
# ██      ██ ████   ██ ██       ██      ██          ██         ██    ██      ██   ██
# ███████ ██ ██ ██  ██ ██   ███ ██      █████       ███████    ██    █████   ██████
#      ██ ██ ██  ██ ██ ██    ██ ██      ██               ██    ██    ██      ██
# ███████ ██ ██   ████  ██████  ███████ ███████     ███████    ██    ███████ ██

# ███    ███  ██████  ██████  ███████ ██      ███████
# ████  ████ ██    ██ ██   ██ ██      ██      ██
# ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████
# ██  ██  ██ ██    ██ ██   ██ ██      ██           ██
# ██      ██  ██████  ██████  ███████ ███████ ███████

# The simplest model you can build on this sort of data is one that predicts a single feature's value, 1 timestep (1h) in the future based only on the current conditions.
# Configure a WindowGenerator object to produce these single-step (input, label) pairs:

single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1, label_columns=['price'])

for example_inputs, example_labels in single_step_window.train().take(1):
	print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
	print(f'Labels shape (batch, time, features): {example_labels.shape}')


#---# ██████   █████  ███████ ███████ ██      ██ ███    ██ ███████
	# ██   ██ ██   ██ ██      ██      ██      ██ ████   ██ ██
	# ██████  ███████ ███████ █████   ██      ██ ██ ██  ██ █████
	# ██   ██ ██   ██      ██ ██      ██      ██ ██  ██ ██ ██
	# ██████  ██   ██ ███████ ███████ ███████ ██ ██   ████ ███████


# Creation of baseline
class Baseline(tf.keras.Model):
	def __init__(self, label_index=None):
		super().__init__()
		self.label_index = label_index

	def call(self, inputs):
		if self.label_index is None:
			return inputs
		result = inputs[:, :, self.label_index]
		return result[:, :, tf.newaxis]

baseline = Baseline(label_index=column_indices['price'])
baseline.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val())
performance['Baseline'] = baseline.evaluate(single_step_window.test(), verbose=0)
performance['Baseline']

# wide_window = WindowGenerator(input_width=24, label_width=24, shift=1,label_columns=['price'])
wide_window = WindowGenerator(input_width=24, label_width=8, shift=8,label_columns=['price'])

print('Input shape:', single_step_window.example()[0].shape)
print('Output shape:', baseline(single_step_window.example()[0]).shape)

wide_window.plot(baseline)

"""
In the above plots of three examples the single step model is run over the course of 24h. This deserves some explaination:
	- The blue "Inputs" line shows the input temperature at each time step. The model recieves all features, this plot only shows the temperature.
	- The green "Labels" dots show the target prediction value. These dots are shown at the prediction time, not the input time. That is why the range of labels is shifted 1 step relative to the inputs.
	- The orange "Predictions" crosses are the model's prediction's for each output time step. If the model were predicting perfectly the predictions would land directly on the "labels".
"""

#---# ██      ██ ███    ██ ███████  █████  ██████      ███    ███  ██████  ██████  ███████ ██
	# ██      ██ ████   ██ ██      ██   ██ ██   ██     ████  ████ ██    ██ ██   ██ ██      ██
	# ██      ██ ██ ██  ██ █████   ███████ ██████      ██ ████ ██ ██    ██ ██   ██ █████   ██
	# ██      ██ ██  ██ ██ ██      ██   ██ ██   ██     ██  ██  ██ ██    ██ ██   ██ ██      ██
	# ███████ ██ ██   ████ ███████ ██   ██ ██   ██     ██      ██  ██████  ██████  ███████ ███████

linear = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])
print('Input shape:', single_step_window.example()[0].shape)
print('Output shape:', linear(single_step_window.example()[0]).shape)

# This tutorial trains many models, so package the training procedure into a function:
MAX_EPOCHS = 100

def compile_and_fit(model, window, patience=2):
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,mode='min')

	model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])

	history = model.fit(window.train(), epochs=MAX_EPOCHS, validation_data=window.val(), callbacks=[early_stopping])
	return history

history = compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val())
performance['Linear'] = linear.evaluate(single_step_window.test(), verbose=0)

print('Input shape:', wide_window.example()[0].shape)
print('Output shape:', baseline(wide_window.example()[0]).shape)

wide_window.plot(linear)

# Return weights assigned to inputs
plt.bar(x = range(len(train_df.columns)),height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
_ = axis.set_xticklabels(train_df.columns, rotation=90)


#---# ██████  ███████ ███    ██ ███████ ███████
	# ██   ██ ██      ████   ██ ██      ██
	# ██   ██ █████   ██ ██  ██ ███████ █████
	# ██   ██ ██      ██  ██ ██      ██ ██
	# ██████  ███████ ██   ████ ███████ ███████

dense = tf.keras.Sequential([tf.keras.layers.Dense(units=64, activation='relu'), tf.keras.layers.Dense(units=64, activation='relu'), tf.keras.layers.Dense(units=1)])

history = compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val())
performance['Dense'] = dense.evaluate(single_step_window.test(), verbose=0)

wide_window.plot(dense)

# -----------------------------------------------------------

# Create a WindowGenerator that will produce batches of the 3h of inputs and, 1h of labels:
CONV_WIDTH = 20
conv_window = WindowGenerator(input_width=CONV_WIDTH,label_width=1,shift=1,label_columns=['price'])

conv_window.plot()
plt.title("Given 3h as input, predict 1h into the future.")

# Multi-step dense model

# Dense_1
multi_step_dense = tf.keras.Sequential([
	# Shape: (time, features) => (time*features)
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(units=32, activation='relu'),
	tf.keras.layers.Dense(units=32, activation='relu'),
	tf.keras.layers.Dense(units=1),
	# Add back the time dimension.
	# Shape: (outputs) => (1, outputs)
	tf.keras.layers.Reshape([1, -1]),
	])

print('Input shape:', conv_window.example()[0].shape)
print('Output shape:', multi_step_dense(conv_window.example()[0]).shape)

history = compile_and_fit(multi_step_dense, conv_window)

IPython.display.clear_output()
val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val())
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test(), verbose=0)
conv_window.plot(multi_step_dense)

multi_step_dense.summary()

# The main down-side of this approach is that the resulting model can only be executed on input wndows of exactly this shape.

print('Input shape:', wide_window.example()[0].shape)
try:
	print('Output shape:', multi_step_dense(wide_window.example()[0]).shape)
except Exception as e:
	print(f'\n{type(e).__name__}:{e}')


	#---#  ██████ ███    ██ ███    ██
		# ██      ████   ██ ████   ██
		# ██      ██ ██  ██ ██ ██  ██
		# ██      ██  ██ ██ ██  ██ ██
		#  ██████ ██   ████ ██   ████

# Create CNN
conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(CONV_WIDTH,),
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
])

# Run it on an example batch to see that the model produces outputs with the expected shape:
print("Conv model on `conv_window`")
print('Input shape:', conv_window.example()[0].shape)
print('Output shape:', conv_model(conv_window.example()[0]).shape)

# Train and evaluate it on the conv_window and it should give performance similar to the multi_step_dense model.
history = compile_and_fit(conv_model, conv_window)

IPython.display.clear_output()
val_performance['Conv'] = conv_model.evaluate(conv_window.val())
performance['Conv'] = conv_model.evaluate(conv_window.test(), verbose=0)

conv_window.plot(conv_model)

# The difference between this conv_model and the multi_step_dense model is that the conv_model can be run on inputs on inputs of any length. The convolutional layer is applied to a sliding window of inputs.
# If you run it on wider input, it produces wider output:
print("Wide window")
print('Input shape:', wide_window.example()[0].shape)
print('Labels shape:', wide_window.example()[1].shape)
print('Output shape:', conv_model(wide_window.example()[0]).shape)

"""
Note that the output is shorter than the input. To make training or plotting work, you need the labels, and prediction to have the same length. So build a WindowGenerator to produce wide windows with a few extra input time steps so the label and prediction lengths match:
"""
LABEL_WIDTH = 24
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['price'])


print("Wide conv window")
print('Input shape:', wide_conv_window.example()[0].shape)
print('Labels shape:', wide_conv_window.example()[1].shape)
print('Output shape:', conv_model(wide_conv_window.example()[0]).shape)

wide_conv_window.plot(conv_model)


	#---# ██████  ███    ██ ███    ██
		# ██   ██ ████   ██ ████   ██
		# ██████  ██ ██  ██ ██ ██  ██
		# ██   ██ ██  ██ ██ ██  ██ ██
		# ██   ██ ██   ████ ██   ████

# you will use an RNN layer called Long Short Term Memory (LSTM).

# An important constructor argument for all keras RNN layers is the return_sequences argument. This setting can configure the layer in one of two ways.
	# 1. If False, the default, the layer only returns the output of the final timestep, giving the model time to warm up its internal state before making a single prediction.

	# 2. If True the layer returns an output for each input. This is useful for:
		# Stacking RNN layers.
		# Training a model on multiple timesteps simultaneously.

wide_window = WindowGenerator(input_width=24, label_width=24, shift=1,label_columns=['price'])

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

print('Input shape:', wide_window.example()[0].shape)
print('Output shape:', lstm_model(wide_window.example()[0]).shape)

history = compile_and_fit(lstm_model, wide_window)

IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val(), verbose=1)
performance['LSTM'] = lstm_model.evaluate(wide_window.test(), verbose=0)

wide_window.plot(lstm_model)


#---# ██████  ███████ ██████  ███████  ██████  ██████  ███    ███  █████  ███    ██  ██████ ███████
	# ██   ██ ██      ██   ██ ██      ██    ██ ██   ██ ████  ████ ██   ██ ████   ██ ██      ██
	# ██████  █████   ██████  █████   ██    ██ ██████  ██ ████ ██ ███████ ██ ██  ██ ██      █████
	# ██      ██      ██   ██ ██      ██    ██ ██   ██ ██  ██  ██ ██   ██ ██  ██ ██ ██      ██
	# ██      ███████ ██   ██ ██       ██████  ██   ██ ██      ██ ██   ██ ██   ████  ██████ ███████

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [T (degC), normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()

for name, value in performance.items():
	print(f'{name:12s}: {value[1]:0.4f}')



#---# ███    ███ ██    ██ ██   ████████ ██      ██████  ██    ██ ████████ ██████  ██    ██ ████████
	# ████  ████ ██    ██ ██      ██    ██     ██    ██ ██    ██    ██    ██   ██ ██    ██    ██
	# ██ ████ ██ ██    ██ ██      ██    ██     ██    ██ ██    ██    ██    ██████  ██    ██    ██
	# ██  ██  ██ ██    ██ ██      ██    ██     ██    ██ ██    ██    ██    ██      ██    ██    ██
	# ██      ██  ██████  ███████ ██    ██      ██████   ██████     ██    ██       ██████     ██

"""
If (when) interested: https://www.tensorflow.org/tutorials/structured_data/time_series#multi-output_models
"""

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --

# ███    ███ ██    ██ ██   ████████ ██     ███████ ████████ ███████ ██████
# ████  ████ ██    ██ ██      ██    ██     ██         ██    ██      ██   ██
# ██ ████ ██ ██    ██ ██      ██    ██     ███████    ██    █████   ██████
# ██  ██  ██ ██    ██ ██      ██    ██          ██    ██    ██      ██
# ██      ██  ██████  ███████ ██    ██     ███████    ██    ███████ ██

"""
In a multi-step prediction, the model needs to learn to predict a range of future values. Thus, unlike a single step model, where only a single future point is predicted, a multi-step model predicts a sequence of the future values.

There are two rough approaches to this:
	- Single shot predictions where the entire time series is predicted at once.
	- Autoregressive predictions where the model only makes single step predictions and its output is fed back as its input.
"""

# Generate the fitting window
OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)
multi_window.plot()
multi_window

#---# ██████   █████  ███████ ███████ ██      ██ ███    ██ ███████ ███████
	# ██   ██ ██   ██ ██      ██      ██      ██ ████   ██ ██      ██
	# ██████  ███████ ███████ █████   ██      ██ ██ ██  ██ █████   ███████
	# ██   ██ ██   ██      ██ ██      ██      ██ ██  ██ ██ ██           ██
	# ██████  ██   ██ ███████ ███████ ███████ ██ ██   ████ ███████ ███████

# A simple baseline for this task is to repeat the last input time step for the required number of output timesteps

class MultiStepLastBaseline(tf.keras.Model):
	def call(self, inputs):
		return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val())
multi_performance['Last'] = last_baseline.evaluate(multi_window.val(), verbose=0)
multi_window.plot(last_baseline)

# Since this task is to predict 24h given 24h another simple approach is to repeat the previous day, assuming tomorrow will be similar

class RepeatBaseline(tf.keras.Model):
	def call(self, inputs):
		return inputs

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val())
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test(), verbose=0)
multi_window.plot(repeat_baseline)

#---# ███████ ██ ███    ██  ██████  ██      ███████     ███████ ██   ██  ██████  ████████
	# ██      ██ ████   ██ ██       ██      ██          ██      ██   ██ ██    ██    ██
	# ███████ ██ ██ ██  ██ ██   ███ ██      █████ █████ ███████ ███████ ██    ██    ██
	#      ██ ██ ██  ██ ██ ██    ██ ██      ██               ██ ██   ██ ██    ██    ██
	# ███████ ██ ██   ████  ██████  ███████ ███████     ███████ ██   ██  ██████     ██

# the model makes the entire sequence prediction in a single step

# --------------------------------------------------------------------------------------------------

	#---# ██      ██ ███    ██ ███████  █████  ██████
		# ██      ██ ████   ██ ██      ██   ██ ██   ██
		# ██      ██ ██ ██  ██ █████   ███████ ██████
		# ██      ██ ██  ██ ██ ██      ██   ██ ██   ██
		# ███████ ██ ██   ████ ███████ ██   ██ ██   ██

multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_linear_model, multi_window)

IPython.display.clear_output()
multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val())
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test(), verbose=0)
multi_window.plot(multi_linear_model)

# --------------------------------------------------------------------------------------------------

	#---# ██████  ███████ ███    ██ ███████ ███████
		# ██   ██ ██      ████   ██ ██      ██
		# ██   ██ █████   ██ ██  ██ ███████ █████
		# ██   ██ ██      ██  ██ ██      ██ ██
		# ██████  ███████ ██   ████ ███████ ███████

# DENSE_1 --------------
multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_dense_model, multi_window)

IPython.display.clear_output()
multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val())
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test(), verbose=0)
multi_window.plot(multi_dense_model)

# DENSE_2 -------------
multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_dense_model, multi_window)

IPython.display.clear_output()
multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val())
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test(), verbose=0)
multi_window.plot(multi_dense_model)

# --------------------------------------------------------------------------------------------------

	#---#  ██████ ███    ██ ███    ██
		# ██      ████   ██ ████   ██
		# ██      ██ ██  ██ ██ ██  ██
		# ██      ██  ██ ██ ██  ██ ██
		#  ██████ ██   ████ ██   ████


# CNN_1
CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_conv_model, multi_window)

IPython.display.clear_output()

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val())
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test(), verbose=0)
multi_window.plot(multi_conv_model)

# CNN_2

CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    tf.keras.layers.Dense(CONV_WIDTH),
])

history = compile_and_fit(multi_conv_model, multi_window)

IPython.display.clear_output()

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val())
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test(), verbose=0)
multi_window.plot(multi_conv_model)

# --------------------------------------------------------------------------------------------------

	#---# ██████  ███    ██ ███    ██
		# ██   ██ ████   ██ ████   ██
		# ██████  ██ ██  ██ ██ ██  ██
		# ██   ██ ██  ██ ██ ██  ██ ██
		# ██   ██ ██   ████ ██   ████

# Generate the fitting window
OUT_STEPS = 4
multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

# RNN 1 ---------
multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val())
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.train(), verbose=0)
multi_window.plot(multi_lstm_model)

# RNN 2 ---------
multi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units = 50, return_sequences = True),
	tf.keras.layers.Dropout(0.2),

	tf.keras.layers.LSTM(units = 50, return_sequences = True),
	tf.keras.layers.Dropout(0.2),

	tf.keras.layers.LSTM(units = 50, return_sequences = True),
	tf.keras.layers.Dropout(0.2),

	tf.keras.layers.LSTM(units = 50),
	tf.keras.layers.Dropout(0.2),

	tf.keras.layers.Dense(units = 1)
])

multi_lstm_model.summary()


history = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val(), verbose=0)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.train(), verbose=0)
multi_val_performance['LSTM']
multi_performance['LSTM']
multi_window.plot(multi_lstm_model)


#  █████  ██    ██ ████████  ██████
# ██   ██ ██    ██    ██    ██    ██
# ███████ ██    ██    ██    ██    ██ █████
# ██   ██ ██    ██    ██    ██    ██
# ██   ██  ██████     ██     ██████

# ██████  ███████  ██████  ██████  ███████ ███████ ███████ ██ ██    ██ ███████
# ██   ██ ██      ██       ██   ██ ██      ██      ██      ██ ██    ██ ██
# ██████  █████   ██   ███ ██████  █████   ███████ ███████ ██ ██    ██ █████
# ██   ██ ██      ██    ██ ██   ██ ██           ██      ██ ██  ██  ██  ██
# ██   ██ ███████  ██████  ██   ██ ███████ ███████ ███████ ██   ████   ███████

#---# ██████  ███    ██ ███    ██
	# ██   ██ ████   ██ ████   ██
	# ██████  ██ ██  ██ ██ ██  ██
	# ██   ██ ██  ██ ██ ██  ██ ██
	# ██   ██ ██   ████ ██   ████

"""
The model will have the same basic form as the single-step LSTM models: An LSTM followed by a layers.Dense that converts the LSTM outputs to model predictions.

A layers.LSTM is a layers.LSTMCell wrapped in the higher level layers.RNN that manages the state and sequence results for you (See Keras RNNs for details).

In this case the model has to manually manage the inputs for each step so it uses layers.LSTMCell directly for the lower level, single time step interface.
"""

# LSTM 1
class FeedBack(tf.keras.Model):
	def __init__(self, units, out_steps):
		super().__init__()
		self.out_steps = out_steps
		self.units = units
		self.lstm_cell = tf.keras.layers.LSTMCell(units)
		# Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
		self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
		self.dense = tf.keras.layers.Dense(num_features)

feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

# LSTM 2
class FeedBack(tf.keras.Model):
	def __init__(self, units, out_steps):
		super().__init__()
		self.out_steps = out_steps
		self.units = units
		self.lstm_cell = tf.keras.layers.LSTMCell(units)
		self.lstm_cell = tf.keras.layers.LSTMCell(units)
		# Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
		self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
		self.dense = tf.keras.layers.Dense(num_features)

feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)

"""
The first method this model needs is a warmup method to initialize is its internal state based on the inputs. Once trained this state will capture the relevant parts of the input history. This is equivalent to the single-step LSTM model from earlier:
"""

def warmup(self, inputs):
	# inputs.shape => (batch, time, features)
	# x.shape => (batch, lstm_units)
	x, *state = self.lstm_rnn(inputs)

	# predictions.shape => (batch, features)
	prediction = self.dense(x)
	return prediction, state

FeedBack.warmup = warmup

# This method returns a single time-step prediction, and the internal state of the LSTM:
prediction, state = feedback_model.warmup(multi_window.example()[0])
prediction.shape

"""
With the RNN's state, and an initial prediction you can now continue iterating the model feeding the predictions at each step back as the input.

The simplest approach to collecting the output predictions is to use a python list, and tf.stack after the loop.
"""

def call(self, inputs, training=None):
	# Use a TensorArray to capture dynamically unrolled outputs.
	predictions = []
	# Initialize the lstm state
	prediction, state = self.warmup(inputs)

	# Insert the first prediction
	predictions.append(prediction)

	# Run the rest of the prediction steps
	for n in range(1, self.out_steps):
		# Use the last prediction as input.
		x = prediction
		# Execute one lstm step.
		x, state = self.lstm_cell(x, states=state,
                          	training=training)
		# Convert the lstm output to a prediction.
		prediction = self.dense(x)
		# Add the prediction to the output
		predictions.append(prediction)

	# predictions.shape => (time, batch, features)
	predictions = tf.stack(predictions)
	# predictions.shape => (batch, time, features)
	predictions = tf.transpose(predictions, [1, 0, 2])
	return predictions

FeedBack.call = call

# Test run this model on the example inputs:
print('Output shape (batch, time, features): ', feedback_model(multi_window.example()[0]).shape)

# Now train the model:
history = compile_and_fit(feedback_model, multi_window)

IPython.display.clear_output()

multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val())
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test(), verbose=0)
multi_window.plot(feedback_model)


# ██████  ███████ ██████  ███████  ██████  ██████  ███    ███  █████  ███    ██  ██████ ███████
# ██   ██ ██      ██   ██ ██      ██    ██ ██   ██ ████  ████ ██   ██ ████   ██ ██      ██
# ██████  █████   ██████  █████   ██    ██ ██████  ██ ████ ██ ███████ ██ ██  ██ ██      █████
# ██      ██      ██   ██ ██      ██    ██ ██   ██ ██  ██  ██ ██   ██ ██  ██ ██ ██      ██
# ██      ███████ ██   ██ ██       ██████  ██   ██ ██      ██ ██   ██ ██   ████  ██████ ███████

x = np.arange(len(multi_performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()

for name, value in multi_performance.items():
	print(f'{name:8s}: {value[1]:0.4f}')
