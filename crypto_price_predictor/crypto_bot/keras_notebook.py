import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
Use of sequential models
https://keras.io/guides/sequential_model/
"""

# ██    ██ ███████ ███████
# ██    ██ ██      ██
# ██    ██ ███████ █████
# ██    ██      ██ ██
#  ██████  ███████ ███████


"""
A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
"""

# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
x
y = model(x)
y

"""
A Sequential model is not appropriate when:

    - Your model has multiple inputs or multiple outputs
    - Any of your layers has multiple inputs or multiple outputs
    - You need to do layer sharing
    - You want non-linear topology (e.g. a residual connection, a multi-branch model)
"""

#  ██████ ██████  ███████  █████  ████████ ██ ███    ██  ██████
# ██      ██   ██ ██      ██   ██    ██    ██ ████   ██ ██
# ██      ██████  █████   ███████    ██    ██ ██ ██  ██ ██   ███
# ██      ██   ██ ██      ██   ██    ██    ██ ██  ██ ██ ██    ██
#  ██████ ██   ██ ███████ ██   ██    ██    ██ ██   ████  ██████

# ███    ███  ██████  ██████  ███████ ██      ███████
# ████  ████ ██    ██ ██   ██ ██      ██      ██
# ██ ████ ██ ██    ██ ██   ██ █████   ██      ███████
# ██  ██  ██ ██    ██ ██   ██ ██      ██           ██
# ██      ██  ██████  ██████  ███████ ███████ ███████


"""
You can create a Sequential model by passing a list of layers to the Sequential constructor:
"""

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)

model.layers

# You can also create a Sequential model incrementally via the add() method:
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))

# Note that there's also a corresponding pop() method to remove layers: a Sequential model behaves very much like a list of layers.
model.pop()
print(len(model.layers))  # 2

"""
Also note that the Sequential constructor accepts a name argument, just like any layer or model in Keras. This is useful to annotate TensorBoard graphs with semantically meaningful names.
"""

model = keras.Sequential(name="my_sequential")
model.add(layers.Dense(2, activation="relu", name="layer1"))
model.add(layers.Dense(3, activation="relu", name="layer2"))
model.add(layers.Dense(4, name="layer3"))


# ███████ ██████  ███████  ██████ ██ ███████ ██    ██
# ██      ██   ██ ██      ██      ██ ██       ██  ██
# ███████ ██████  █████   ██      ██ █████     ████
#      ██ ██      ██      ██      ██ ██         ██
# ███████ ██      ███████  ██████ ██ ██         ██

# ██ ███    ██ ██████  ██    ██ ████████     ███████ ██   ██  █████  ██████  ███████
# ██ ████   ██ ██   ██ ██    ██    ██        ██      ██   ██ ██   ██ ██   ██ ██
# ██ ██ ██  ██ ██████  ██    ██    ██        ███████ ███████ ███████ ██████  █████
# ██ ██  ██ ██ ██      ██    ██    ██             ██ ██   ██ ██   ██ ██      ██
# ██ ██   ████ ██       ██████     ██        ███████ ██   ██ ██   ██ ██      ███████

"""
Generally, all layers in Keras need to know the shape of their inputs in order to be able to create their weights. So when you create a layer like this, initially, it has no weights:
"""

layer = layers.Dense(3)
layer.weights  # Empty

"""
It creates its weights the first time it is called on an input, since the shape of the weights depends on the shape of the inputs:
"""
# Call layer on a test input
x = tf.ones((1, 4))
y = layer(x)
layer.weights  # Now it has weights, of shape (4, 3) and (3,)

"""
Naturally, this also applies to Sequential models. When you instantiate a Sequential model without an input shape, it isn't "built": it has no weights (and calling model.weights results in an error stating just this). The weights are created when the model first sees some input data:
"""

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)  # No weights at this stage!

# At this point, you can't do this:
# model.weights

# You also can't do this:
# model.summary()

# Call the model on a test input
x = tf.ones((1, 4))
y = model(x)
print("Number of weights after calling the model:", len(model.weights))  # 6

"""
Once a model is "built", you can call its summary() method to display its contents:
"""

model.summary()

"""
However, it can be very useful when building a Sequential model incrementally to be able to display the summary of the model so far, including the current output shape. In this case, you should start your model by passing an Input object to your model, so that it knows its input shape from the start:
"""
model = keras.Sequential()
model.add(keras.Input(shape=(4,)))
model.add(layers.Dense(2, activation="relu"))

model.summary()

"""
A simple alternative is to just pass an input_shape argument to your first layer:
"""
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu", input_shape=(4,)))

model.summary()

"""
Models built with a predefined input shape like this always have weights (even before seeing any data) and always have a defined output shape.

In general, it's a recommended best practice to always specify the input shape of a Sequential model in advance if you know what it is.
"""


# ██████  ███████ ██████  ██    ██  ██████   ██████  ██ ███    ██  ██████
# ██   ██ ██      ██   ██ ██    ██ ██       ██       ██ ████   ██ ██
# ██   ██ █████   ██████  ██    ██ ██   ███ ██   ███ ██ ██ ██  ██ ██   ███
# ██   ██ ██      ██   ██ ██    ██ ██    ██ ██    ██ ██ ██  ██ ██ ██    ██
# ██████  ███████ ██████   ██████   ██████   ██████  ██ ██   ████  ██████


"""
When building a new Sequential architecture, it's useful to incrementally stack layers with add() and frequently print model summaries. For instance, this enables you to monitor how a stack of Conv2D and MaxPooling2D layers is downsampling image feature maps:
"""

model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))  # 250x250 RGB images
model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))

# Can you guess what the current output shape is at this point? Probably not.
# Let's just print it:
model.summary()

# The answer was: (40, 40, 32), so we can keep downsampling...

model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))

# And now?
model.summary()

# Now that we have 4x4 feature maps, time to apply global max pooling.
model.add(layers.GlobalMaxPooling2D())

# Finally, we add a classification layer.
model.add(layers.Dense(10))

model.summary()


# ███████ ███████  █████  ████████ ██    ██ ██████  ███████
# ██      ██      ██   ██    ██    ██    ██ ██   ██ ██
# █████   █████   ███████    ██    ██    ██ ██████  █████
# ██      ██      ██   ██    ██    ██    ██ ██   ██ ██
# ██      ███████ ██   ██    ██     ██████  ██   ██ ███████


# ███████ ██   ██ ████████ ██████   █████   ██████ ████████ ██  ██████  ███    ██
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██    ██ ██    ██ ████   ██
# █████     ███      ██    ██████  ███████ ██         ██    ██ ██    ██ ██ ██  ██
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██    ██ ██    ██ ██  ██ ██
# ███████ ██   ██    ██    ██   ██ ██   ██  ██████    ██    ██  ██████  ██   ████


"""
Once a Sequential model has been built, it behaves like a Functional API model. This means that every layer has an input and output attribute. These attributes can be used to do neat things, like quickly creating a model that extracts the outputs of all intermediate layers in a Sequential model:
"""

initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)

feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)

# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)

"""
Here's a similar example that only extract features from one layer:
"""
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)

feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)

# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
features
