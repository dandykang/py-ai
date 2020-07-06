import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers, activations, Input


class Layers(layers.Layer):
    def __init__(self, num_outputs):
        super(Layers, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.fc = tf.keras.layers.Dense(self.num_outputs)

    def call(self, input):
        return self.fc(input)