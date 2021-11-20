import tensorflow as tf
import numpy as np


# Custom dense layer
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units=256, activation=tf.nn.sigmoid):
        """
        Creates layer with the repsective number of units/neurons in it.

        :param integer units: number of neurons in the layer
        :param function activation: activation function used by the neurons in the layer
        :rtype: (Layer)object
        """
        super(MyDenseLayer, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        """
        Instantiates the weights of the layer.

        :param tf.TensorShape input_shape: determins number of weights
        """
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs, *args, **kwargs):
        """
        Computes the layer activation.

        :param tf.Tensor inputs: layer input
        :return: layer activation/outputs
        :rtype: tf.Tensor

        **kwargs dropout_value
        """

        x = tf.matmul(inputs, self.w) + self.b
        x = self.activation(x)
        if "dropout_value" in kwargs.keys() and 1 >= kwargs["dropout_value"] >= 0:
            x *= np.random.binomial(1, kwargs["dropout_value"], len(x))

        return x
