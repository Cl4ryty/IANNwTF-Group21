import tensorflow as tf
from layer import MyDenseLayer


class MyModel(tf.keras.Model):

    def __init__(self):
        """
        Creates model object with its specific number of layer(-objects).

        :rtype: object
        """
        super(MyModel, self).__init__()
        self.dense1 = MyDenseLayer(64)
        self.dense2 = MyDenseLayer(64)
        self.out = MyDenseLayer(1, tf.nn.sigmoid)
        self.dropout = False
        self.dropout_value = 0.2

    def call(self, inputs):
        """
        Performs a forward pass.

        :param tf.Tensor inputs: the model input
        :return: Model prediction for the respective input
        :rtype: tf.Tensor
        """
        x = self.dense1(inputs)
        if self.dropout:
            x = self.dense2(x, self.dropout_value)
        else:
            x = self.dense2(x)
        x = self.out(x)
        return x

    def enable_dropout(self):
        self.dropout = True

    def disable_dropout(self):
        self.dropout = False

    def set_dropout_value(self, value):
        self.dropout_value = value
