import tensorflow as tf
from layer import MyDenseLayer


class MyModel(tf.keras.Model):

    def __init__(self):
        """
        Creates model object with its specific number of layer(-objects).

        :rtype: object
        """
        super(MyModel, self).__init__()
        self.dense1 = MyDenseLayer()
        self.dense2 = MyDenseLayer()
        self.out = MyDenseLayer(10, tf.nn.softmax)

    def call(self, inputs):
        """
        Performs a forward pass.

        :param tf.Tensor inputs: the model input
        :return: Model prediction for the respective input
        :rtype: tf.Tensor
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x