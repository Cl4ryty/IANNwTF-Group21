import tensorflow as tf


class MyModel(tf.keras.Model):

    def __init__(self):
        """
        Creates model object with its specific number of layer(-objects).

        :rtype: object
        """
        print("test")
        super(MyModel, self).__init__()

        self.layer_list = []
        # -- feature extractor --
        # start with convolution that takes the whole image as input
        self.layer_list.append(tf.keras.layers.Conv2D(filters=32,
                                                  kernel_size=3,
                                                  strides=(1, 1),
                                                  padding="same",
                                                  input_shape=(28, 28, 1)
                                                  ))

        # then a pooling layer to decrease feature map size
        #self.layer_list.append(tf.keras.layers.MaxPool2D())

        # alternate these, gradually narrowing and deepening
        self.layer_list.append(tf.keras.layers.Conv2D(filters=64,
                                                  kernel_size=5,
                                                  strides=(1, 1),
                                                  padding="same",
                                                  ))
        self.layer_list.append(tf.keras.layers.MaxPool2D())

        self.layer_list.append(tf.keras.layers.Conv2D(filters=64,
                                                      kernel_size=12,
                                                      strides=(1, 1),
                                                      padding="same",
                                                      ))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=64,
                                                      kernel_size=12,
                                                      strides=(1, 1),
                                                      padding="same",
                                                      ))
        self.layer_list.append(tf.keras.layers.MaxPool2D())



        # -- classifier --
        # use dense layer(s) and an output layer of the size the output should be

        # before using a dense layer the input needs to be flattened
        self.layer_list.append(tf.keras.layers.GlobalAvgPool2D())

        #self.layer_list.append(tf.keras.layers.Dense(64))
        # output layer, fashion_mnist contains 10 classes, so 10 output neurons are required
        #self.layer_list.append(tf.keras.layers.Dense(10))
        self.layer_list.append(tf.keras.layers.Dense(10, activation="softmax"))

    def call(self, inputs):
        """
        Performs a forward pass.

        :param tf.Tensor inputs: the model input
        :return: Model prediction for the respective input
        :rtype: tf.Tensor
        """
        x = inputs
        for i,layer in enumerate(self.layer_list):
            x = layer(x)

        return x
