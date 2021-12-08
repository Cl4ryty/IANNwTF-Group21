import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, n_filters=64, channel_dimension=64):
        super(ResidualBlock, self).__init__()
        # list to store all layers for easier calling
        self.layer_list = []

        # multiple alterations of Convolution and Batch Normalization layers
        self.layer_list.append(tf.keras.layers.Conv2D(filters=n_filters, kernel_size=1, padding="same"))
        self.layer_list.append(tf.keras.layers.BatchNormalization())
        self.layer_list.append(tf.keras.layers.Activation("relu"))

        self.layer_list.append(tf.keras.layers.Conv2D(filters=n_filters, kernel_size=3, padding="same"))
        self.layer_list.append(tf.keras.layers.BatchNormalization())
        self.layer_list.append(tf.keras.layers.Activation("relu"))

        self.layer_list.append(tf.keras.layers.Conv2D(filters=channel_dimension, kernel_size=1, padding="same"))
        self.layer_list.append(tf.keras.layers.BatchNormalization())
        self.layer_list.append(tf.keras.layers.Activation("relu"))

    def call(self, input, training=None):
        x = input
        # call all layers with the input
        for layer in self.layer_list:
            x = layer(x, training=training)
        # add input to output
        x = tf.keras.layers.Add()([x, input])
        return x


class ResNet(tf.keras.Model):

    def __init__(self, res_blocks=3):
        """
        Creates model object with its specific number of layer(-objects).

        :rtype: object
        """
        super(ResNet, self).__init__()

        self.layer_list = []
        self.layer_list.append(tf.keras.layers.Conv2D(filters=64,
                                                      kernel_size=3,
                                                      strides=(1, 1),
                                                      padding="same",
                                                      input_shape=(32, 32, 3)
                                                      ))
        # start with batch normalization and a non-linearity
        self.layer_list.append(tf.keras.layers.BatchNormalization())
        self.layer_list.append(tf.keras.layers.Activation("relu"))

        for i in range(res_blocks):
            self.layer_list.append(ResidualBlock())

        # flatten to feed into output layer
        self.layer_list.append(tf.keras.layers.GlobalAvgPool2D())

        self.layer_list.append(tf.keras.layers.BatchNormalization())
        self.layer_list.append(tf.keras.layers.Activation("relu"))

        self.layer_list.append(tf.keras.layers.Dense(10, activation="softmax"))

    def call(self, inputs, training):
        """
        Performs a forward pass.

        :param tf.Tensor inputs: the model input
        :return: Model prediction for the respective input
        :rtype: tf.Tensor
        """
        x = inputs
        for layer in self.layer_list:
            x = layer(x, training=training)

        return x


class TransitionLayer(tf.keras.layers.Layer):

    def __init__(self, n_features=64):
        """
        Instantiates the layers involved in a TransitionLayer.

        A transition layer is used to reduce the size of the feature maps and halve the number of feature maps.

        Args:
        n_features (int) : number of feature maps, which will be halved in this block
        """
        super(TransitionLayer, self).__init__()
        # list to store all layers for easier calling
        self.layer_list = []
        self.layer_list.append(tf.keras.layers.BatchNormalization(epsilon=1.001e-05))
        self.layer_list.append(tf.keras.layers.Activation("relu"))

        # bottleneck, reducing the number of feature maps
        # (floor divide current number of filters by two for the bottleneck)
        reduce_filters_to = n_features // 2

        # convolution with kernel size 1 and strides of 2 and a halved filter number to
        # halve number of feature maps and reduce their size
        self.layer_list.append(tf.keras.layers.Conv2D(filters=reduce_filters_to, kernel_size=1, padding="valid", strides=2, use_bias=False))
        self.layer_list.append(tf.keras.layers.BatchNormalization())
        self.layer_list.append(tf.keras.layers.Activation("relu"))

        # further reduce the feature map size through average pooling
        self.layer_list.append(tf.keras.layers.AvgPool2D(pool_size=2, padding="valid", strides=2))

    def call(self, input, training=None):
        # call all layers with the input
        for layer in self.layer_list:
            input = layer(input, training=training)
        return input


class DenseBlock(tf.keras.layers.Layer):

    def __init__(self, n_filters=64, new_channels=32):
        """
        Instantiates the layers involved in a DenseBlock.
        A block increases the number of channels but keeps the size of the feature maps constant.

        Args:
        n_filters (int) : number of filters used within the block (does not have an effect on n of output channels)

        new_channels (int) : number of channels to be added to the input by the block
        """

        super(DenseBlock, self).__init__()
        # list to store all layers for easier calling
        self.layer_list = []

        # start with a batch normalization and non-linearity
        self.layer_list.append(tf.keras.layers.BatchNormalization(epsilon=1.001e-05))
        self.layer_list.append(tf.keras.layers.Activation("relu"))

        # Conv2D to increase number of channels
        self.layer_list.append(tf.keras.layers.Conv2D(filters=n_filters, kernel_size=1, padding="valid", use_bias=False))
        self.layer_list.append(tf.keras.layers.BatchNormalization(epsilon=1.001e-05))
        self.layer_list.append(tf.keras.layers.Activation("relu"))

        # Conv2D to reduce number of feature maps before concatenating
        self.layer_list.append(tf.keras.layers.Conv2D(filters=new_channels, kernel_size=3, padding="same", use_bias=False))

    def __call__(self, input, training=None):
        x = input
        # call all layers with the input
        for layer in self.layer_list:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training)
            else:
                x = layer(x)

        # concatenate input and output along axis 3 (channel dimension)
        x = tf.keras.layers.Concatenate(axis=3)([x, input])
        return x


class DenseNet(tf.keras.Model):

    def __init__(self, blocks=3, dense_block_sizes=[3, 3, 3]):
        """
        Creates model object with its specific number of layer(-objects).

        :rtype: object
        """
        super(DenseNet, self).__init__()

        self.layer_list = []
        # start with a convolution to increase number of feature maps
        self.layer_list.append(tf.keras.layers.Conv2D(filters=64,
                                                      kernel_size=3,
                                                      strides=(1, 1),
                                                      padding="same",
                                                      input_shape=(32, 32, 3)
                                                      ))

        for i in range(blocks-1):
            for ii in range(dense_block_sizes[i]):
                self.layer_list.append(DenseBlock())
            self.layer_list.append(TransitionLayer())

        # last residual block outside of the loop as we do not want a transition layer after this one
        for i in range(dense_block_sizes[-1]):
            self.layer_list.append(DenseBlock())

        self.layer_list.append(tf.keras.layers.BatchNormalization(epsilon=1.001e-05))
        self.layer_list.append(tf.keras.layers.Activation("relu"))

        # flatten to feed into output layer
        self.layer_list.append(tf.keras.layers.GlobalAvgPool2D())
        self.layer_list.append(tf.keras.layers.Dense(10, activation="softmax"))

    def __call__(self, inputs, training=None):
        """
        Performs a forward pass.

        :param tf.Tensor inputs: the model input
        :return: Model prediction for the respective input
        :rtype: tf.Tensor
        """
        x = inputs
        for layer in self.layer_list:
            x = layer(x, training=training)

        return x
