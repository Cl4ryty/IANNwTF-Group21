import tensorflow as tf


class LSTM_Cell(tf.keras.layers.Layer):
    """
    A simple LSTM cell.
    """
    @tf.function
    def __init__(self, units):
        """
        Initializes the LSTM cell.

        :param int units: The number of units
        """
        super(LSTM_Cell, self).__init__()

        # store the number of units for later use
        self.units = units

        # forget gate
        # - bias needs to be initialized as ones as a smaller value would basically introduce a vanishing gradient
        self.dense_forget = tf.keras.layers.Dense(units,
                                                  use_bias=True,
                                                  activation=tf.keras.activations.sigmoid,
                                                  bias_initializer='ones'
                                                  )

        # input gate
        self.dense_input = tf.keras.layers.Dense(units,
                                                 use_bias=True,
                                                 activation=tf.keras.activations.sigmoid)

        # output gate
        self.dense_output = tf.keras.layers.Dense(units,
                                                  use_bias=True,
                                                  activation=tf.keras.activations.sigmoid)

        # cell-state candidates
        self.dense_candidates = tf.keras.layers.Dense(units,
                                                      use_bias=True,
                                                      activation=tf.keras.activations.tanh)

    @tf.function
    def call(self, x, states):
        """
        Calls the cell with the input of a single time step and the previous states
        to get the hidden and cell states for the new time step.

        :param tf.Tensor x: the input for a single time step, expected shape is [batch_size, 1]
        :param tuple states: tuple of previous hidden and cell state - (hidden_state, cell_state)
        :return: hidden_state - the new hidden state of the LSTM cell
        :rtype: tf.Tensor
        :return: cell_state - the new cell state of the LSTM cell
        :rtype: tf.Tensor
        """
        # get hidden and cell state from states
        (hidden_state, cell_state) = states

        # concatenate previous hidden state and input
        input = tf.keras.layers.concatenate([hidden_state, x])

        # compute intermediate results for better readability
        f = self.dense_forget(input)
        i = self.dense_input(input)
        c_hat = self.dense_candidates(input)
        o = self.dense_output(input)

        cell_state = tf.multiply(f, cell_state) + tf.multiply(i, c_hat)
        hidden_state = tf.multiply(o, tf.keras.activations.tanh(cell_state))

        return hidden_state, cell_state


class LSTM_Layer(tf.keras.layers.Layer):
    """
    A single cell LSTM Layer.
    """
    @tf.function
    def __init__(self, cell):
        """
        Initializes the layer with the given LSTM cell.

        :param LSTM_Cell cell: The LSTM cell in this layer.
        """
        super(LSTM_Layer, self).__init__()
        self.cell = cell

    @tf.function
    def zero_states(self, batch_size):
        """
        Returns a tuple of the correct size tensors of zeros for the hidden and cell state of this layer's LSTM cell,
        given the batch_size.

        :param int batch_size: The batch size for which the zero states should be returned.
        :return: hidden_state of the correct size containing only zeros
        :rtype: tf.Tensor
        :return: cell_state of the correct size containing only zeros
        :rtype: tf.Tensor
        """
        return tf.zeros((batch_size, self.cell.units), tf.float32), tf.zeros((batch_size, self.cell.units), tf.float32)

    @tf.function
    def call(self, x, states):
        """
        Calls the layer with the input of multiple time steps and the initial states
        to get the hidden states for all time steps.

        :param tf.Tensor x: the input for multiple time step, expected shape is [batch_size, seq_len, input_size]
        :param tuple states: tuple of initial hidden and cell state - (hidden_state, cell_state)
        :return: states - the hidden states for all time steps
        :rtype: tf.Tensor
        """

        # input is expected to be of shape [batch size, seq len, input size]
        length = x.shape[1]

        # initialize state of the simple rnn cell
        (hstate, cstate) = states
        states = tf.TensorArray(tf.float32, size=length)

        # for each time step
        for t in tf.range(length):
            input_t = x[:, t, :]
            hstate, cstate = self.cell(input_t, states=(hstate, cstate))
            states = states.write(t, hstate)

        return states.stack()


class LSTM_Model(tf.keras.models.Model):
    """
    A simple LSTM model consisting of a LSTM layer with one LSTM cell and an output layer.
    """
    def __init__(self, units):
        """
        Initializes the Model with one layer of one cell with the given units,
        and an output layer for a binary classification task.

        :param int units: the number of hidden units of the LSTM cell
        """
        super(LSTM_Model, self).__init__()
        # self.dense = tf.keras.layers.Dense(128, activation="relu")
        self.lstm = LSTM_Layer(cell=LSTM_Cell(units))
        self.out = tf.keras.layers.Dense(2, activation="softmax")

    @tf.function
    def call(self, x):
        """
        Calls the model with the input of multiple time steps, initializes states to zero,
        and returns the output/prediction of the model.

        :param x: The input, expected shape is [batch_size, seq_len, input_size]
        :return: The output of the model - probabilities of the two outcomes.
        """
        x = self.lstm(x, self.lstm.zero_states(x.shape[0]))
        x = self.out(x)

        return x
