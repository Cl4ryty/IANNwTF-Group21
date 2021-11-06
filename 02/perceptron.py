import numpy as np
import activation_functions as af


class Perceptron:
    def __init__(self, input_units: int, alpha: float = 1, function=af.sigmoid):
        """
        Initializes a perceptron with the given number of inputs, the given learning rate alpha,
        the given activation function and random weights and bias.

        :param int input_units: the number of inputs
        :param float alpha: the learning rate
        """
        self.alpha = alpha
        self.number_of_inputs = input_units
        self.func = function

        # set weights to random values
        self.weights = np.random.uniform(-1, 1, input_units + 1)  # including bias

        # set drive, activation, and inputs to None, these values will be stored for convenient access
        self.drive = None
        self.activation = None
        self.inputs = None

    def forward_step(self, inputs):
        """
        Performs the forward step by calculating the activation of the perceptron for the given inputs
        and storing inputs, drive, and activation.

        :return: the activation
        :type inputs: numpy array of numeric inputs
        """
        self.inputs = np.insert(inputs, 0, 1)
        self.drive = np.sum(self.inputs * self.weights)
        self.activation = self.func(self.drive)

        return self.activation

    def update(self, delta):
        """
        Updates the perceptron's weights and bias using the given delta.

        :param delta: the backpropagated error
        """
        gradient = delta * self.inputs
        self.weights -= self.alpha * gradient
