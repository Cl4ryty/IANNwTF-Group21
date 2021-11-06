import numpy as np
import activation_functions as af


class Perceptron:
    def __init__(self, input_units: int, alpha: float = 1):
        self.alpha = alpha
        self.number_of_inputs = input_units

        # set weights to random values
        self.weights = np.random.uniform(-1, 1, input_units+1) # including bias

        # initialize activation to -1 to represent that no forward step has been performed yet
        self.drive = None
        self.activation = None
        self.inputs = None

    def forward_step(self, inputs):
        """

        :type inputs: numpy array
        """
        self.inputs = np.insert(inputs, 0, 1)
        self.drive = np.sum(self.inputs * self.weights)
        self.activation = af.sigmoid(self.drive)

        return self.activation

    def update(self, delta):
        gradient = delta * self.inputs
        self.weights += self.alpha * gradient



