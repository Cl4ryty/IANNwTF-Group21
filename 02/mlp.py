import numpy as np
from perceptron import Perceptron
import activation_functions as af


class MLP:
    def __init__(self, inputs: int, hidden_layers, output_neurons: int):
        """
        Initializes a multi-layer perceptron with the given number of inputs,
        hidden layers with specified numbers of perceptrons, and given number of output neurons.

        :param int inputs: the number of inputs of this MLP
        :param hidden_layers: array or list of number of perceptrons in each hidden layer
        :param int output_neurons: number of output perceptrons
        """

        # for each hidden layer (entry in hidden layers), initialize corresponding number of perceptrons
        self.hidden_layers = []
        for i, perceptrons_in_layer in enumerate(hidden_layers):
            layer = []
            for ii in range(perceptrons_in_layer):
                layer.append(Perceptron(inputs if i == 0 else hidden_layers[i - 1]))
            self.hidden_layers.append(layer)

        # initialize output neurons
        self.output_neurons = []
        for i in range(output_neurons):
            self.output_neurons.append(Perceptron(hidden_layers[-1]))

        self.output = np.array([-1.0] * output_neurons)

    def forward_step(self, inputs):
        """
        Performs a forward pass through the MLP using the given inputs.

        :param inputs: array of numeric inputs
        :return: numpy array of outputs
        """
        # step through each layer and use the respective outputs as inputs for the next layer
        for layer in self.hidden_layers:
            next_inputs = []
            for perceptron in layer:
                next_inputs.append(perceptron.forward_step(inputs))
            inputs = next_inputs
        for i, perceptron in enumerate(self.output_neurons):
            self.output[i] = perceptron.forward_step(inputs)

        return self.output

    def backprop_step(self, targets):
        """
        Performs backpropagation through the MLP,
        adapting the perceptrons according to the error calculated given the targets.

        :param targets: target values for the MLP output given the last inputs
        """
        # calculate errors from results and targets
        m = len(self.output_neurons)
        deltas = -(targets - self.output) * [af.sigmoidprime(n.drive) for n in self.output_neurons]

        for i, neuron in enumerate(self.output_neurons):
            neuron.update(deltas[i])

        # iterate backwards through layers
        for l, layer in enumerate(reversed(self.hidden_layers)):
            layer_weights = np.array([])
            new_deltas = []  # list to store the error signals of all neurons of this layer

            for i, neuron in enumerate(layer):
                # calculate error for this neuron
                if l > 0:
                    tmp = [n.weights[i + 1] for n in self.hidden_layers[len(self.hidden_layers)-l]]
                else:
                    tmp = [n.weights[i + 1] for n in self.output_neurons]

                e = np.sum(np.array(deltas) * tmp) * af.sigmoidprime(neuron.drive)

                new_deltas.append(e)

                # update neuron
                for d in deltas:
                    neuron.update(d)

                np.append(layer_weights, neuron.weights)

            # the error the next layer uses is the error calculated for this layer
            deltas = new_deltas
