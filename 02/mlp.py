import numpy as np
from perceptron import Perceptron
import activation_functions as af


class MLP:
    def __init__(self, inputs, hidden_layers, output_neurons):
        # hidden layers: array of neurons in each layer
        # output_neurons: int number

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

        # calculate errors from results and targets
        m = len(self.output_neurons)
        deltas = (2 / m) * (targets - self.output) * [af.sigmoidprime(n.drive) for n in self.output_neurons]

        for i, neuron in enumerate(self.output_neurons):
            neuron.update(deltas[i])

        # iterate backwards through layers
        for l, layer in enumerate(reversed(self.hidden_layers)):
            layer_weights = np.array([])
            new_deltas = []  # list to store the error signals of all neurons of this layer

            for i, neuron in enumerate(layer):
                # calculate error for this neuron
                e = np.sum(deltas *
                           [n.weights[i + 1] for n in self.hidden_layers[l - 1]] if l > 0 else [n.weights[i + 1] for n in self.output_neurons])\
                    * af.sigmoidprime(neuron.drive)

                new_deltas.append(e)

                # update neuron
                for d in deltas:
                    neuron.update(d)

                np.append(layer_weights, neuron.weights)

            # the error the next layer uses is the error calculated for this layer
            deltas = new_deltas
