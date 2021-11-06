import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP

# data set and correct labels for different logical functions
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_labels = np.array([row[0] ^ row[1] for row in inputs])
or_labels = np.array([row[0] or row[1] for row in inputs])
and_labels = np.array([row[0] and row[1] for row in inputs])
nand_labels = np.array([1, 1, 1, 0])
nor_labels = np.array([1, 0, 0, 1])
zero = np.array([0, 0, 0, 0])
one = np.array([1, 1, 1, 1])

mlp = MLP(2, [4], 1)

nb_correct = 0
nb_false = 0
average_accuracy = []
loss = []
cum_correct = []
cum_false = []

# training

target = or_labels
epochs = 1000
for epoch in range(epochs):
    #print("epoch", epoch)
    # loop over each data point in data set
    for i, data in enumerate(inputs):
        output = mlp.forward_step(data)
        mlp.backprop_step(target[i])
        if (output[0] < 0.5 and target[i] == 0) or (output[0] > 0.5 and target[i] == 1):
            nb_correct += 1
        else:
            nb_false += 1

        average_accuracy.append(nb_correct / (nb_correct + nb_false))
        cum_correct.append(nb_correct)
        cum_false.append(nb_false)
        loss.append(np.power((output[0] - target[i]), 2))

plt.plot(np.arange(epochs * 4)/4, average_accuracy)
plt.show()
plt.plot(np.arange(epochs * 4)/4, loss)
plt.show()
print("correct", nb_correct)
print("false", nb_false)
