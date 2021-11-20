import pandas
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt

from model import MyModel
from train_and_test_func import training_step, test

wine_quality = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                           sep=";")
print(wine_quality.head())

# keys: 'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#        'pH', 'sulphates', 'alcohol', 'quality'

# quality should be target, all else should be input

# wine_ds = tf.data.Dataset.from_tensors(tf.convert_to_tensor(wine_quality))

# shuffling the data
wine_quality = wine_quality.sample(frac=1).reset_index(drop=True)
print("median", np.median(wine_quality["quality"]))
print("mean", np.mean(wine_quality["quality"]))
print("over 6", np.sum(wine_quality["quality"]>6))

# proportion of training and validation set, testing set size is determined automatically -> need to sum up to one
prop = [0.70, 0.15]  # testing: 0.1


# split data into train, validation, and test set
original_size = len(wine_quality)
train = wine_quality.head(int(original_size*prop[0]))
valid = wine_quality.tail(int(original_size*prop[1]))
testing = wine_quality.iloc[(int(original_size * prop[0])):(original_size - int(original_size * prop[1]))]

# separate labels from the input - might switch with previous step
train_labels = train[["quality"]]
train_data = train.drop(columns=["quality"])
valid_labels = valid[["quality"]]
valid_data = valid.drop(columns=["quality"])
test_labels = testing[["quality"]]
test_data = testing.drop(columns=["quality"])
# turn dataframes into tensorflow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
valid_ds = tf.data.Dataset.from_tensor_slices((valid_data, valid_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))


# function to binarize targets according to threshold
def make_binary(target):
    # arbitrary threshold
    threshold = 6
    return 0 if target <= threshold else 1


# create data pipeline
def prepare_dataset(data):
    """
    Input Pipeline which prepares the dataset for further processing
    :param data: the dataset
    :return: preprocessed dataset
    """
    data = data.map(lambda e, target: (e, make_binary(target)))
    data = data.cache()
    data = data.shuffle(1000)
    data = data.batch(8)
    data = data.prefetch(20)
    return data


train_ds = train_ds.apply(prepare_dataset)
valid_ds = valid_ds.apply(prepare_dataset)
test_ds = test_ds.apply(prepare_dataset)


# set hyperparameters
epochs = 10
learning_rate = 0.01

# initialize model, loss and optimizer
model = MyModel()
loss_func = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate)

# Initialize lists for later visualization.
train_losses = []
valid_losses = []
valid_accuracies = []
test_losses = []
test_accuracies = []

# testing once before we begin
valid_loss, valid_accuracy = test(model, valid_ds, loss_func)
valid_losses.append(valid_loss)
valid_accuracies.append(valid_accuracy)
test_loss, test_accuracy = test(model, test_ds, loss_func)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)


# check how model performs on train data once before we begin
train_loss, _ = test(model, train_ds, loss_func)
train_losses.append(train_loss)

# We train for epochs.
print("training baseline model")
for epoch in range(epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {valid_accuracies[-1]}')

    # training (and checking in with training)
    epoch_loss_agg = []
    for input, target in train_ds:
        #input = tf.reshape(input, (1, input.shape[0]))
        train_loss = training_step(model, input, target, loss_func, optimizer)
        epoch_loss_agg.append(train_loss)

    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    # testing, so we can track accuracy and test loss
    valid_loss, valid_accuracy = test(model, valid_ds, loss_func)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)
    test_loss, test_accuracy = test(model, test_ds, loss_func)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

# Visualize accuracy and loss for training and test data.
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(valid_losses)
line3, = plt.plot(valid_accuracies)
line4, = plt.plot(test_losses)
line5, = plt.plot(test_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.title("Optimized model")
plt.legend((line1, line2, line3, line4, line5), ("training", "validation", "validation accuracy", "test", "test accuracy"))
plt.show()

tf.keras.backend.clear_session()
# set hyperparameters
epochs = 10
learning_rate = 0.001

# initialize model, loss and optimizer
model = MyModel()
loss_func = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Initialize lists for later visualization.
train_losses = []
valid_losses = []
valid_accuracies = []
test_losses = []
test_accuracies = []

# testing once before we begin
valid_loss, valid_accuracy = test(model, valid_ds, loss_func)
valid_losses.append(valid_loss)
valid_accuracies.append(valid_accuracy)
test_loss, test_accuracy = test(model, test_ds, loss_func)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# check how model performs on train data once before we begin
train_loss, _ = test(model, train_ds, loss_func)
train_losses.append(train_loss)

# We train for epochs.
print("training optimized model")
for epoch in range(epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {valid_accuracies[-1]}')

    model.enable_dropout()
    # training (and checking in with training)
    epoch_loss_agg = []
    for input, target in train_ds.shuffle(1000).take(256):
        #input = tf.reshape(input, (1, input.shape[0]))
        train_loss = training_step(model, input, target, loss_func, optimizer)
        epoch_loss_agg.append(train_loss)

    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    model.disable_dropout()
    # testing, so we can track accuracy and test loss
    valid_loss, valid_accuracy = test(model, valid_ds, loss_func)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)
    test_loss, test_accuracy = test(model, test_ds, loss_func)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

# Visualize accuracy and loss for training and test data.
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(valid_losses)
line3, = plt.plot(valid_accuracies)
line4, = plt.plot(test_losses)
line5, = plt.plot(test_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.title("Optimized model")
plt.legend((line1, line2, line3, line4, line5), ("training", "validation", "validation accuracy", "test", "test accuracy"))
plt.show()

