import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from model import MyModel
from train_and_test_func import test, training_step


# functions specific for working with the genomics_ood dataset

def onehotify(tensor):
    """
    turns a genome sequence into a usable one-hot vector

    :param tensor: genome sequence
    :return: flattened one-hot vectors
    """
    vocab = {'A': '1', 'C': '2', 'G': '3', 'T': '0'}
    for key in vocab.keys():
        tensor = tf.strings.regex_replace(tensor, key, vocab[key])
    split = tf.strings.bytes_split(tensor)
    labels = tf.cast(tf.strings.to_number(split), tf.uint8)
    onehot = tf.one_hot(labels, 4)
    onehot = tf.reshape(onehot, (-1,))
    return onehot


def prepare_genomics_data(genomics):
    """
    Input Pipeline which prepares the dataset for further processing

    :param genomics: genomics_ood dataset
    :return: preprocessed dataset
    """
    genomics = genomics.map(lambda e, target: (onehotify(e), tf.one_hot(target, depth=10)))
    genomics = genomics.cache()
    genomics = genomics.shuffle(1000)
    genomics = genomics.batch(64)
    genomics = genomics.prefetch(20)
    return genomics


# load and prepare the data
train_ds, test_ds = tfds.load("genomics_ood", split=['train', 'test'], shuffle_files=True, as_supervised=True)
train_ds = train_ds.take(100000)
test_ds = test_ds.take(1000)
train_ds = train_ds.apply(prepare_genomics_data)
test_ds = test_ds.apply(prepare_genomics_data)

# set hyperparameters
epochs = 10
learning_rate = 0.1

# initialize model, loss and optimizer
model = MyModel()
loss_func = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate)

# Initialize lists for later visualization.
train_losses = []
test_losses = []
test_accuracies = []

# testing once before we begin
test_loss, test_accuracy = test(model, test_ds, loss_func)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# check how model performs on train data once before we begin
train_loss, _ = test(model, train_ds, loss_func)
train_losses.append(train_loss)

# We train for epochs.
for epoch in range(epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    # training (and checking in with training)
    epoch_loss_agg = []
    for input, target in train_ds:
        train_loss = training_step(model, input, target, loss_func, optimizer)
        epoch_loss_agg.append(train_loss)

    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    # testing, so we can track accuracy and test loss
    test_loss, test_accuracy = test(model, test_ds, loss_func)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

# Visualize accuracy and loss for training and test data.
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
line3, = plt.plot(test_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend((line1, line2, line3), ("training", "test", "test accuracy"))
plt.show()
