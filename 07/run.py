import numpy as np
import tensorflow as tf

from model import LSTM_Model
from train_and_test_func import run_training_and_testing


def integration_task(seq_len, num_samples):
    """
    Generator function that yields random noise the size of seq_len for num_samples times and a target indicating
    the sum of the noise.

    :param int seq_len: The length of the random noise sequence to be returned
    :param int num_samples: The number of times noise samples should be yielded

    :return random noise with a size of seq_len
    :return the target (the sum of the noise)
    """

    for _ in range(num_samples):
        # create array of random noise
        result = np.random.normal(size=seq_len)
        target = np.array(int(np.sum(result) < 0), dtype=np.int32)
        yield np.expand_dims(result, -1), np.expand_dims(target, -1)


def my_integration_task():
    """
    Wrapper function for integration_task. Iterates through integration_task with a specified seq_len and
    num_samples and yields the function’s yield.
    """
    generator = integration_task(seq_len, num_samples)

    for sample in generator:
        yield sample


# create a data pipeline
@tf.function
def prepare_dataset(data):
    """
    Prepares the dataset by creating one-hot targets, shuffling, batching, and prefetching.

    :param data: the dataset to be prepared
    :return: data, the preprocessed dataset
    :rtype: tf.Tensor
    """
    # cast target to int to be able to do the next step
    data = data.map(lambda img, target: (img, tf.cast(target, tf.int32)))
    # create one-hot targets
    data = data.map(lambda img, target: (img, tf.one_hot(target, depth=2)))
    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    data = data.cache()
    # shuffle, batch, prefetch
    data = data.shuffle(6000)
    data = data.batch(64)
    data = data.prefetch(20)
    # return preprocessed dataset
    return data


# define number of time steps /sequence length
seq_len = 25

# number of samples for the validation dataset
num_samples = 8000

# create tensorflow dataset
valid_dataset = tf.data.Dataset.from_generator(my_integration_task,
                                         output_signature=(tf.TensorSpec(shape=(seq_len, 1)),
                                                           tf.TensorSpec(shape=1)))
valid_dataset = valid_dataset.apply(prepare_dataset)

# number of samples for the training dataset
num_samples = 80000

train_ds = tf.data.Dataset.from_generator(my_integration_task,
                                         output_signature=(tf.TensorSpec(shape=(seq_len, 1)),
                                                           tf.TensorSpec(shape=1)))
train_ds = train_ds.apply(prepare_dataset)


hidden_state_size = 3
lstm = LSTM_Model(hidden_state_size)

run_training_and_testing(lstm,
                         train_ds,
                         valid_dataset,
                         tf.keras.optimizers.Adam(learning_rate=0.1),
                         loss_func=tf.keras.losses.BinaryCrossentropy())


