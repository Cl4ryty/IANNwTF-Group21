import tensorflow as tf
import tensorflow_datasets as tfds

from model import ResNet, DenseNet
from train_and_test_func import run_training_and_testing


# create data pipeline
@tf.function
def prepare_dataset(data):
    # convert data from uint8 to float32
    data = data.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    data = data.map(lambda img, target: ((img / 128.) - 1., target))
    # create one-hot targets
    data = data.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    data = data.cache()
    # shuffle, batch, prefetch
    data = data.shuffle(6000)
    data = data.batch(64)
    data = data.prefetch(20)
    # return preprocessed dataset
    return data


# load the dataset
train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], as_supervised=True, shuffle_files=True)


# preprocessing the data
train_ds = train_ds.apply(prepare_dataset)
#valid_ds = valid_ds.apply(prepare_dataset)
test_ds = test_ds.apply(prepare_dataset)

run_training_and_testing(model=ResNet(),
                         test_ds=test_ds,
                         train_ds=train_ds,
                         optimizer=tf.optimizers.Adam(learning_rate=0.001),
                         epochs=10)

run_training_and_testing(model=DenseNet(),
                         test_ds=test_ds,
                         train_ds=train_ds,
                         optimizer=tf.optimizers.Adam(learning_rate=0.001),
                         epochs=10)


