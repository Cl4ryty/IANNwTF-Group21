import tensorflow as tf
import numpy as np


def training_step(model, input, target, loss_function, optimizer):
    """
    Performs a training step of the model using the given imput and target,
    calculating the loss with the given function and then using the optimizer to optimize the model.

    :param tf.keras.Model model: the model to be trained
    :param tf.Tensor input: the input
    :param tf.Tensor target: the target
    :param tf.keras.losses.Loss loss_function: the loss function
    :param tf.keras.optimizers.Optimizer optimizer: the optimizer
    :return: loss - the loss for this training step
    :rtype: tf.Tensor
    """
    with tf.GradientTape() as tape:
        prediction = model.call(input)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def test(model, test_data, loss_function):
    """
    Tests how well the model performs over the test data by calculating mean loss and accuracy.

    :param tf.keras.Model model: The model to test
    :param tf.Tensor test_data: the test data, with each entry containing input and target, as a tensor
    :param tf.keras.losses.Loss loss_function: the loss function used to calculate the loss
    :return: test_loss - the mean loss
    :rtype: tf.Tensor
    :return: test_accuracy - the mean accuracy
    :rtype: tf.Tensor
    """
    # test over complete test data

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        #input = tf.reshape(input, (1, input.shape[0]))
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = target == np.round(prediction, 0)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy
