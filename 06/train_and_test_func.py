import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
        prediction = model(input)
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
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


def run_training_and_testing(model,
                             train_ds,
                             test_ds,
                             optimizer,
                             loss_func=tf.keras.losses.CategoricalCrossentropy(),
                             epochs=10
                             ):
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

    print("Last test accuracy:", test_accuracies[-1])
    # Visualize accuracy and loss for training and test data.
    plt.figure()
    line1, = plt.plot(train_losses)
    line2, = plt.plot(test_losses)
    line3, = plt.plot(test_accuracies)
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1, line2, line3), ("training", "test", "test accuracy"))
    plt.show()
