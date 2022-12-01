from tensorflow import keras
import matplotlib.pyplot as plt

# Returns a compiled and trained model with the given hyper parameters


def get_compiled_model(optimizer='adam', loss='mse', metrics=['accuracy']):
    # Create model
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    # Return *compiled* model
    return model


def get_trained_mnist_model(epochs=10, batch_size=32, optimizer='adam', loss='mse', metrics=['accuracy'], y_one_hot_encode=True):

    # Load MNIST data
    (X_train, y_train), (X_test,  y_test) = keras.datasets.mnist.load_data()

    # Normalize data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    if y_one_hot_encode:
        # One hot encode y data
        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)

    model = get_compiled_model(
        optimizer=optimizer, loss=loss, metrics=metrics)

    # Train model
    model.fit(X_train, y_train, epochs=epochs,
              validation_data=(X_test, y_test), verbose=False, batch_size=batch_size)

    # Return *trained* model
    return model

# Plots a graph where the x axis is the number of epochs and the y axis is the accuracy


def plot_accuracy_by_total_epochs(trained_models):
    # Sort the models by the number of epochs they were trained for
    trained_models.sort(key=lambda x: len(x.history.history["loss"]))

    plt.plot([len(model.history.history["accuracy"]) for model in trained_models], [model.history.history['accuracy'][-1]
             for model in trained_models])
    plt.xlabel('Total number of epochs')
    plt.ylabel('Accuracy of model')
    plt.show()


def plot_accuracy_by_batch_size(trained_models, batch_sizes):
    for model, batch_size in zip(trained_models, batch_sizes):
        plt.plot(range(1, len(model.history.history['loss'])+1),
                 model.history.history['accuracy'], label=f"Batch size: {batch_size}")
    plt.legend(loc="lower right")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy of model')
    plt.show()


def plot_loss_by_learning_rate(trained_models, learning_rates):
    for model, learning_rate in zip(trained_models, learning_rates):
        plt.plot(range(1, len(model.history.history['loss'])+1),
                 model.history.history['loss'], label=f"Learning Rate: {learning_rate}")
    plt.legend(loc="upper left")
    plt.xlabel('Epoch')
    plt.ylabel('Loss of model')
    plt.show()
