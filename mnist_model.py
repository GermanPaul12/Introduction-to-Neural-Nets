from tensorflow import keras
import matplotlib.pyplot as plt

# Returns a compiled and trained model with the given hyper parameters


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

    # Create model
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    # Train model
    model.fit(X_train, y_train, epochs=epochs,
              validation_data=(X_test, y_test), verbose=False, batch_size=batch_size)

    # Return *trained* model
    return model

# Plots a graph where the x axis is the number of epochs and the y axis is the accuracy


def plot_accuracy_by_total_epochs(trained_models):
    # Sort the models by the number of epochs they were trained for
    trained_models.sort(key=lambda x: len(x.history.history["loss"]))
    
    plt.plot([len(model.history.history["val_accuracy"]) for model in trained_models], [model.history.history['val_accuracy'][-1]
             for model in trained_models])
    plt.xlabel('Total number of epochs')
    plt.ylabel('Accuracy of model')
    plt.show()
