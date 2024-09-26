import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Generating synthetic data for a regression problem
def generate_data(samples=1000):
    """
    Generates synthetic data for a regression problem.
    Returns feature matrix X and target vector y.
    """
    X = np.random.rand(samples, 3)  # 3 features
    y = 3 * X[:, 0] + 2 * X[:, 1] - 4 * X[:, 2] + np.random.randn(samples) * 0.1  # linear equation with noise
    return X, y

# Step 2: Function to create, train, and evaluate a simple neural network model
def train_and_evaluate_model(X, y, test_size=0.2, epochs=50, batch_size=32):
    """
    Creates, trains, and evaluates a neural network model.
    Also, generates and saves a plot of training/validation MAE over epochs.

    Parameters:
    - X (array-like): Feature matrix.
    - y (array-like): Target variable.
    - test_size (float): Proportion of the data to be used for testing.
    - epochs (int): Number of epochs to train the model.
    - batch_size (int): Batch size for training.
    """

    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Step 3: Create the neural network model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    # Step 4: Compile the model with loss function, optimizer, and evaluation metric
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    # Step 5: Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Step 6: Evaluate the model on the test set
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test Mean Absolute Error: {mae}")

    # Step 7: Save the model performance to a file
    with open('model_performance.txt', 'w') as f:
        f.write(f"Test Mean Absolute Error: {mae}\n")

    # Step 8: Plot the training and validation MAE over the epochs
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Mean Absolute Error Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig('mae_plot.png')
    plt.show()

# Example usage
X, y = generate_data()
train_and_evaluate_model(X, y)
