# Importing the necessary libraries
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Step 1: Generating synthetic data for a regression problem
def generate_data(samples=1000):
    # Generating random features and targets
    X = np.random.rand(samples, 3)  # 3 features
    y = 3 * X[:, 0] + 2 * X[:, 1] - 4 * X[:, 2] + np.random.randn(samples) * 0.1  # linear equation with noise
    return X, y

# Generating data
X, y = generate_data()

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 2: Create a simple neural network model using TensorFlow
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

# Step 3: Compile the model with loss function, optimizer, and evaluation metric
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Step 5: Evaluate the model on the test set
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {mae}")

with open('model_performance.txt', 'w') as f:
    f.write(f"Test Mean Absolute Error: {mae}\n")
