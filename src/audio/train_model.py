import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os

def train_model():

    # Load data
    X_train = np.load("models/X_train.npy")
    X_test = np.load("models/X_test.npy")
    y_train = np.load("models/y_train.npy")
    y_test = np.load("models/y_test.npy")

    print("Training data shape:", X_train.shape)

    # Get number of classes
    num_classes = len(np.unique(y_train))

    # Build Model
    model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),  # Explicit Input layer
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest Accuracy:", test_acc)

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/emotion_model.h5")

    print("\nModel saved successfully!")

if __name__ == "__main__":
    train_model()