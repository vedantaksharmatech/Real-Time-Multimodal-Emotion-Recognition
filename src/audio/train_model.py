import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle

# Load data
X_train = np.load("models/X_train.npy")
X_test = np.load("models/X_test.npy")
y_train = np.load("models/y_train.npy")
y_test = np.load("models/y_test.npy")

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

num_classes = len(label_encoder.classes_)
print("Number of classes:", num_classes)

# ---------------- OLD MODEL (Dense-only before CNN) ----------------
# model = models.Sequential([
#     layers.Input(shape=(X_train.shape[1], X_train.shape[2], 1)),
#     layers.Flatten(),
#     layers.Dense(256, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(num_classes, activation='softmax')
# ])
# -------------------------------------------------------------------

# ---------------- CURRENT CNN MODEL ----------------
model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu', input_shape=(40,174,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation='softmax')
])
# -------------------------------------------------------------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save("models/audio_cnn_model.h5")
print("Model training complete and saved!")