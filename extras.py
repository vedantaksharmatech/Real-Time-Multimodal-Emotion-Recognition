from tensorflow.keras.models import load_model

model = load_model("models/emotion_model.h5")
print("Model output units:", model.layers[-1].output_shape)