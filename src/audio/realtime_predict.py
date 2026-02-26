import numpy as np
import tensorflow as tf
import pickle
import sounddevice as sd
from scipy.io.wavfile import write
import os
from preprocess import extract_features

SAMPLE_RATE = 22050
DURATION = 4  # seconds

def record_audio(filename="temp.wav"):
    print("\nRecording... Speak now!")

    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1)

    sd.wait()

    write(filename, SAMPLE_RATE, audio)
    print("Recording complete.\n")

def predict_emotion_from_mic():

    # Record audio
    record_audio()

    # Load model
    model = tf.keras.models.load_model("models/emotion_model.h5")

    # Load scaler
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load label encoder
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Extract features
    features = extract_features("temp.wav")

    if features is None:
        print("Feature extraction failed.")
        return

    features = features.reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)
    emotion = label_encoder.inverse_transform(predicted_class)

    print("Predicted Emotion:", emotion[0])

    # Optional cleanup
    os.remove("temp.wav")


if __name__ == "__main__":
    predict_emotion_from_mic()