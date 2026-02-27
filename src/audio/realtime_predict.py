import numpy as np
import tensorflow as tf
import pickle
import sounddevice as sd
from scipy.io.wavfile import write
import os
from preprocess import extract_features

# ==============================
# SETTINGS
# ==============================
SAMPLE_RATE = 16000  # match your training audio rate
DURATION = 5         # seconds to record

# ==============================
# RECORD AUDIO FUNCTION
# ==============================
def record_audio(filename="temp.wav"):
    print("\nRecording... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1)
    sd.wait()
    write(filename, SAMPLE_RATE, audio)
    print("Recording complete.\n")

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_emotion_from_mic():
    # Record live audio
    record_audio()

    # -----------------------------
    # Load trained CNN model
    # -----------------------------
    model = tf.keras.models.load_model("models/audio_cnn_model.h5")

    # Load label encoder
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # -----------------------------
    # Extract features
    # -----------------------------
    features = extract_features("temp.wav")

    if features is None:
        print("Feature extraction failed.")
        return

    # Reshape to match CNN input (40, 174, 1)
    features = features.reshape(1, 40, 174, 1)

    # -----------------------------
    # Predict emotion
    # -----------------------------
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)
    emotion = label_encoder.inverse_transform(predicted_class)

    print("Predicted Emotion:", emotion[0])

    # Optional: delete temporary audio file
    os.remove("temp.wav")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    predict_emotion_from_mic()