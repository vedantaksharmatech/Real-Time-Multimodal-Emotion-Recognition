# src/fusion/fusion_predict.py

import os
import numpy as np
import tensorflow as tf
import pickle
import whisper
from src.realtime.speech_to_text import record_audio
from src.audio.preprocess import extract_features

# -------------------------------
# CONFIG
# -------------------------------

SAMPLE_RATE = 22050
EXPECTED_TIME_FRAMES = 174  # Must match CNN input during training
AUDIO_MODEL_PATH = "models/audio_cnn_model.h5"
TEXT_MODEL_PATH = "models/text_emotion_model.pkl"
VECTORIZER_PATH = "models/text_vectorizer.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# 4-class emotions
EMOTIONS = ["happy", "sad", "angry", "fear"]

# Weight for audio vs text in fusion
AUDIO_WEIGHT = 0.7

# -------------------------------
# LOAD MODELS
# -------------------------------

print("Loading models...")
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
with open(LABEL_ENCODER_PATH, "rb") as f:
    audio_le = pickle.load(f)
with open(TEXT_MODEL_PATH, "rb") as f:
    text_model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)
whisper_model = whisper.load_model("small")
print("✅ All models loaded successfully.\n")

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def transcribe_text(file_path):
    """
    Transcribe audio file to English text using Whisper.
    """
    result = whisper_model.transcribe(file_path, language="en")
    return result["text"]

def predict_audio(file_path):
    """
    Predict emotion from audio using CNN.
    Returns: probability vector (length 4)
    """
    features = extract_features(file_path)
    if features is None:
        return np.zeros(len(EMOTIONS))
    features = features.reshape(1, 40, EXPECTED_TIME_FRAMES, 1)
    pred = audio_model.predict(features).flatten()
    
    # Map to 4-class if audio model has more classes
    if len(pred) != len(EMOTIONS):
        # Convert to 4-class using label encoder mapping
        mapped_pred = np.zeros(len(EMOTIONS))
        for i, emotion in enumerate(EMOTIONS):
            if emotion in audio_le.classes_:
                idx = np.where(audio_le.classes_ == emotion)[0][0]
                mapped_pred[i] = pred[idx]
        pred = mapped_pred
    return pred

def predict_text(sentence):
    """
    Predict emotion from text using Logistic Regression.
    Returns: probability vector (length 4)
    """
    vec = vectorizer.transform([sentence])
    text_probs = text_model.predict_proba(vec).flatten()
    
    # Map text model classes to EMOTIONS order
    mapped_probs = np.zeros(len(EMOTIONS))
    for i, emotion in enumerate(EMOTIONS):
        if emotion in text_model.classes_:
            idx = np.where(text_model.classes_ == emotion)[0][0]
            mapped_probs[i] = text_probs[idx]
    return mapped_probs

def fuse_predictions(audio_probs, text_probs, audio_weight=AUDIO_WEIGHT):
    """
    Weighted fusion of audio and text probabilities.
    """
    combined = audio_probs * audio_weight + text_probs * (1 - audio_weight)
    final_index = np.argmax(combined)
    return EMOTIONS[final_index]

# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":
    print("\n🎤 Recording audio (5 seconds)...")
    audio_file = record_audio(duration=5)

    print("\n📝 Transcribing audio to text...")
    text = transcribe_text(audio_file)
    print("🟢 Transcribed Text:", text)

    print("\n🎧 Predicting audio emotion...")
    audio_probs = predict_audio(audio_file)
    audio_emotion = EMOTIONS[np.argmax(audio_probs)]
    print("🟢 Audio Emotion:", audio_emotion)

    print("\n📝 Predicting text emotion...")
    text_probs = predict_text(text)
    text_emotion = EMOTIONS[np.argmax(text_probs)]
    print("🟢 Text Emotion:", text_emotion)

    print("\n🎯 Fusing predictions...")
    final_emotion = fuse_predictions(audio_probs, text_probs)
    print("\n✅ Final Emotion:", final_emotion)