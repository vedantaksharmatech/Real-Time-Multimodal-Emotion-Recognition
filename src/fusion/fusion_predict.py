# src/fusion/fusion_predict.py

import os
import numpy as np
import tensorflow as tf
import pickle
import whisper
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from src.realtime.speech_to_text import record_audio
from src.audio.preprocess import extract_features

# -------------------------------
# CONFIG
# -------------------------------

SAMPLE_RATE = 22050
EXPECTED_TIME_FRAMES = 174  # Must match CNN input during training

AUDIO_MODEL_PATH = "models/audio_cnn_model.h5"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

TEXT_MODEL_DIR = "D:/A_MY_FOLDER/Real_Time_Audio_Recognition_Frontend/models/text_emotion"

EMOTIONS = ["happy", "sad", "angry", "fear"]

AUDIO_WEIGHT = 0.3  # Base weight (text = 0.7)

# -------------------------------
# LOAD MODELS
# -------------------------------

print("Loading models...")

audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)

with open(LABEL_ENCODER_PATH, "rb") as f:
    audio_le = pickle.load(f)

print("Loading DistilBERT text model...")
tokenizer = DistilBertTokenizer.from_pretrained(TEXT_MODEL_DIR)
text_model = DistilBertForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
text_model.eval()

whisper_model = whisper.load_model("small")

print("✅ All models loaded successfully.\n")

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def transcribe_text(file_path):
    result = whisper_model.transcribe(file_path, language="en")
    return result["text"].strip()


def predict_audio(file_path):
    features = extract_features(file_path)

    if features is None:
        return np.zeros(len(EMOTIONS))

    features = features.reshape(1, 40, EXPECTED_TIME_FRAMES, 1)
    pred = audio_model.predict(features).flatten()

    # Map to 4-class if needed
    if len(pred) != len(EMOTIONS):
        mapped_pred = np.zeros(len(EMOTIONS))
        for i, emotion in enumerate(EMOTIONS):
            if emotion in audio_le.classes_:
                idx = np.where(audio_le.classes_ == emotion)[0][0]
                mapped_pred[i] = pred[idx]
        pred = mapped_pred

    # Normalize (IMPORTANT)
    pred = np.array(pred)
    if np.sum(pred) > 0:
        pred = pred / np.sum(pred)

    return pred


def predict_text(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy().flatten()

    mapped_probs = np.zeros(len(EMOTIONS))
    for i in range(min(len(probs), len(EMOTIONS))):
        mapped_probs[i] = probs[i]

    # Normalize (IMPORTANT)
    if np.sum(mapped_probs) > 0:
        mapped_probs = mapped_probs / np.sum(mapped_probs)

    return mapped_probs


def fuse_predictions(audio_probs, text_probs, audio_weight=AUDIO_WEIGHT):

    audio_conf = np.max(audio_probs)
    text_conf = np.max(text_probs)

    print("\n🔎 Audio confidence:", round(float(audio_conf), 4))
    print("🔎 Text confidence:", round(float(text_conf), 4))

    # Dynamic adjustment (prevents domination)
    if text_conf > audio_conf:
        audio_weight = 0.2
    else:
        audio_weight = AUDIO_WEIGHT

    text_weight = 1 - audio_weight

    print(f"🔹 Using weights → Audio: {audio_weight} | Text: {text_weight}")

    combined = audio_probs * audio_weight + text_probs * text_weight

    # Final normalization
    if np.sum(combined) > 0:
        combined = combined / np.sum(combined)

    print("🔹 Fused probabilities:", np.round(combined, 4))

    final_index = np.argmax(combined)
    return EMOTIONS[final_index]


# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":

    print("\n🎤 Recording audio (7 seconds)...")
    audio_file = record_audio(duration=7)

    print("\n📝 Transcribing audio to text...")
    text = transcribe_text(audio_file)
    print("🟢 Transcribed Text:", text)

    print("\n🎧 Predicting audio emotion...")
    audio_probs = predict_audio(audio_file)
    audio_emotion = EMOTIONS[np.argmax(audio_probs)]
    print("🟢 Audio Emotion:", audio_emotion)

    print("\n📝 Predicting text emotion (DistilBERT)...")
    text_probs = predict_text(text)
    text_emotion = EMOTIONS[np.argmax(text_probs)]
    print("🟢 Text Emotion:", text_emotion)

    print("\n🔹 Audio probabilities  = [happy: {:.2f}, sad: {:.2f}, angry: {:.2f}, fear: {:.2f}]".format(
        audio_probs[0], audio_probs[1], audio_probs[2], audio_probs[3]
    ))
    print("🔹 Text probabilities   = [happy: {:.2f}, sad: {:.2f}, angry: {:.2f}, fear: {:.2f}]".format(
        text_probs[0], text_probs[1], text_probs[2], text_probs[3]
    ))

    print("\n🎯 Fusing predictions...")
    final_emotion = fuse_predictions(audio_probs, text_probs)

    print("\n✅ Final Emotion:", final_emotion)


# -------------------------------
# PIPELINE FUNCTION
# -------------------------------

def run_fusion_pipeline():

    audio_file = record_audio(duration=10)

    if audio_file is None:
        raise ValueError("Audio recording failed — file path is None")

    text = transcribe_text(audio_file)

    audio_probs = predict_audio(audio_file)
    audio_emotion = EMOTIONS[np.argmax(audio_probs)]

    text_probs = predict_text(text)
    text_emotion = EMOTIONS[np.argmax(text_probs)]

    final_emotion = fuse_predictions(audio_probs, text_probs)

    return {
        "text": text,
        "audio_emotion": audio_emotion,
        "text_emotion": text_emotion,
        "final_emotion": final_emotion
    }




