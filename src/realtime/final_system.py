import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

# ==============================
# CONFIGURATION
# ==============================

SAMPLE_RATE = 22050
DURATION = 5  # seconds
EXPECTED_TIME_FRAMES = 174  # Must match training

emotion_labels = [
    'neutral',
    'calm',
    'happy',
    'sad',
    'angry',
    'fearful',
    'disgust',
    'surprised'
]

# ==============================
# LOAD MODEL
# ==============================

print("Loading CNN model...")
model = tf.keras.models.load_model(
    "D:/A_MY_FOLDER/Real_Time_Audio_Recognition/models/audio_cnn_model.h5"
)
print("Model loaded successfully.\n")

print("Model expected input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

# ==============================
# RECORD AUDIO
# ==============================

print("\nRecording for 5 seconds... Speak now!")
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32'
)
sd.wait()
print("Recording complete.\n")

audio = np.squeeze(audio)

# ==============================
# FEATURE EXTRACTION (CNN VERSION)
# ==============================

def extract_features(data, sample_rate):
    mfcc = librosa.feature.mfcc(
        y=data,
        sr=sample_rate,
        n_mfcc=40
    )

    # Fix time dimension to match training (174)
    if mfcc.shape[1] < EXPECTED_TIME_FRAMES:
        pad_width = EXPECTED_TIME_FRAMES - mfcc.shape[1]
        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode='constant'
        )
    else:
        mfcc = mfcc[:, :EXPECTED_TIME_FRAMES]

    return mfcc


features = extract_features(audio, SAMPLE_RATE)

# Add channel dimension (for CNN)
features = np.expand_dims(features, axis=-1)

# Add batch dimension
features = np.expand_dims(features, axis=0)

print("Feature shape for prediction:", features.shape)

# ==============================
# PREDICTION
# ==============================

prediction = model.predict(features)

print("\nRaw prediction vector:", prediction)

predicted_index = np.argmax(prediction)
confidence = float(np.max(prediction)) * 100

# ==============================
# SAFE LABEL MAPPING
# ==============================

if predicted_index < len(emotion_labels):
    predicted_emotion = emotion_labels[predicted_index]
else:
    predicted_emotion = "Unknown (Label mismatch)"

# ==============================
# RESULT
# ==============================

print("\n🎯 Predicted Emotion:", predicted_emotion)
print("Confidence:", round(confidence, 2), "%")