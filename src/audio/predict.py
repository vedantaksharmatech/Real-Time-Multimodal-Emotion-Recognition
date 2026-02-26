import numpy as np
import tensorflow as tf
import pickle
from preprocess import extract_features

def predict_emotion(file_path):

    # Load trained model
    model = tf.keras.models.load_model("models/emotion_model.h5")

    # Load scaler
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Load label encoder
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Extract features
    features = extract_features(file_path)

    if features is None:
        print("Feature extraction failed.")
        return

    # Reshape (1 sample, 40 features)
    features = features.reshape(1, -1)

    # Scale features
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)

    predicted_class = np.argmax(prediction, axis=1)

    # Convert number back to emotion label
    emotion = label_encoder.inverse_transform(predicted_class)

    print("\nPredicted Emotion:", emotion[0])


if __name__ == "__main__":

    # Test with one RAVDESS file
    test_file = "data/audio/Actor_01/03-01-05-01-01-01-01.wav"

    predict_emotion(test_file)