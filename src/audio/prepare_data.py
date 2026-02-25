import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from build_dataset import build_audio_dataset
import pickle
import os

def prepare_data():

    # Load dataset
    X, y = build_audio_dataset()

    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)

    # Encode labels (text → numbers)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save processed data
    os.makedirs("models", exist_ok=True)

    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save("models/X_train.npy", X_train)
    np.save("models/X_test.npy", X_test)
    np.save("models/y_train.npy", y_train)
    np.save("models/y_test.npy", y_test)

    print("\nData preparation complete!")
    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)

if __name__ == "__main__":
    prepare_data()