import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ---------------------------------------------------
# STEP 1: Create training data
# ---------------------------------------------------

data = {
    "text": [
        "I am very happy today",
        "This is the best day ever",
        "I feel amazing and joyful",
        "I am so sad and depressed",
        "I feel terrible today",
        "This is very disappointing",
        "I am extremely angry",
        "I can't believe you did that",
        "This makes me furious",
        "I am scared and nervous",
        "I feel afraid right now",
        "I am worried about everything"
    ],
    "emotion": [
        "happy",
        "happy",
        "happy",
        "sad",
        "sad",
        "sad",
        "angry",
        "angry",
        "angry",
        "fear",
        "fear",
        "fear"
    ]
}

df = pd.DataFrame(data)

# ---------------------------------------------------
# STEP 2: Train-test split (FIXED)
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["emotion"],
    test_size=0.33,   # <-- FIXED
    random_state=42,
    stratify=df["emotion"]
)

# ---------------------------------------------------
# STEP 3: TF-IDF Vectorization
# ---------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------------------------------
# STEP 4: Train classifier
# ---------------------------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------------------------------------------------
# STEP 5: Evaluate
# ---------------------------------------------------

y_pred = model.predict(X_test_vec)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------
# STEP 6: Save model
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "text_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(MODEL_DIR, "text_emotion_model.pkl"), "wb") as f:
    pickle.dump(model, f)

print("\nText emotion model trained and saved successfully!")