import pickle

with open("models/text_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/text_emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

while True:
    sentence = input("Enter sentence: ")
    vec = vectorizer.transform([sentence])
    prediction = model.predict(vec)[0]
    print("Predicted Emotion:", prediction)