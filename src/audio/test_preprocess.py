from preprocess import extract_features
import os

# file_path="data\audio\Actor_01\03-01-01-01-01-01-01.wav"
file_path="data/audio/Actor_01/03-01-01-01-01-01-01.wav"


print("File exists:", os.path.exists(file_path))

features=extract_features(file_path)

print("Feature shape:", features.shape)
