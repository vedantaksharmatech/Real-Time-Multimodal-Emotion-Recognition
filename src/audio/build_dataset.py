import os
import numpy as np
#from preprocess import extract_features -> replacing with the 4th line
from src.audio.preprocess import extract_features

# NEW 4-CLASS EMOTION MAPPING
# Mapping 8 original emotions -> 4 final emotions
emotion_map = {
    "03": "happy",
    "08": "happy",     # surprised -> happy
    "04": "sad",
    "05": "angry",
    "07": "angry",     # disgust -> angry
    "06": "fear"       # fearful -> fear
    # "01" neutral -> removed
    # "02" calm -> removed
}

def build_audio_dataset(data_path="data/audio"):
    x = []
    y = []

    # LOOP through each actor folder
    for actor in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor)

        if os.path.isdir(actor_path):

            # LOOP through each file inside actor folder
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(actor_path, file)

                    # Extract emotion from filename
                    emotion_code = file.split("-")[2]

                    # Skip emotions not in our 4-class mapping
                    if emotion_code not in emotion_map:
                        continue

                    emotion_label = emotion_map[emotion_code]

                    # Extract features
                    features = extract_features(file_path)

                    if features is not None:
                        x.append(features)
                        y.append(emotion_label)

    return np.array(x), np.array(y)


if __name__ == "__main__":
    x, y = build_audio_dataset()

    print("Dataset shape:", x.shape)
    print("Labels shape:", y.shape)
    print("Unique emotions:", np.unique(y))