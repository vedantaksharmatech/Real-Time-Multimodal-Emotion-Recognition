import os
import numpy as np
from preprocess import extract_features

#EMOTION MAPPING DICTIONARY
emotion_dict={
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def build_audio_dataset(data_path="data/audio"):
    x=[]
    y=[]

    #LOOP through each actor folder
    for actor in os.listdir(data_path):
        actor_path=os.path.join(data_path, actor)

        if os.path.isdir(actor_path):

            #Loop through each file inside actor folder
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    file_path=os.path.join(actor_path, file)

                    #Extract emotion from filename
                    emotion_code = file.split("-")[2]
                    emotion_label = emotion_dict.get(emotion_code)

                    # Extract features
                    features=extract_features(file_path)

                    if features is not None:
                        x.append(features)
                        y.append(emotion_label)
    return np.array(x), np.array(y)

if __name__ == "__main__":
    x,y = build_audio_dataset()

    print("Dataset shape:", x.shape)
    print("Labels shape:",y.shape)
    print("Unique emotions:",np.unique(y))
                    