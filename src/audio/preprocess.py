import librosa
import numpy as np

def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)

        return mfcc

    except Exception as e:
        print("Actual Error:", e)
        print("Error processing:", file_path)
        return None