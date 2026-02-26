import librosa
import numpy as np


def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)

        # ============================================================
        # PHASE 1-7 (OLD VERSION - DENSE MODEL)
        # Averaged MFCC (Time dimension removed)
        # ============================================================
        #
        # mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        # mfcc = np.mean(mfcc.T, axis=0)
        # return mfcc
        #
        # ============================================================
        # PHASE 8 (NEW VERSION - CNN MODEL)
        # Full MFCC with time dimension preserved
        # ============================================================

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        # Pad or trim to fixed length
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc,
                          pad_width=((0, 0), (0, pad_width)),
                          mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc

    except Exception as e:
        print("Actual Error:", e)
        print("Error processing:", file_path)
        return None