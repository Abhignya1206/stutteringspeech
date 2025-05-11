import librosa
import numpy as np

def extract_mfcc(audio, sr, n_mfcc=29):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)

    if mfccs.shape[1] > 1:
        delta = librosa.feature.delta(mfccs)
        sorted_delta = np.sort(delta, axis=1)
        delta_mean = np.mean(sorted_delta, axis=1)
        features = np.concatenate((mfccs_mean, delta_mean), axis=0)
    else:
        features = mfccs_mean

    return features
