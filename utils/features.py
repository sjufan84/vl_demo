''' Utils related to vocal feature extraction
and comparisons. '''
import numpy as np

def load_vocal_features(filename):
    ''' Load vocal features from a .npy file. '''
    return np.load(filename)