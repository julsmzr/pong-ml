import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler

def convert_str_to_int(array: np.ndarray):
    map = {'I': 0, 'D': 1, 'U': 2}
    array = np.vectorize(lambda x: map.get(x, 0))(array)
    return array

def min_max_scale(array: np.ndarray):
    scaler = MinMaxScaler()
    return scaler.fit_transform(array)

def undersample(X, y):
    return RandomUnderSampler().fit_resample(X,y)