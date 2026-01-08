import numpy as np

def convert_str_to_int(array: np.ndarray):
    map = {'I': 0, 'D': 1, 'U': 2}
    array = np.vectorize(lambda x: map.get(x, 0))(array)
    return array
