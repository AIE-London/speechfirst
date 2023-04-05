"""Utils that are not used anymore, but having it here just in case we ever need it."""
import numpy as np

def wav_to_numbers(file):
    try:
        numbers = file().data[0]
        return np.array(numbers, dtype=np.float32)
    except:
        return np.array([0], dtype=np.float32)