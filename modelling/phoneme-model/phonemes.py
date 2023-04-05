from typing import Dict
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('phoneme_convolutional_softmax')
consonants_ordered_list = ['b', 'd', 'g', 'k', 'm', 'n', 'p', 't']  # matches preds order

def get_phoneme_prediction(wav_m: np.ndarray) -> Dict[str, float]:
    wav_m = wav_m.reshape((1, 16, 1, 96))
    preds = model(wav_m)
    # EagerTensor cannot be converted to a Python list directly
    # convert first to np.array then Python list
    preds_arr = np.array(preds)
    preds_arr = preds_arr.flatten()  # convert from 2D to 1D
    preds_list = preds_arr.tolist()
    # create dict of consonant: prediction
    preds_dict = dict(zip(consonants_ordered_list, preds_list))
    
    return preds_dict