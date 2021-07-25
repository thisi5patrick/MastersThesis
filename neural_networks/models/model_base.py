from abc import ABC

import cv2
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import Model

from .helpers import numbered_array_to_text


class ModelBase(ABC):
    model: Model = None

    def predictWord(self, x):

        encoded_prediction = self.model.predict(x=x, verbose=1)
        decoded_prediction = K.get_value(K.ctc_decode(encoded_prediction,
                                                      input_length=np.ones(encoded_prediction.shape[0]) *
                                                                   encoded_prediction.shape[1],
                                                      greedy=True)[0][0])
        final_text = []
        for item in decoded_prediction:
            final_text.append(numbered_array_to_text(item))

        return final_text
