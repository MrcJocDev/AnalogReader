"""Import MobileNet and load Model"""

from keras.applications.mobilenet_v2 import MobileNetV2 # type:ignore
model = MobileNetV2(weights='imagenet')

"""Loading Image"""

import numpy as np
from imageio import imread # type:ignore
from keras.applications.mobilenet_v2 import preprocess_input # type:ignore

data = np.empty((1, 375, 500, 3))
data[0] = imread('imagenet/img (1).jpeg')
data = preprocess_input(data)

"""Classify image"""
from keras.applications.mobilenet_v2 import decode_predictions # type:ignore

predictions = model.predict(data)
for name, desc, score in decode_predictions(predictions, top=5)[0]:
    print("- {} ({:2f}%".format(desc, 100 * score))

