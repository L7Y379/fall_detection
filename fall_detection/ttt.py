import numpy as np
import math
from keras.utils import np_utils

label=0
label2=1
label = np.row_stack((label, label2))
label = np_utils.to_categorical(label)
print(label)
print(label.shape)



#label=np.concatenate((label, label2), axis=0)
