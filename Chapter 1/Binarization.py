#Binarization:
'''Binarization is used when you want to convert a numerical feature vector into a Boolean
vector. In the field of digital image processing, image binarization is the process by which a
color or grayscale image is transformed into a binary image, that is, an image with only two
colors. This technique is used for the recognition of objects, shapes, and, specifically, characters.
Through binarization, it is possible to distinguish the object of interest from the background
on which it is found'''

import numpy as np
#Using sklearn:
from sklearn import preprocessing

data = np.array([[ 3, -1.5,  2, -5.4],
                 [ 0,  4,  -0.3, 2.1],
                 [ 1,  3.3, -1.9, -4.3]])


data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
'''Values > threshold --> 1, values <= threshold --> 0'''
print(data_binarized)















