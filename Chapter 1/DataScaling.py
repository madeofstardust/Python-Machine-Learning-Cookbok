#DataScaling
''' Remember, it is good practice to rescale data before training a machine
learning algorithm. With rescaling, data units are eliminated, allowing
you to easily compare data from different locations. '''

import numpy as np
#Using sklearn:
from sklearn import preprocessing

data = np.array([[ 3, -1.5,  2, -5.4],
                 [ 0,  4,  -0.3, 2.1],
                 [ 1,  3.3, -1.9, -4.3]])


print("Min: ",data.min(axis=0))
print("Max: ",data.max(axis=0))

data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)

print("Min: ",data_scaled.min(axis=0))
print("Max: ",data_scaled.max(axis=0))

print(data_scaled)

'''This method does not handle anomalous values well! '''








