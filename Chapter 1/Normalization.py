#Normalization
'''preprocessing.normalize() function scales input vectors individually to a unit norm
 (vector length)'''
 
import numpy as np
#Using sklearn:
from sklearn import preprocessing

data = np.array([[ 3, -1.5,  2, -5.4],
                 [ 0,  4,  -0.3, 2.1],
                 [ 1,  3.3, -1.9, -4.3]])


#L1- sum of columns = 1:
data_normalized_l1 = preprocessing.normalize(data, norm='l1', axis=0)
print(data_normalized_l1)

#L2 - Euclidean length, i.e. the square root of the sum of the squared vector elements"
data_normalized_l2 = preprocessing.normalize(data, norm='l2', axis=0)
print(data_normalized_l2)

#max - the maximum absolute vector element:
data_normalized_max = preprocessing.normalize(data, norm='max', axis=0)
print(data_normalized_max)

'''Thanks to normalization, the unit norm of each vector = 1
Scaling inputs to a unit norm is a very common task in text classification and clustering
problems.'''








