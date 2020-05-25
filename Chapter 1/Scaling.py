##Preprocessing:

'''Standardization or mean removal is a technique that simply centers data by removing the average value of each characteristic, 
and then scales it by dividing non-constant characteristics by their standard deviation. '''

import numpy as np
#Using sklearn:
from sklearn import preprocessing

data = np.array([[ 3, -1.5,  2, -5.4],
                 [ 0,  4,  -0.3, 2.1],
                 [ 1,  3.3, -1.9, -4.3]])


print("Mean: ",data.mean(axis=0))
print("Standard Deviation: ",data.std(axis=0))

data_standardized = preprocessing.scale(data)

print("Mean: ",data_standardized.mean(axis=0))
print("Standard Deviation: ",data_standardized.std(axis=0))

''' the scale() function has been used (z-score standardization). In summary, the z-score 
(also called the standard score) represents the number of standarddeviations by which the 
value of an observation point or data is greater than the mean value of what is observed or
measured. Values more than the mean have positive z-scores, while values less than the mean
have negative z-scores. The z-score is a quantity without dimensions that is obtained by 
subtracting the population's mean from a single rough score and then dividing the difference
by the standard deviation of the population. '''
''' Standardization is particularly useful when we do not know the minimum and maximum
for data distribution. In this case, it is not possible to use other forms of data
transformation. As a result of the transformation, the normalized values do not have a
minimum and a fixed maximum. Moreover, this technique is not influenced by the
presence of outliers, or at least not the same as other methods. '''

