# Arrays in numpy:
import numpy as np

#Matrices:
array1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(array1)

#one dim array with equidistance values 0-10:
array2 = np.arange(10)
print(array2)

#0-50, step=5:
array3 = np.arange(0, 50, 5)
print(array3)

#one dim array between two limits, with 50 equispaced values
array4 = np.linspace(0, 10, 50)
print(array4)

print(np.arange(6,11))

array5 = np.array([[np.linspace(0,10,5)],[np.arange(5)],[np.arange(6,11)]])
print(array5)
