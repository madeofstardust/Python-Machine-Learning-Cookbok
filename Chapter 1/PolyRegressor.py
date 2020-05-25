#Polynomial Regressor
import numpy as np
import matplotlib.pyplot as plt

Time = np.array([6, 8, 9, 12, 16, 18, 19])
Temp = np.array([4, 7, 10, 11, 11.5, 9, 5])

'''polyfit function returns the coefficients for a polynomial of
degree n (given by us) that is the best fit for the data.'''
beta = np.polyfit(Time, Temp, 5)
'''To evaluate the
model at the specified points, we can use the poly1d() function. This function
returns the value of a polynomial of degree n evaluated at the points provided by
us. '''
p = np.poly1d(beta)
xp = np.linspace(6, 19, 100)
plt.figure()
plt.plot(Time, Temp, 'bo', xp, p(xp), '-')
plt.show()



