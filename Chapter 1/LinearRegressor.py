#Linear regressor:

import numpy as np
from sklearn import linear_model
#
import sklearn.metrics as sm
#saving the model:
import pickle

#Data:
filename = 'VehiclesItaly.txt'
X = []
y = []
with open (filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)
        
#splitting the ds:
num_training = int(0.8*len(X))
num_test = len(X) - num_training

X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])        
    
#model:
#The fit() method takes the input data and trains the model.
lin_regressor = linear_model.LinearRegression()
lin_regressor.fit(X_train, y_train)

#predictions:
y_train_pred = lin_regressor.predict(X_train)

#visualisation:
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()

y_test_pred = lin_regressor.predict(X_test)
plt.figure()
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()

##Evaluating accuracy:
print("Mean absolute error =", round(sm.mean_absolute_error(y_test,
y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test,
y_test_pred), 2))
print("Median absolute error =",
round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =",
round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

'''
An R2 score near 1 means that the model is able to predict the data very
well. Keeping track of every single metric can get tedious, so we pick one or 
two metrics to evaluate our model. A good practice is to make sure that the mean
squared error is low and the explained variance score is high.
'''

#Saving the model:
output_model_file = "3_model_linear_regr.pkl"
with open(output_model_file, 'wb') as f:
    pickle.dump(lin_regressor, f)

#Using model:
with open(output_model_file, 'rb') as f:
    model_linregr = pickle.load(f)
y_test_pred_new = model_linregr.predict(X_test)
print("New mean absolute error =",
round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))






