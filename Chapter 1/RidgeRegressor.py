#Ridge regressor
#Ridge regressor deals with outliers. 
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm

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
    

ridge_regressor = linear_model.Ridge(alpha=0.01,
                                     fit_intercept=True,    
                                     max_iter=10000)

#model:
#The fit() method takes the input data and trains the model.
ridge_regressor.fit(X_train, y_train)
y_test_pred_ridge = ridge_regressor.predict(X_test)
print( "Mean absolute error =",
round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2))
print( "Mean squared error =", round(sm.mean_squared_error(y_test,
y_test_pred_ridge), 2))
print( "Median absolute error =",
round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2))
print( "Explain variance score =",
round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2))
print( "R2 score =", round(sm.r2_score(y_test, y_test_pred_ridge),
2))

'''Ridge regression is a regularization method where a penalty is imposed on the size of the
coefficients. Ridge regression is identical to least squares, barring the fact that ridge
coefficients are computed by decreasing a quantity that is somewhat different. In
ridge regression, a scale transformation has a substantial effect. Therefore, to avoid
obtaining different results depending on the predicted scale of measurement, it is advisable
to standardize all predictors before estimating the model. To standardize the variables, we
must subtract their means and divide by their standard deviations.'''
