#Decision Tree
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import (mean_squared_error,
explained_variance_score)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

housing_data = datasets.load_boston()
'''Shuffling data reduces
variance and makes sure that the patterns remain general and less
overfitted.'''
X, y = shuffle(housing_data.data, housing_data.target,
random_state=7)

#training and test set
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

#regressor without Adaboost:
dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train, y_train)

'''AdaBoost stands for
adaptive boosting, and this is a technique that is used to boost the accuracy 
of the results from another system. This combines the outputs from different 
versions of the algorithms, called weak learners, using a weighted summation to
get the final output. The information that's collected at each stage of the 
AdaBoost algorithm is fed back into the system so that the learners at the 
latter stages focus on training samples that are difficult to classify.'''

#regressor with AdaBoost
ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                                 n_estimators=400, random_state=7)
ab_regressor.fit(X_train, y_train)


#evaluating (without AdaBoost):
y_pred_dt = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_dt)
evs = explained_variance_score(y_test, y_pred_dt)
print("#### Decision Tree performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

#evaluating (with AB)
y_pred_ab = ab_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_ab)
evs = explained_variance_score(y_test, y_pred_ab)
print("#### AdaBoost performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

#Features importance:
'''not all features contribute equally to the output. In
case we want to discard some of them later, we need to know which features are
less important. We have this functionality available in scikit-learn'''

#feature_importances_ method that gives us the relative importance of each feature
FeatureImp= dt_regressor.feature_importances_
#Normalization:
FeatureImp= 100.0 * (FeatureImp / max(FeatureImp))
#flipping (descending order of importance)
index_sorted = np.flipud(np.argsort(FeatureImp))
pos = np.arange(index_sorted.shape[0]) + 0.5
#Visualisation:

plt.figure()
plt.bar(pos, FeatureImp[index_sorted], align='center')
plt.xticks(pos, housing_data.feature_names[index_sorted])
plt.tick_params(axis='x', which='major', labelsize=6)
plt.ylabel('Relative Importance')
plt.title("Decision Tree regressor")
plt.show()

#The same, but with AdaBoost:
ABFeatureImp= ab_regressor.feature_importances_
ABFeatureImp= 100.0 * (ABFeatureImp / max(ABFeatureImp))
index_sorted = np.flipud(np.argsort(ABFeatureImp))
pos = np.arange(index_sorted.shape[0]) + 0.5

plt.figure()
plt.bar(pos, ABFeatureImp[index_sorted], align='center')
plt.xticks(pos, housing_data.feature_names[index_sorted])
plt.tick_params(axis='x', which='major', labelsize=6)
plt.ylabel('Relative Importance')
plt.title("AdaBoost regressor")
plt.show()

'''Withoud AdaBoost we may conclude that the most important feature is RM 
(Average number of rooms per dwelling). Using AdaBoost shows though, that the
most important feature is LSTAT (Percent of the lower status of the population),
which, in reality, is true.'''
