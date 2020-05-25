#Random Forest
import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, explained_variance_score)


filename="bike_day.csv"
file_reader = csv.reader(open(filename, 'r'), delimiter=',')
X, y = [], []
for row in file_reader:
    X.append(row[2:13])
    y.append(row[-1])

#Extracting features names:
feature_names = np.array(X[0])

#removing features names
X=np.array(X[1:]).astype(np.float32)
y=np.array(y[1:]).astype(np.float32)

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=7)

num_training = int(0.9 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

'''n_estimators refers to the number of estimators, which is the number of
decision trees that we want to use in our random forest. The max_depth
parameter refers to the maximum depth of each tree, and the
min_samples_split parameter refers to the number of data samples that are
needed to split a node in the tree.'''
rf_regressor = RandomForestRegressor(n_estimators=1000,
                                     max_depth=10, 
                                     min_samples_split=2)
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print( "#### Random Forest regressor performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

RFFeatureImp= rf_regressor.feature_importances_
RFFeatureImp= 100.0 * (RFFeatureImp / max(RFFeatureImp))
index_sorted = np.flipud(np.argsort(RFFeatureImp))
pos = np.arange(index_sorted.shape[0]) + 0.5

plt.figure()
plt.bar(pos, RFFeatureImp[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted], rotation=90)
plt.tick_params(axis='x', which='major', labelsize=6)
plt.ylabel('Relative Importance')
plt.title("Random Forest regressor")
plt.show()


