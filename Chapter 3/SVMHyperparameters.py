# -*-coding: utf-8 -*-
# Hyperparameters
'''extracting hyperparameters for a model based on an SVM algorithm using the
grid search method.'''

from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import classification_report
import pandas as pd
import utilities

input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, 
                                                                    y, 
                                                                    test_size=0.25,
                                                                    random_state=5)
# Set the parameters by cross-validation
parameter_grid = {"C": [1, 10, 50, 600],
'kernel':['linear','poly','rbf'],
"gamma": [0.01, 0.001],
'degree': [2, 3]}

metrics = ['precision']

#Grid search:
'''The sklearn.model_selection.GridSearchCV() function performs an exhaustive
search over specified parameter values for an estimator. Exhaustive search (also 
named direct search, or brute force) is a comprehensive examination of all 
possibilities, and therefore represents an efficient solution method in which 
every possibility is tested to determine whether it is the solution.'''
from sklearn.model_selection import GridSearchCV

for metric in metrics:
    print("#### Grid Searching optimal hyperparameters for", metric)
    classifier = GridSearchCV(svm.SVC(C=1), 
                              parameter_grid, 
                              cv=5,
                              scoring=metric,
                              return_train_score=True)
    classifier.fit(X_train, y_train)
    
    print("Scores across the parameter grid:")
    GridSCVResults = pd.DataFrame(classifier.cv_results_)
    for i in range(0,len(GridSCVResults)):
        print(GridSCVResults.params[i], '-->', round(GridSCVResults.mean_test_score[i],3))
    print("Highest scoring parameter set:", classifier.best_params_)

y_true, y_pred = y_test, classifier.predict(X_test)
print("Full performance report:\n")
print(classification_report(y_true, y_pred))

# Perform a randomized search on hyper parameters
'''Unlike the GridSearchCV method, not all parameter values are tested in this 
method, but the parameter settings are sampled in a fixed number. The parameter 
settings that are tested are set through the n_iter attribute. Sampling without 
replacement is performed if the parameters are presented as a list. If at least 
one parameter is supplied as a distribution, substitution sampling is used.'''
from sklearn.model_selection import RandomizedSearchCV
parameter_rand = {'C': [1, 10, 50, 600],
                  'kernel':['linear','poly','rbf'],
                  'gamma': [0.01, 0.001],
                  'degree': [2, 3]}

metrics = ['precision']
for metric in metrics:
    print("#### Randomized Searching optimal hyperparameters for", metric)
    classifier = RandomizedSearchCV(svm.SVC(C=1),
                                    param_distributions=parameter_rand,n_iter=30,
                                    cv=5,
                                    return_train_score=True)
    classifier.fit(X_train, y_train)
    print("Scores across the parameter grid:")
    RandSCVResults = pd.DataFrame(classifier.cv_results_)
    for i in range(0,len(RandSCVResults)):
        print(RandSCVResults.params[i], '-->', round(RandSCVResults.mean_test_score[i]))

print("Highest scoring parameter set:", classifier.best_params_)
y_true, y_pred = y_test, classifier.predict(X_test)
print("Full performance report:\n")
print(classification_report(y_true, y_pred))
