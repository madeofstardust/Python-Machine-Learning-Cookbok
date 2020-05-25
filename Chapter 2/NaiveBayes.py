# Naive Bayes
'''The underlying principle of a Bayesian classifier is that some individuals belong to a class
of interest with a given probability based on some observations. This probability is based
on the assumption that the characteristics observed can be either dependent or independent
from one another; in this second case, the Bayesian classifier is called Naive because it
assumes that the presence or absence of a particular characteristic in a given class of interest
is not related to the presence or absence of other characteristics, greatly simplifying the
calculation.'''
'''The probability that a given event (E) occurs, is the ratio between the number (s) of
favorable cases of the event itself and the total number (n) of the possible cases, provided
all the considered cases are equally probable.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection

input_file = 'data_multivar.txt'
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])
X = np.array(X)
y = np.array(y)

classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
y_pred = classifier_gaussiannb.predict(X)

accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")

x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

# denotes the step size that will be used in the mesh grid
step_size = 0.01
# define the mesh grid
x_values, y_values = np.meshgrid(np.arange(x_min, x_max,
step_size), np.arange(y_min, y_max, step_size))
# compute the classifier output
mesh_output = classifier_gaussiannb.predict(np.c_[x_values.ravel(),
y_values.ravel()])
# reshape the array
mesh_output = mesh_output.reshape(x_values.shape)


# Plot the output using a colored plot
plt.figure()
# choose a color scheme
plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
# Overlay the training points on the plot
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='black',
linewidth=1, cmap=plt.cm.Paired)
# specify the boundaries of the figure
plt.xlim(x_values.min(), x_values.max())
plt.ylim(y_values.min(), y_values.max())
# specify the ticks on the X and Y axes
plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1),
1.0)))

plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1),
1.0)))
plt.show()

# Cross-Validation
'''Precision refers to the number of items that are correctly classified as a
percentage of the overall number of items in the list. Recall refers to the number of items
that are retrieved as a percentage of the overall number of items in the training list.'''

num_validations = 5
accuracy = model_selection.cross_val_score(classifier_gaussiannb,
                                           X,
                                           y,
                                           scoring='accuracy',
                                           cv=num_validations)
print ("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")

f1 = model_selection.cross_val_score(classifier_gaussiannb, 
                                     X,
                                     y,
                                     scoring='f1_weighted',
                                     cv=num_validations)
print ("F1: " + str(round(100*f1.mean(), 2)) + "%")

precision = model_selection.cross_val_score(classifier_gaussiannb,
                                            X,
                                            y,
                                            scoring='precision_weighted',
                                            cv=num_validations)
print ("Precision: " + str(round(100*precision.mean(), 2)) + "%")


