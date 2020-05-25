# Decision Tree - wines

import numpy as np
from sklearn import datasets

input_file = datasets.load_wine
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[1:])
        y.append(data[0])
X = np.array(X)
y = np.array(y)

from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                    y,
                                                                    test_size=0.25,
                                                                    random_state=5)
from sklearn.tree import DecisionTreeClassifier
classifier_DecisionTree = DecisionTreeClassifier()
classifier_DecisionTree.fit(X_train, y_train)

y_test_pred = classifier_DecisionTree.predict(X_test)
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")

from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test, y_test_pred)
print(confusion_mat)

'''A decision tree shows graphically the choices made or proposed. It does not 
happen so often that things are so clear that the choice between two solutions 
is immediate. Often, a decision is determined by a series of cascading 
conditions. Representing this concept with tables and numbers is difficult.
In fact, even if a table represents a phenomenon, it may confuse the reader 
because the justification for the choice is not obvious.'''











