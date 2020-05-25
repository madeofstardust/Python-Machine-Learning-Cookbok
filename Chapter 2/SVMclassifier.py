# SVM linear classifier

import numpy as np
import matplotlib.pyplot as plt
import utilities
# Load input data
input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)

# Load multivar data in the input file
def load_data(input_file):
    X = []
    y = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = [float(x) for x in line.split(',')]
            X.append(data[:-1])
            y.append(data[-1])
    X = np.array(X)
    y = np.array(y)
    return X, y

class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], facecolors='black',edgecolors='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], facecolors='None', edgecolors='black', marker='s')
plt.title('Input data')
plt.show()


# Train test split and SVM training
from sklearn import cross_validation
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25,
random_state=5)

params = {'kernel': 'linear'}
classifier = SVC(**params, gamma='auto')

classifier.fit(X_train, y_train)
utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()

y_test_pred = classifier.predict(X_test)
utilities.plot_classifier(classifier, X_test, y_test, 'Test dataset')
plt.show()

#Accuracy:
from sklearn.metrics import classification_report
target_names = ['Class-' + str(int(i)) for i in set(y)]
print("\n" + "#"*30)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train),
target_names=target_names))
print("#"*30 + "\n")
      
#Classification report:
print("#"*30)
print("\nClassification report on test dataset\n")
print(classification_report(y_test, y_test_pred,
target_names=target_names))
print("#"*30 + "\n")

