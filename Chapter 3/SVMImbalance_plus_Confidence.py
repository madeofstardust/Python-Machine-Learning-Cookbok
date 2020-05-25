# Imbalance
'''Until now, we dealt with problems where we had a similar number of datapoints
in all our classes. In the real world, we might not be able to get data in such 
an orderly fashion. Sometimes, the number of datapoints in one class is a lot 
more than the number of datapoints in other classes. If this happens, then the 
classifier tends to get biased. The boundary won't reflect the true nature of 
your data, just because there is a big difference in the number of datapoints 
between the two classes. Therefore, it is important to account for this 
discrepancy and neutralize it so that our classifier remains impartial.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import utilities

input_file = 'data_multivar_imbalance.txt'
X, y = utilities.load_data(input_file)

# Separate the data into classes based on 'y'
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])
# Plot the input data
plt.figure()

plt.scatter(class_0[:,0], class_0[:,1], 
            facecolors='black', 
            edgecolors='black', 
            marker='s')
plt.scatter(class_1[:,0], class_1[:,1], 
            facecolors='None',
            edgecolors='black', 
            marker='s')
plt.title('Input data')
plt.show()


from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    test_size=0.25, 
                                                                    random_state=5)

'''The class_weight parameter will count the number of datapoints in each class
to adjust the weights so that the imbalance doesn't adversely affect the
performance.'''
'''Probability counting '''
params = {'kernel': 'linear', 'class_weight':'balanced', 'probability': True}
classifier = SVC(**params, gamma='auto')
classifier.fit(X_train, y_train)
utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()

'''C is a hyperparameter that determines the penalty for the incorrect 
classification of an observation. So, we used a weight for the classes to manage 
unbalanced classes. In this way, we will assign a new value of C to the classes, 
defined as follows: 
    C(i) = C * w(i)
Where C is the penalty, w(i) is a weight inversely  proportional to class i's 
frequency, and C(i) is the C value for class i. This method suggests increasing 
the penalty to classify the less represented classes  so as to prevent them 
from being outclassed by the most represented class. In the scikit-learn library, 
when using SVC, we can set the values for Ci  automaticallyby setting 
class_weight='balanced'.'''

print("Confidence measure:")
for i in class_0:
    print(i, '-->', classifier.predict_proba([i])[0])
    
utilities.plot_classifier(classifier, 
                          class_0, 
                          [0]*len(class_0),
                          'Input datapoints',
                          'False')

'''When estimating a parameter, the simple identification of a single value is 
often not sufficient. It is therefore advisable to accompany the estimate of a 
parameter with a plausible range of values for that parameter, which is defined 
as the confidence interval. It is therefore associated with a cumulative 
probability value that indirectly, in terms of probability, characterizes its 
amplitude with respect to the maximum values assumed by the random variable that 
measures the probability that the random event described by that variable in 
question falls into this interval and is equal to this area graphically, 
subtended by the probability distribution curve of the random variable in that 
specific interval.'''