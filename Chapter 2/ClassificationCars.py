#Classification Example: cars.
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

#data:
input_file = 'car.data.txt'
# Reading the data
X = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)
X = np.array(X)

# Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i]) 
    #we ignore the last char, it's a newline character.
X = X_encoded[:, :-1].astype(int)
#The last value on each line is the class, so we assign it to the y variable.
y = X_encoded[:, -1].astype(int)

# Build a Random Forest classifier
params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}
classifier = RandomForestClassifier(**params)
classifier.fit(X, y)

# Cross validation
from sklearn import model_selection
accuracy = model_selection.cross_val_score(classifier,
X, y, scoring='accuracy', cv=3)
print("Accuracy of the classifier: " +
str(round(100*accuracy.mean(), 2)) + "%")

# Testing encoding on single data instance
input_data = ['high', 'low', '2', 'more', 'med', 'high']
input_data_encoded = [-1] * len(input_data)
for i,item in enumerate(input_data):
    input_data_encoded[i] = int(label_encoder[i].transform([input_data[i]]))
input_data_encoded = np.array(input_data_encoded)

# Predict and print output for a particular datapoint
#We need to use the label encoders that we used during training because we want it to be consistent
output_class = classifier.predict([input_data_encoded])
print("Output class:", label_encoder[-1].inverse_transform(output_class)[0])


# Validation curves
'''Validation curves help us understand how each hyperparameter (n_estimators 
and max_depth) influences the training score. Basically, all other parameters 
are kept constant and we vary the hyperparameter of interest according to our 
range.'''
from sklearn.model_selection import validation_curve

classifier = RandomForestClassifier(max_depth=4, random_state=7)
parameter_grid = np.linspace(25, 200, 8).astype(int)
train_scores, validation_scores = validation_curve(classifier,
                                                   X,
                                                   y,
                                                   "n_estimators",
                                                   parameter_grid,
                                                   cv=5)
print("##### VALIDATION CURVES #####")
print("\nParam: n_estimators\nTraining scores:\n", train_scores)
print("\nParam: n_estimators\nValidation scores:\n", validation_scores)

# Plot the curve
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1),
color='black')
plt.title('Training curve')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()

# the same with max_depth:
classifier = RandomForestClassifier(n_estimators=20, random_state=7)
parameter_grid = np.linspace(2, 10, 5).astype(int)
train_scores, valid_scores = validation_curve(classifier,
                                              X,
                                              y,
                                              "max_depth",
                                              parameter_grid,
                                              cv=5)
print("\nParam: max_depth\nTraining scores:\n", train_scores)
print("\nParam: max_depth\nValidation scores:\n",
validation_scores)

# Plot the curve
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1),
color='black')
plt.title('Validation curve')
plt.xlabel('Maximum depth of the tree')
plt.ylabel('Accuracy')
plt.show()

#Learning curve
'''Learning curves help us understand how the size of our training dataset 
influences the machine learning model.'''
from sklearn.model_selection import validation_curve
classifier = RandomForestClassifier(random_state=7)
parameter_grid = np.array([200, 500, 800, 1100])
train_scores, validation_scores = validation_curve(classifier,
                                                   X,
                                                   y,
                                                   "n_estimators",
                                                   parameter_grid,
                                                   cv=5)
print("\n##### LEARNING CURVES #####")
print("\nTraining scores:\n", train_scores)
print("\nValidation scores:\n", validation_scores)

# Plot the curve
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1),
color='black')
plt.title('Learning curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()
'''Although smaller training sets seem to give better accuracy, they are prone to
overfitting. If we choose a bigger training dataset, it consumes more resources.
Therefore, we need to make a trade-off here to pick the right size for the training
dataset.'''
'''A learning curve allows us to check whether the addition of training data 
leads to a benefit. It also allows us to estimate the contribution deriving 
from variance error and bias error. If the validation score and the training 
score converge with the size of the training set too low, we will not benefit 
from further training data.'''


