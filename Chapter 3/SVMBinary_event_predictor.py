# -*- coding: utf-8 -*-
#Binary Event predictor SVM
#Is there some event going on inside the building (taking onto account people 
#coming in and out)

import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC

#Loading data
input_file = 'building_event_binary.txt'
# Reading the data
X = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append([data[0]] + data[2:])

X = np.array(X)

# Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Build SVM
params = {'kernel': 'rbf', 'probability': True, 'class_weight':'balanced'}
classifier = SVC(**params, gamma='auto')
classifier.fit(X, y)

from sklearn import model_selection

accuracy = model_selection.cross_val_score(classifier,
X, y, scoring='accuracy', cv=3)
print("Accuracy of the classifier: " + 
      str(round(100*accuracy.mean(), 2)) + "%")

# Testing encoding on single data instance
input_data = ['Tuesday', '12:30:00','21','23']
input_data_encoded = [-1] * len(input_data)
count = 0
for i,item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count = count + 1
input_data_encoded = np.array(input_data_encoded)
# Predict and print(output for a particular datapoint
output_class = classifier.predict([input_data_encoded])
print("Output class:", 
      label_encoder[-1].inverse_transform(output_class)[0])





