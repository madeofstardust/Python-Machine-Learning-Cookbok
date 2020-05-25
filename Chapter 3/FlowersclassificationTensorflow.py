# -*- coding: utf-8 -*-
#Flowers classification using tensorflow

from sklearn import datasets
from sklearn import model_selection
import tensorflow as tf

iris  = datasets.load_iris()

x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data,
                                                                    iris.target,
                                                                    test_size=0.7,
                                                                    random_state=1)

#Simple NN using tensorflow:
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train)
classifier_tf = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                               hidden_units=[10],
                                               n_classes=3)


