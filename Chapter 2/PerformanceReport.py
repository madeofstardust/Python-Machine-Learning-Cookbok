#Performance report:
'''This function builds a text report showing the main classification metrics. 
A text summary of the precision, recall, and the F1 score for each class
is returned.'''
from sklearn.metrics import classification_report

y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]
target_names = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
print(classification_report(y_true, y_pred, target_names=target_names))