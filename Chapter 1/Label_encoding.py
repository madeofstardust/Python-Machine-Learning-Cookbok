# Label Encoding
'''Label encoding refers to transforming word labels into a numerical form so that algorithms
can understand how to operate on them'''

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

input_classes = ['Moomintroll', 'Snufkin', 'Snorkmaiden', 'Little My', 'Snufkin',
'Little My']

label_encoder.fit(input_classes)
print("Class mapping: ")
for i, item in enumerate(label_encoder.classes_):
    print(item, "-->", i)

names = ['Snufkin', 'Snorkmaiden', 'Moomintroll']
#in original: 'encoded_labels;
encoded_names = label_encoder.transform(names)
print("Names =", names)
print("Encoded names =", list(encoded_names))

encoded_name = 3
decoded_name = label_encoder.inverse_transform([encoded_name])

print('Encoded name: ', encoded_name, '\n Decoded name: ', decoded_name)

'''Comparison of one-hot encoding and label encoding:
Label encoding can transform categorical data into numeric data, but the
imposed ordinality creates problems if the obtained values are submitted to
mathematical operations.
One-hot encoding has the advantage that the result is binary rather than ordinal,
and that everything is in an orthogonal vector space. The disadvantage is that for
high cardinality, the feature space can explode.
'''
