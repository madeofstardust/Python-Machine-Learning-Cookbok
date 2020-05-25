# One-hot encoding
'''One-hot encoding a tool that tightens feature vectors. It looks at
each feature and identifies the total number of distinct values. It uses a one-of-k scheme to
encode values. Each feature in the feature vector is encoded based on this scheme. This
helps us to be more efficient in terms of space. '''

import numpy as np
#Using sklearn:
from sklearn import preprocessing

data = np.array([[1, 1, 2], [0, 2, 3], [1, 0, 1], [0, 1, 0]])
print(data)

encoder = preprocessing.OneHotEncoder(categorical_features=None, categories=None, drop=None, handle_unknown='error',
         n_values=None, sparse=True)
encoder.fit(data)

encoded_vector = encoder.transform([[1, 2, 3]]).toarray()
print(encoded_vector)