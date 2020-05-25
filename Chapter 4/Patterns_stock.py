# Finding patterns in stock market data
'''affinity propagation (AP) - tries to find a representative datapoint for 
each cluster in our data, along with measures of similarity between pairs of 
datapoints, and considers all our datapoints as potential representatives, 
also called exemplars, of their respective clusters.'''

'''Analyze the stock market variations of companies over a specified duration - 
the goal is to then find out what companies behave similarly in terms of their
quotes over time.'''

import json
import sys
import pandas as pd
import numpy as np
from sklearn import covariance, cluster

# Input symbol file
symbol_file = 'symbol_map.json'

# Load the symbol map
with open(symbol_file, 'r') as f:
    symbol_dict = json.loads(f.read())
symbols, names = np.array(list(symbol_dict.items())).T

quotes = []
excel_file = 'stock_market_data.xlsx'
for symbol in symbols:
    print('Quote history for %r' % symbol, file=sys.stderr)
    quotes.append(pd.read_excel(excel_file, symbol))

# Extract opening and closing quotes
opening_quotes = np.array([quote.open for quote in
quotes]).astype(np.float)

closing_quotes = np.array([quote.close for quote in
quotes]).astype(np.float)
# The daily fluctuations of the quotes
delta_quotes = closing_quotes - opening_quotes

#Build a graph model from the correlations
edge_model = covariance.GraphicalLassoCV(cv=3)

# Standardize the data
X = delta_quotes.copy().T
X /= X.std(axis=0)

# Train the model
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

#Build clustering model using affinity propagation
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()
# Print the results of clustering
for i in range(num_labels + 1):
    print ("Cluster", i+1, "-->", ', '.join(names[labels == i]))


    
    
