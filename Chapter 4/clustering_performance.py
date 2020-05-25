#clustering evaluation
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

input_file = ('data_perf.txt')
x = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        x.append(data)
        data = np.array(x)

scores = []
range_values = np.arange(2, 10)
for i in range_values:
    # Train the model
    kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
    kmeans.fit(data)
    #Score:
    score = metrics.silhouette_score(data, kmeans.labels_,
    metric='euclidean', sample_size=len(data))
    print("Number of clusters =", i)
    print("Silhouette score =", score)
    scores.append(score)
    
# Plot scores
plt.figure()
plt.bar(range_values, scores, width=0.6, color='k', align='center')
plt.title('Silhouette score vs number of clusters')
# Plot data
plt.figure()
plt.scatter(data[:,0], data[:,1], color='k', s=30, marker='o',
facecolors='none')
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

'''The sklearn.metrics.silhouette_score function computes the mean silhouette
coefficient of all the samples. For each sample, two distances are calculated: 
the mean intracluster distance (x), and the mean nearest-cluster distance (y). 
The silhouette coefficient for a sample is given by the following equation:
    score = (x-y)/max(x,y)
    where x is a mean distance between the given point and points in its clusters,
    and y - between it and points in the closest other cluster.
    The best value is 1, and the worst value is -1.'''

