'''
Tutorial from
https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203

Algorithm
1. Select K random points as cluster centers call centroid
2. Assign each data point to the closet cluster by calculating it distance(manhatton/cartesian) with respect to each centroid
3. Determine the new cluster center by computing the verage of the assign point
4. Repeat 2, 3 until non of the cluster assignments change

Choosing right number of clusters using the relationship between the number of clusters and Within Cluster Sum of Squares (WCSS) then we select the number of clusters where the change in WCSS begins to level off (elbow method).

WCSS is defined as the sum of the squared distance between each member of the cluster and its centroid.
WSS = sum(xi-ci)**2
'''

# Import part

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs # for generating data
from sklearn.cluster import KMeans

# Generating data
x,y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
print('READ DATA 20 data--------------------')
for i in range(20):
    print('x_val {}  y_val  {}'.format(round(x[:,0][i],3),round(x[:,1][i],3)))

plt.scatter(x[:,0],x[:,1])
plt.show() # visualize

# Find optimum K using elbow method
# by storing KMeans.inertia_ (sum of square distances of samples to their closet cluster center ) in wcss list
# ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence
wcss = []
print('OPTIMUM K ----------------------')
for i in range(1,10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    print('K number  {}   wcss  {}'.format(i,round(kmeans.inertia_,3)))
plt.plot(range(1,10),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.show() # visualize

# After found out that 4 is the optimum number for K, plot the center of the cluster
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=500, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(x)
print('CENTROIDS ---------------------')
for i in range(len(kmeans.cluster_centers_[:,0])):
    print('x_val {}  y_val  {}'.format(round(kmeans.cluster_centers_[:,0][i],3),round(kmeans.cluster_centers_[:,1][i],3)))
plt.scatter(x[:,0],x[:,1])
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='red',alpha=0.5)
plt.show()

