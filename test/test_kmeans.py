# Description: Unit tests for k-means clustering
# Author:     Cindy Pino
# Date:       2023-01-31
# Version:    1.0
# Environment: Python 3.7.3 and Jupyter notebook

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from cluster import myKMeans  # Import myKMeans class
# import warnings filter
from warnings import simplefilter

# Write your k-means unit tests here
# Create a dataset with 500 observations and 2 features and 6 clusters
X, y = make_blobs(n_samples=500, n_features=2, centers=6, random_state=42)

n_clusters = 6 # Number of clusters

# Using sklearn's KMeans

kmeans = KMeans(n_clusters, random_state=42)
kmeans.fit(X)
kmeans_labels = kmeans.predict(X)
kmeans_centroids = kmeans.cluster_centers_


# Using myKMeans class

mykmeans = myKMeans(n_clusters)
mykmeans.fit(X)
mykmeans_labels = mykmeans.predict(X)
mykmeans_centroids = mykmeans.get_centroids()

#################################################
# Test 1
# Test that the number of labels is correct
#################################################
assert (len(kmeans_labels) == len(mykmeans_labels))
assert len(mykmeans_labels) != len(mykmeans_labels), "The number of labels is incorrect."

#################################################
# Test 2
# Test that the number of centroids is correct
#################################################
assert len(mykmeans_centroids) == n_clusters
assert len(mykmeans_centroids) != n_clusters, "The number of centroids is incorrect.  It should match n_cluster."

#################################################
# Test 3
# Test that the number of features in the centroids is correct
#################################################
assert len(mykmeans_centroids[0]) == len(kmeans.cluster_centers_[0])
assert len(mykmeans_centroids[0]) != len(kmeans.cluster_centers_[0]), "The number of features in the centroids is incorrect.  It should be 2."

#################################################
# Test 4
# Test that the labels are correct
#################################################
assert len(mykmeans_labels) == len(kmeans_labels) 
assert len(mykmeans_labels) != len(kmeans_labels), "The labels are incorrect."

#################################################
# Test 5
# Test that the centroids are correct
#################################################
assert np.array_equal(mykmeans_centroids[3][0].round(4), kmeans_centroids[2][0].round(4))
assert np.array_equal(mykmeans_centroids[3][0].round(4), kmeans_centroids[2][0].round(4)), "The centroids are incorrect."

print("Success!!!")