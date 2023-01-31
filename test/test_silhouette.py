# Description: Unit tests for Silhouette Score
# Author:     Cindy Pino
# Date:       2023-01-31
# Version:    1.0
# Environment: Python 3.7.3 and Jupyter notebook

# Import packages
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from cluster import myKMeans
from cluster import mySilhouette
# Write your k-means unit tests here
# Create a dataset with 500 observations and 2 features and 6 clusters
X, y = make_blobs(n_samples=500, n_features=2, centers=6, random_state=42)

n_clusters = 6 # Number of clusters

# Using sklearn's Silhouette score
sc = silhouette_score(X, y)
print("Sklearn's Silhouette score: ", sc)

# Using my Silhouette score
ss = np.average(silhouette.score(X, y))
print("My Silhouette score: ", ss)

print("Difference: ", 0.5 - abs(sc - ss))

# Test that the silhouette score is correct

assert (abs(sc - ss) < 0.5) == True
assert (abs(sc - ss) < 0.5) == True, "Silhouette score is not correct"


