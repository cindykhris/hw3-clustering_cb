import numpy as np
from scipy.spatial.distance import cdist

"""
myKMeans class for k-means clustering algorithm
Steps:
1. Initialize centroids by first shuffling the dataset and then randomly
    selecting k data points for the centroids without replacement.
2. Keep iterating until there is no change to the centroids. i.e assignment
    of data points to clusters isnt changing.
    a. Compute the sum of the squared distance between data points and all
        centroids.
    b. Assign each data point to the closest cluster (centroid).
    c. Compute the centroids for the clusters by taking the average of the
        all data points that belong to each cluster.

Parameters
----------
k : int
    Number of clusters
tol : float
    Tolerance for stopping criteria
max_iter : int
    Maximum number of iterations for algorithm to converge

Attributes
----------
centroids : array, shape = [k, n_features]
    Centroids found at the last iteration of k-means algorithm
labels : array, shape = [n_samples,]
    Cluster labels for each point
error : float
    Sum of squared distances of samples to their closest cluster center.

Methods
-------
fit(X)
    Compute k-means clustering.
"""

class myKMeans:

    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100, random_state=42):
        '''
        Initialize required parameters
        '''
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.error = None

    def fit(self, mat: np.ndarray) -> None:
        '''
        ERROR CHECKING:
        '''
        if self.k > mat.shape[0]:
            raise ValueError("Number of clusters cannot be greater than number of data points")
        if self.k < 1:
            raise ValueError("Number of clusters must be greater than 0")
        if self.tol < 0:
            raise ValueError("Tolerance must be greater than 0")
        if self.max_iter < 1:
            raise ValueError("Maximum number of iterations must be greater than 0")

        '''
        ALGORITHM:
        input: matrix of data points(2D), rows are observations, columns are features
        '''

        # Initialize centroids
        centroids = mat[np.random.choice(mat.shape[0], self.k, replace=False), :]
        for i in range(self.max_iter):
            # Compute distance between each point and centroid
            distances = cdist(mat, centroids)
            # Assign each point to closest centroid
            labels = np.argmin(distances, axis=1)
            # Compute new centroids
            new_centroids = np.array([mat[labels == i].mean(axis=0) for i in range(self.k)])
            # Check for convergence
            if np.allclose(centroids, new_centroids, atol=self.tol):
                break
            centroids = new_centroids
        self.__centroids = centroids
        self.__labels = labels
        self.__error = np.sum(np.min(cdist(mat, self.__centroids), axis=1))

    def predict(self, mat: np.ndarray) -> np.ndarray:
        '''
        predition function: returns the cluster labels for each data point
        input: matrix of data points(2D)
        output: array of labels(1D)
        '''
        return np.argmin(cdist(mat, self.__centroids), axis=1)
        

    def __get_error(self) -> float:
        return self.__error

    def get_centroids(self) -> np.ndarray:
        return self.__centroids

