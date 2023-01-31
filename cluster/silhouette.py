import numpy as np
from scipy.spatial.distance import cdist

'''
Silhouette score class for silhouette score algorithm. 
Calculates the silhouette score for each data point in a cluster.
Steps:
1. Compute mean distance between each point and all other points in same cluster
2. Compute mean distance between each point and all other points in next nearest cluster
3. Compute silhouette score for each point
'''
class mySilhouette:
    def __init__(self) -> None:
        self.silhouette = None

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''
        input: matrix of data points(2D), array of labels(1D)
        output: array of silhouette scores(1D)
        '''
        # Compute mean distance between each point and all other points in same cluster
        a = np.array([np.mean(cdist(X[y == i], X[y == i])) for i in np.unique(y)])
        # Compute mean distance between each point and all other points in next nearest cluster
        b = np.array([np.mean(cdist(X[y == i], X[y != i])) for i in np.unique(y)])
        # Compute silhouette score for each point
        self.silhouette = ((b - a) / np.maximum(a, b))
        return self.silhouette