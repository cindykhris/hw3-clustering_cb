"""
BMI203: Biocomputing algorithms Winter 2022
Assignment 3: Clustering and Silhouette Score
"""

from .kmeans import myKMeans
from .silhouette import mySilhouette
from .utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)

__version__ = "0.1.1"
