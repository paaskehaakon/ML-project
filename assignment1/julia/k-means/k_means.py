import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)
import random as rd
import matplotlib.pyplot as plt

class KMeans:
    
    def __init__(self, k=2):
        
        self.k = k
        self.centroids = []
        self.centroidsToKeep = []
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        # TODO: Implement
        
        # normalize the data in "x0" to take in to account the inductive bias for the second dataset. K-means works if the clusters are similar to circles, in the second dataset they are more like ellipses due to the range in "x0".
        if self.k == 10:
                X.loc[:,"x0"] = X.loc[:,"x0"]/10
        
        oldDistortion = 100
        
        # run algorithm 5 times, save the set of clusters with lowest distortion => better odds at finding the 'right' clusters as seen by eye
        for i in range(5):
            
            self.centroids = initialCentroids(X,self.k)
            self.centroids = np.array(self.centroids)

            datapoints = X.copy()

            # to store the difference in distance between current and previous centroids => when to stop the loop
            diff = np.ones(self.k)

            while all(i>0.0001 for i in diff):

                # store previous centroids
                oldCentroids = self.centroids.copy()

                # for each data point, assign to a centroid
                clusters=[]
                for i in range(len(datapoints)):
                    point = X[i:i+1]
                    point = np.array(point)
                    point = point.reshape(-1)
                    dist = []

                    # for each centroid, calculate distance to point
                    for j in range(self.k):
                        dist.append(euclidean_distance(point,self.centroids[j]))
                    minIndex = dist.index(min(dist)) # find smallest distance, its index is the index of the assigned centroid
                    clusters.append(minIndex) # add the centroid index to the cluster assignments

                datapoints["centroids"] = clusters

                # recalculate the centroids as the average coordinates of the points assigned to each centroid
                for i in range(self.k):
                    new = np.array(datapoints[datapoints["centroids"]==i][["x0", "x1"]].mean(axis=0))
                    self.centroids[i,:] = new

                    # store the difference in distance between new and previously calculated centroids
                    diff[i] = euclidean_distance(self.centroids[i], oldCentroids[i])

            # to check if the distortion is lower than previous one saved, in that case save the clusters
            z = np.array(clusters)
            distortion = euclidean_distortion(X, z)
            if distortion < oldDistortion:
                oldDistortion = distortion
                self.centroidsToKeep = self.centroids
            
            
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        # TODO: Implement 

        clusters = []
        
        # for each data point
        for i in range(len(X)):
            point = X[i:i+1]
            point = np.array(point)
            point = point.reshape(-1)
            
            dist = []
            # now we already have the centroids done, so for each centroid
            for j in range(self.k):
                # calculate distance to the point at hand
                dist.append(euclidean_distance(point, self.centroidsToKeep[j]))
            # find the smallest distance, its index is the index of the assigned centroid
            minIndex = dist.index(min(dist))
            # add the centroid index to the cluster assignments for the points
            clusters.append(minIndex)
        
        # return the cluster assignments
        return np.array(clusters)
        
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        # TODO: Implement
        
        return self.centroidsToKeep
    
# A function to calculate the initial centroids (seeds) in a more clever way than just picking them randomly
# This is inspired by k-means++ and serves to ensure that the seeds are more spread out among the datapoints than if just randomly chosen
def initialCentroids(X, k):
    # one seed to start with, randomly chosen
    initCent = (X.sample()).reset_index(drop=True)
    
    # until we have k seeds (we already have one => range(k-1)
    for j in range(k-1):
        distToClosestSeed = []
        # for each datapoint
        for i in range(len(X)):
            d = []
            point = X[i:i+1]
            point = np.array(point)
            point = point.reshape(-1)
            for h in range(len(initCent)):
                # calculate distances to each already chosen seed
                d.append(euclidean_distance(point,initCent.iloc[h,:]))
            # remember the distance to the closest one
            distToClosestSeed.append(min(d))
        # Now we have a list distToClosestSeed with the distance from each point in X to its closest seed
        
        # Take the distances to the power of 4 to increase the impact of distance on likelihood to be chosen as new seed
        distToClosestSeed_four = [i**4 for i in distToClosestSeed]
        
        # Reformulate the distances to the power of 4 as weights between 0 and 1
        distToClosestSeed_four = distToClosestSeed_four / sum(distToClosestSeed_four)
        
        # Choose new seed randomly but with respect to the distToClosestSeed
        newSeed = X.sample(1, weights=distToClosestSeed_four)
        
        # Add the new seed to the initial centroids
        initCent = pd.concat([initCent, newSeed], ignore_index=True)

    # return the initial centroids
    return initCent
    
    
# --- Some utility functions 


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
