import numpy as np

def read_file(f):
    points = np.loadtxt(f, delimiter=' ')
    return points

def kMeansInitCentroids(X, K):
    ma = np.amax(X, axis=0)
    mi = np.amin(X, axis=0)
    xs = np.random.uniform(mi[0], ma[0], K)
    ys = np.random.uniform(mi[1], ma[1], K)
    ks = np.c_[xs, ys]
    return ks

def findClosestCentroids(X, initial_centroids):
    idx = 0
    k = initial_centroids
    ''' Calclate distance from points to clusters'''
    dist = np.empty([X.shape[0], 0])
    for i in range(initial_centroids.shape[0]):
        dist = np.c_[dist, np.linalg.norm(X-k[i,:], axis=1)]
    print("dist.shape:", dist.shape)
    return idx

def computeCentroids(X, idx, K):
    return 0

def runkMeans(X, initial_centroids, max_iters, true=False):
    ks = kMeansInitCentroids(X, initial_centroids)
    print("X.shape:", X.shape)
    print("ks.shape:", ks.shape)
    findClosestCentroids(X,ks)
    return 0

runkMeans(read_file("./newData.txt"), 3, 10)
