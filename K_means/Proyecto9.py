import numpy as np
import matplotlib.pyplot as plt

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
    k = initial_centroids
    ''' Calclate distance from points to clusters'''
    dist = np.empty([X.shape[0], 0])
    for i in range(initial_centroids.shape[0]):
        dist = np.c_[dist, np.linalg.norm(X-k[i,:], axis=1)]
    #print("dist.shape:", dist.shape)
    idx = np.argmin(dist, axis=1) + 1
    #print("idx.shape:", idx.shape)
    return idx

def computeCentroids(X, idx, K):
    means = np.zeros((K.shape[0], X.shape[1]))
    means_y = np.zeros((K.shape[0]))
    conts = np.zeros((K.shape[0],1))
    cont = 0
    for it in idx:
        #means[it-1] = means[it-1] + np.linalg.norm(X[cont,:]-K[it-1,:], axis = 0)
        means[it-1] = means[it-1] + (X[cont, :])
        conts[it-1] += 1
        cont += 1
    means = np.divide(means, conts)
    #print("means.shape:", means.shape)
    return means

def runkMeans(X, initial_centroids, max_iters, true=False):
    ks = kMeansInitCentroids(X, initial_centroids)
    h = []
    iteraciones = 10
    diff = 0
    #print("X.shape:", X.shape)
    #print("ks.shape:", ks.shape)
    for it in range(iteraciones):
        ks_n = computeCentroids(X,findClosestCentroids(X,ks), ks)
        diff = np.sum(ks - ks_n)
        ks = ks_n
        h.append(ks.tolist())
        if diff == 0:
            break
    #print("len(h)", len(h[0]))
    if (true):
        idx = findClosestCentroids(X,ks)
        for i in range(ks.shape[0]):
            list = [[] for x in range(2)]
            ps = [[] for x in range(2)]
            for j in range(len(h)):
                list[0].append(h[j][i][0])
                list[1].append(h[j][i][1])
            plt.plot(list[0], list[1])
            cont = 0
            for itx in idx:
                if itx == i+1:
                    ps[0].append(X[cont][0])
                    ps[1].append(X[cont][1])
                cont += 1
            plt.plot(ps[0], ps[1], 'x')
            #print(list)

        plt.show()
    return ks

print("Centroids:",runkMeans(read_file("./newData.txt"), 6, 10, True))
