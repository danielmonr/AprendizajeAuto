import numpy as np
import matplotlib.pyplot as plt

entradas = np.matrix([[1,0], [0,1], [0,0], [1,1]])
salidas_and = np.array([0,0,0,1])
salidas_or = np.array([1,1,0,1])

theta = np.random.random(entradas.shape[1])*2
#print("theta init:\n", theta)
#print("X:\n", entradas)

def funcionCostoPerceptron(theta,X,y):
    r = X.dot(theta.T)
    for i in np.arange(r.shape[1]):
        r[0,i] = (1 if r[0,i] >= 1 else 0)
        #print("r:\n", r[0,i])
    J = y-r
    #print(J.shape)
    grad = 0.5*J*X
    #J = 0
    return [J, grad]

def	entrenaPerceptron(X, y, theta):
    #x = np.c_[np.ones(X.shape[0]), X[:,:]] # Add column of 1s
    #alpha = np.random.random_sample()
    alpha = 0.5
    cont = True
    e = 0.01
    h_err = []
    #print("alpha:",alpha)
    while cont:
        Net = np.zeros(X.shape[0])
        out = np.zeros(X.shape[0])
        ct = 0
        for it in np.arange(X.shape[0]):
            err, g = funcionCostoPerceptron(theta, X[it, :], y[it])
            theta = theta + g
            ct += err[0,0]
        h_err.append(ct)
        cont = (True if abs(ct) > e else False)
    plt.plot(h_err)
    plt.show()
    return theta

def predicePerceptron(theta, X):
    r = X.dot(theta.T)
    for i in np.arange(r.shape[0]):
        r[i] = (1 if r[i] >= 1 else 0)
    return r

t = entrenaPerceptron(entradas, salidas_and, theta)
samples = np.matrix([[1,1],[0,1],[0,0]])

print(predicePerceptron(t, samples))
