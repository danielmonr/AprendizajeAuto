import numpy as np
import matplotlib.pyplot as plt

entradas = np.matrix([[1,0], [0,1], [0,0], [1,1]])
salidas_and = np.array([0,0,0,1])
salidas_or = np.array([1,1,0,1])

theta = np.random.random(entradas.shape[1])*2
#print("theta init:\n", theta)
#print("X:\n", entradas)

def funcionCostoAdaline(theta,X,y):
    r = X.dot(theta.T)
    for i in np.arange(r.shape[1]):
        r[0,i] = (1 if r[0,i] >= 1 else 0)
        #print("r:\n", r[0,i])
    J = (r-y).dot((r-y).T)[0,0]
    grad = 0
    #J = 0
    return [J, grad]
    
def	entrenaAdaline(X, y, theta):
    #x = np.c_[np.ones(X.shape[0]), X[:,:]] # Add column of 1s
    #alpha = np.random.random_sample()
    alpha = 0.1
    cont = True
    e = 0
    #print("alpha:",alpha)
    while cont:
        Net = np.zeros(X.shape[0])
        out = np.zeros(X.shape[0])
        for it in np.arange(X.shape[0]):
            Net[it] = X[it].dot(theta.T)
            print("theta:",theta)
            print("Net:", Net[it])
            #print("theta:",theta)
        err = X.T.dot((Net-y))
        print("Test:", np.squeeze(err).shape)
        print("err: ", err.shape)
        theta = theta -((2/X.shape[0])*err)
        #print("\ntheta:\n", theta)
        print("\nErr:\n", err)
        cont = False
        
        for k in err:
            print(k)
            cont = (True if k*k > e*e else cont)
            
            
    #print ("\nout:\n", out)
    
    return theta
    
def prediceAdaline(theta, X):
    r = X.dot(theta.T)
    for i in np.arange(r.shape[0]):
        r[i] = (1 if r[i] >= 1 else 0)
    return r

print("costo 0: ", funcionCostoAdaline(entradas,theta,salidas_or)[0])

t = entrenaAdaline(entradas, salidas_or, theta)
samples = np.matrix([[1,1],[0,1],[0,0]])

print("costo 1: ", funcionCostoAdaline(entradas,t,salidas_or)[0])

print(prediceAdaline(t, samples))