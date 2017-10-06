import numpy as np
import matplotlib.pyplot as plt

entradas = np.matrix([[1,0], [0,1], [0,0], [1,1]])
salidas_and = np.array([0,0,0,1])
salidas_or = np.array([1,1,0,1])

theta = np.zeros(entradas.shape[1])


def funcionCostoAdaline(theta,X,y):
    alpha = 0.001
    #print(y)
    r = (X.dot(theta)).tolist()[0]
    #import pdb; pdb.set_trace()
    J = (y-r)[0]
    grad = (alpha*((y-r)*X)).tolist()[0]
    #J = 0
    return [J, grad]

def	entrenaAdaline(X, y, theta):
    cont = True
    e = 0.042
    h_err = []
    while cont:
        ct = 0
        for it in np.arange(X.shape[0]):
            c, grad = funcionCostoAdaline(theta, X[it,:],y[it])
            #print("Grad: ", grad)
            #print(theta)
            theta = theta + grad
            #print(theta, "---")
            #print("c:",c)
            ct += c*c
        ct = ct / (2*X.shape[0])
        h_err.append(ct)
        #print("CT: ", ct)
        cont = (True if ct > e else False)
    plt.plot(h_err)
    plt.show()
    return theta

def prediceAdaline(theta, X):
    r = X.dot(theta.T).tolist()[0]
    #print(r)
    for i in np.arange(len(r)):
        r[i] = (1 if r[i] >= 0.5 else 0)
    return r

#print("costo 0: ", funcionCostoAdaline(theta, entradas,salidas_or)[0])

t = entrenaAdaline(entradas, salidas_and, theta)
print ("Theta: ", t)
samples = np.matrix([[1,1],[0,1],[0,0]])

#print("costo 1: ", funcionCostoAdaline(t, entradas,salidas_or)[0])

print(prediceAdaline(t, samples))
