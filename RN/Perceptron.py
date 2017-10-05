import numpy as np
import matplotlib.pyplot as plt

entradas = np.matrix([[1,0], [0,1], [0,0], [1,1]])
salidas_and = np.array([0,0,0,1])
salidas_or = np.array([1,1,0,1])

theta = np.random.random(entradas.shape[1])*2
print("theta init:\n", theta)
print("X:\n", entradas)

def funcionCostoPerceptron(theta,X,y):
    J = 0
    grad = 0
    return [J, grad]
    
def	entrenaPerceptron(X, y, theta):
    #x = np.c_[np.ones(X.shape[0]), X[:,:]] # Add column of 1s
    alpha = np.random.random_sample()
    cont = True
    e = 0
    wcont = 0
    print("alpha:",alpha)
    while cont:
        wcont+=1
        Net = np.zeros(X.shape[0])
        out = np.zeros(X.shape[0])
        for it in np.arange(X.shape[0]):
            Net[it] = X[it].dot(theta.T)
            out[it] = (1 if Net[it] >= 1 else 0)
            duo = alpha*(y[it]-out[it])*X[it]
            #print("theta:",theta)
            #print("duo:", duo)
            theta = theta + duo
            #print("theta:",theta)
        err = y-out
        #print("\ntheta:\n", theta)
        #print("\nErr:\n", err)
        cont = False
        for k in err:
            cont = (True if k*k > e*e else cont)
            
            
    print ("\nout:\n", out)
    print("iteraciones:", wcont)
    
    return theta
    
def predicePerceptron(theta, X):
    p = []
    return p
    

print (entrenaPerceptron(entradas, salidas_and, theta))