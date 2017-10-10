import numpy as np
import random
'''

'''

def bpnUnaNeurona (nn_params, input_layer_size, X, y, alpha, activacion):
    return 1

def sigmoidal(x): # validated
    x = -1 * x
    s = np.exp(x)
    return 1/(1+s)

def sigmoidGradiente(z): # validated
    g = sigmoidal(z)
    return np.multiply(g,1-g)

def sigmoidCosto(X,y,W): # to be validated
    m = y.size
    h = sigmoidal(X.dot(W))
    j1 = (-1*y).dot(np.log(h).T)
    j2 = (1-y).dot(np.log(1-h).T)
    J =  j1 - j2
    return np.squeeze(np.asarray(J))

def lineal(z): # validated
    return z

def linealGradiante(z): # validated
    return 1

def linealCosto(X,y,W):
    h = X.dot(W)
    err = y-h
    m = y.size
    J = (err.dot(err.T))/(2*m)
    return J

def randInicializaPesos(L_in): #validated
    eps = 0.12
    w = np.random.uniform(-eps,eps,L_in)
    return w # shape:(L_in,)

def prediceRNYaEntrenada(X, nn_params, activacion):
    y = 1
    return y

# Testing vars and function calls
print("Testing")
# matriz de X (4x3), (nxm), m numero de ejemplos para enrtenamiento
x = np.matrix([[0,0,0],[1,1,1],[2,2,2],[1,2,3]])
# matriz de y (4,), (nx1)
y = np.array([1,1,1,1])
print("x.shape:",x.shape)
print("x:", x)
print("y.shape:", y.shape)
print("y:", np.squeeze(np.asarray(y)))
print("Running functions")
w = randInicializaPesos(x.shape[1])
print("randinitW(y.size):", w)
print("w.shape:", w.shape)
print("lineal(2)", lineal(2))
print("linealGradiante(2)", linealGradiante(2))
print("sigmoidal(x)", sigmoidal(x))
print("sigmoidGradiente(x)", sigmoidGradiente(x))
print("sigmoidCosto(x,y)", sigmoidCosto(x,y,w))
