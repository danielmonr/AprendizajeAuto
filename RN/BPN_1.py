import numpy as np
import random
'''

'''

def bpnUnaNeurona (nn_params, input_layer_size, X, y, alpha, activacion):
    eps = 0.053
    iteraciones = 10000
    # forward-propagation
    global b
    b = np.zeros(y.size)
    for it in range(iteraciones):
        z = nn_params.dot(X) + b
        A = (sigmoidal(z) if activacion == "sigmoidal" else lineal(z))
        J = (sigmoidCosto(z,y) if activacion == "sigmoidal" else linealCosto(X,y,nn_params))
        #back-propagation
        dz = A-y
        #print("dz:", dz)
        if activacion == "sigmoidal":
            dw = (1/input_layer_size)*(X.dot(dz.T))
        else:
            dw = X.dot(dz.T)
        dw = np.squeeze(np.asarray(dw))
        db = (1/input_layer_size)*np.sum(dz)

        # updates
        nn_params = nn_params - (alpha * dw)
        b = b - (alpha*db)

        #print("J", J)
        if np.sum(J) < eps:
            print(np.sum(J), it)
            break
    return nn_params, b

def sigmoidal(x): # validated
    x = -1 * x
    s = np.exp(x)
    return 1/(1+s)

def sigmoidGradiente(z): # validated
    g = sigmoidal(z)
    return np.multiply(g,1-g)

def sigmoidCosto(X,y): # to be validated
    m = y.size
    #h = sigmoidal(W.dot(X))
    h = sigmoidal(X)
    #h = np.squeeze(np.asarray(X))
    #print("h:", h)
    j1 = (-1*y).dot(np.log(h).T)
    #print("j1:", j1)
    j2 = (1-y).dot(np.log(1-h).T)
    J =  j1 - j2
    return np.squeeze(np.asarray(J))

def lineal(z): # validated
    return z

def linealGradiante(z): # validated
    return 1

def linealCosto(X,y,W):
    h = W.dot(X)
    err = y-h
    m = y.size
    J = (err.dot(err.T))/(2*m)
    return J

def randInicializaPesos(L_in): #validated
    eps = 0.12
    w = np.random.uniform(-eps,eps,L_in)
    return w # shape:(L_in,)

def prediceRNYaEntrenada(X, nn_params, activacion, b):
    yy = nn_params.dot(X) + b[0]
    y = (sigmoidal(yy) if activacion == "sigmoidal" else lineal(yy))
    return y

# Testing vars and function calls
print("Testing")
x = np.matrix([[1,0], [0,1], [0,0], [1,1]]).T
y = np.array([0,0,0,1])
salidas_or = np.array([1,1,0,1])
print("x.shape:",x.shape)
print("x:", x)
print("y.shape:", y.shape)
print("y:", np.squeeze(np.asarray(y)))
print("Running functions")
w = randInicializaPesos(x.shape[0])
print("randinitW(y.size):", w)
print("w.shape:", w.shape)
print("lineal(2)", lineal(2))
print("linealGradiante(2)", linealGradiante(2))
print("sigmoidal(x)", sigmoidal(x))
print("sigmoidGradiente(x)", sigmoidGradiente(x))
print("sigmoidCosto(x,y)", sigmoidCosto(x,y))
w_l, b_l = bpnUnaNeurona(w,y.size,x,y,0.1,"lineal")
print("bpnUnaNeurona(w,y.size,x,y,0.1,lineal):\n", w_l, b_l)
w_s, b_s = bpnUnaNeurona(w,y.size,x,y,0.1,"sigmoidal")
print("bpnUnaNeurona(w,y.size,x,y,0.1,sigmoidal):\n", w_s, b_s)
print("prediceRNYaEntrenada([1,1], w_s, sigmoidal, b)", prediceRNYaEntrenada([1,1], w_s, "sigmoidal", b_s))
print("prediceRNYaEntrenada([1,1], w_s, lineal, b)", prediceRNYaEntrenada([1,1], w_l, "lineal", b_l))
