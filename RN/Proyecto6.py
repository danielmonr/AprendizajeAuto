import numpy as np
import matplotlib.pyplot as plt

# Definición de variables de ayuda
entradas = np.matrix([[1,0], [0,1], [0,0], [1,1]]) #matriz de entradas de una tabla de verdad, de dos variables
salidas_and = np.array([0,0,0,1]) #matriz de salidas de operacion AND
salidas_or = np.array([1,1,0,1]) #matriz de salidas de operacion OR

theta = np.random.random(entradas.shape[1])*2 # #thetas iniciales, pesos

# PERCEPTRON

#funcion que calcula el costo de una iteracion de Perceptron
def funcionCostoPerceptron(theta,X,y):
    r = X.dot(theta.T)
    for i in np.arange(r.shape[1]):
        r[0,i] = (1 if r[0,i] >= 1 else 0)
    J = y-r
    grad = 0.5*J*X
    return [J, grad]

#Funcion que entrena un Perceptron
def	entrenaPerceptron(X, y, theta):
    alpha = 0.5
    cont = True
    e = 0.01
    h_err = []
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

#Funcion que predice las salidas de una matriz de entradas, utilizando Perceptron
def predicePerceptron(theta, X):
    r = X.dot(theta.T)
    for i in np.arange(r.shape[0]):
        r[i] = (1 if r[i] >= 1 else 0)
    return r

# ADALINE

# fucion que calcula el costo de una iteración de Adaline
def funcionCostoAdaline(theta,X,y):
    alpha = 0.001
    r = (X.dot(theta)).tolist()[0]
    J = (y-r)[0]
    grad = (alpha*((y-r)*X)).tolist()[0]
    return [J, grad]

#Funcion que entrena el PE de Adaline
def	entrenaAdaline(X, y, theta):
    cont = True
    e = 0.042
    h_err = []
    while cont:
        ct = 0
        for it in np.arange(X.shape[0]):
            c, grad = funcionCostoAdaline(theta, X[it,:],y[it])
            theta = theta + grad
            ct += c*c
        ct = ct / (2*X.shape[0])
        h_err.append(ct)
        cont = (True if ct > e else False)
    plt.plot(h_err)
    plt.show()
    return theta

#Funcion que predice los resultados segun una matriz de entradas, utilizando Adaline
def prediceAdaline(theta, X):
    r = X.dot(theta.T).tolist()[0]
    for i in np.arange(len(r)):
        r[i] = (1 if r[i] >= 0.5 else 0)
    return r


#Llamadas de Ejemplo
t = entrenaPerceptron(entradas, salidas_and, theta)
print("Theta Perceptron", t)
samples = np.matrix([[1,1],[0,1],[0,0]])
print("Prediccion con Perceptron, de las muestras:\n", samples,"_____", predicePerceptron(t, samples))
theta = np.zeros(entradas.shape[1])
t = entrenaAdaline(entradas, salidas_and, theta)
print("Prediccion con Adaline, de las muestras:\n", samples,"_____", prediceAdaline(t, samples))
