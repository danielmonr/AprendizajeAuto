import numpy
import matplotlib.pyplot as plt

def leerDatos(file): #Funcion para leer el archivo de datos iniciales.
    points = numpy.loadtxt(file, delimiter=',') # Save all the file's points in a Matrix
    cols = points.shape[1]
    x = numpy.c_[numpy.ones(points.shape[0]), points[:, numpy.arange(0,cols-1)]]# Create Matrix of Xs with 1s
    #x = points[:, numpy.arange(0,cols-1)]
    y = numpy.c_[points[:, cols-1]] # Create Y Vector, as a Matrix
    #print(x.shape)
    return x,y

def sigmoidal(z):
    s = numpy.exp(z) #exponencial
    return 1/(1+s) #regresar la sigmoidal de z

def funcionCosto(theta,X,y):
    hyp = sigmoidal(X.dot(theta)*-1)# hypothesis
    J = -1/(y.size) * ((numpy.log(hyp).T.dot(y)) + (numpy.log(1-hyp).T.dot(1-y))) #Costo
    grad = (1.0/y.size) * (numpy.dot(X.T,(hyp - y))) #gradiente
    return J, grad

def aprende(theta,X,y,iteraciones): # Función que regresa el valor de Thetas de acuerdo al algoritmo de gradiente descendiente.
    m = y.size # Numero de datos
    alpha = 0.003 # Grado de aprendizaje
    for i in range(0,iteraciones):
        hyp = sigmoidal(X.dot(theta)*-1) #hypothesis
        #print(hyp)
        theta = theta - alpha * (1.0/m) * (numpy.dot(X.T,(hyp - y))) # Theta vector for each iteration
    return theta

def predice(theta,X): # Función que recive una matriz de thetas, y una matriz de Xs y arroja una matriz de predicción de aprueba o no.
    x = numpy.c_[numpy.ones(X.shape[0]), X[:]]
    threshold = 0.5
    p = sigmoidal(x.dot(theta)*-1) >= threshold
    return(p.astype(int))

def graficaDatos(X,y,theta): #función para graficar los ejemplos de la población y la recta dada por el vector Theta.
    points = numpy.c_[X[:,:], y[:,:]]
    non = points[:,3] == 0 # puntos de los no admitidos
    admi = points[:,3] == 1 # puntos de los admitidos
    Opoints = ((numpy.arange(0,100) * (-1*theta[1])) - theta[0])/theta[2] # puntos de la recta creada por Theta de 0 a 100
    plt.plot(numpy.arange(0,100), Opoints, 'r')
    axis = plt.gca()
    axis.scatter(points[admi][:,1], points[admi][:,2], marker='X', c ='k', s=60, label='Aditted')
    axis.scatter(points[non][:,1], points[non][:,2], marker='o', c ='g', s=60, label='Non-Aditted')
    axis.set_xlabel('Ex1')
    axis.set_ylabel('Ex2')
    plt.show()



X,Y = leerDatos("./ex1data2.txt")
t = aprende(numpy.zeros((X.shape[1],1)), X,Y, 2000000)
j,g = funcionCosto(t,X,Y)

res = predice(t,numpy.array([[100,100]]))

print("Costo:")
print(j)
print("Vector Theta:")
print(t)
print("Prediccion de [[45,85]]")
print (res)

graficaDatos(X,Y,t)
