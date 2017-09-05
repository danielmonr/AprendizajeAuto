import numpy
import matplotlib.pyplot as plt

def leerDatos(file):
    points = numpy.loadtxt(file, delimiter=',') # Save all the file's points in a Matrix
    cols = points.shape[1]
    #x = numpy.c_[numpy.ones(points.shape[0]), points[:, numpy.arange(0,cols-1)]]# Create Matrix of Xs with 1s
    x = points[:, numpy.arange(0,cols-1)]
    y = numpy.c_[points[:, cols-1]] # Create Y Vector, as a Matrix
    #print(x.shape)
    return x,y

def sigmoidal(z):
    #print(z)
    s = numpy.exp(z)
    #print(s)
    return 1/(1+s)

def funcionCosto(theta,X,y):
    hyp = sigmoidal(X.dot(theta))
    J = -1/(y.size) * ((numpy.log(hyp).T.dot(y)) + (numpy.log(1-hyp).T.dot(1-y)))
    grad = (1.0/y.size) * (numpy.dot(X.T,(hyp - y)))
    return J, grad

def aprende(theta,X,y,iteraciones):
    m = y.size # Numero de datos
    alpha = 0.01
    for i in range(0,iteraciones):
        hyp = sigmoidal(X.dot(theta))
        #print(hyp)
        theta = theta - alpha * (1.0/m) * (numpy.dot(X.T,(hyp - y))) # Theta vector for each iteration
        #print(theta)
    return theta

def predice(theta,X):
    return sigmoidal(X.dot(theta))


X,Y = leerDatos("./ex1data2.txt")
#funcionCosto(numpy.zeros((X.shape[1],1)), X,Y)
t = aprende(numpy.zeros((X.shape[1],1)), X,Y, 1000)
#j,g = funcionCosto(t,X,Y)

res = predice(t,numpy.array([45,85]))

print (res)


#X,mu,s = normalizacionDeCaracteristicas(X)
#print(X)
'''
O,hist = gadienteDescendenteMultivariable(X,Y, numpy.zeros((X.shape[1]+1,1))) # Save final theta in
print("\nTheta vector (gradiente decendente):")
print(O)
O2 = ecuacionNormal(X,Y)
print("\nTheta vector (normal equation):")
print(O2)
print("Cost:")
print(calculaCosto(numpy.c_[numpy.ones(X.shape[0]), X[:,:]],Y,O))
graficaError(hist)
#print(predicePrecio(numpy.array([1,2,3]),O))
'''
