import numpy
import matplotlib.pyplot as plt

def leerDatos(file):
    points = numpy.loadtxt(file, delimiter=',') # Save all the file's points in a Matrix
    x = numpy.r_['1,2,0', numpy.ones(points.shape[0]), points[:, 0]] # Create Matrix of Xs with 1s
    y = numpy.r_['0,2,0', points[:, 1]] # Create Y Vector, as a Matrix
    return x,y

def graficaDatos(X,y,theta):
    Opoints = X.dot(theta)
    plt.plot(X[:,1],y,'b^',X[:,1],Opoints,'r') # Graficar puntos y recta
    plt.show()
    n,mc = gradienteDescendente(X,y) # Obtener la matriz de costos mc
    plt.plot(mc) # Graficar la curva de errores
    plt.show()
    return 0

def gradienteDescendente(X,y, theta = numpy.zeros((2,1)),alpha =  0.01,iteraciones = 1500):
    m = y.size # Numero de datos
    mcostos = numpy.zeros(iteraciones) # vector de costos
    a = alpha # alpha t
    for i in range(0,iteraciones):
        theta = theta - a * (1.0/m) * (numpy.dot(X.T,(numpy.dot(X,theta) - y))) # Theta vector for each iteration
        mcostos[i] = calculaCosto(X,y,theta) # asignar costo
        a = a - (a/((i+2)*(i+2))) # disminuir siguiente alpha
    #print(a)
    return theta, mcostos

def calculaCosto(X,y,theta):
    res = X.dot(theta) - y # (XO - Y)
    cost = (1.0/(2*y.size))*(res.T.dot(res)) # 1/2m * sum of squares of (XO - Y) = (XO - Y)t * (XO - Y)
    return cost

X,Y = leerDatos("./ex1data1.txt")
O,mc = gradienteDescendente(X,Y) # Save final theta in O
print("\nTheta vector:")
print(O)
print("\nPrediccion 1:")
print(numpy.array([1, 3.5]).dot(O) * 10000)
print("Prediccion 2:")
print(numpy.array([1,7]).dot(O) * 10000)
print("Cost:")
print(calculaCosto(X,Y,O))
graficaDatos(X,Y,O)
