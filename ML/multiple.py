import numpy
import matplotlib.pyplot as plt

def leerDatos(file):
    points = numpy.loadtxt(file, delimiter=',') # Save all the file's points in a Matrix
    cols = points.shape[1]
    x = points[:, numpy.arange(0,cols-1)]# Create Matrix of Xs with 1s
    y = numpy.c_['0,2,0', points[:, cols-1]] # Create Y Vector, as a Matrix
    #print(x.shape)
    return x,y

def normalizacionDeCaracteristicas(X):
    mu = numpy.mean(X,axis=0)
    sigma = []
    sigma = numpy.std(X, ddof=1, axis=0)
    X = (X - mu)/sigma
    #print(X.shape[1])
    return X,mu,sigma

def graficaError(J_Historial):
    plt.plot(J_Historial) # Graficar la curva de errores
    plt.show()
    return 0

def gadienteDescendenteMultivariable(X,y, theta,alpha =  0.1,iteraciones = 100):
    X = numpy.c_[numpy.ones(X.shape[0]), X[:,:]]
    m = y.size # Numero de datos
    j_historia = numpy.zeros(iteraciones)
    for i in range(0,iteraciones):
        hyp = X.dot(theta)
        theta = theta - alpha * (1.0/m) * (numpy.dot(X.T,(hyp - y))) # Theta vector for each iteration
        j_historia[i] = calculaCosto(X,y,theta)
    return theta, j_historia

def calculaCosto(X,y,theta):
    res = X.dot(theta) - y # (XO - Y)
    cost = (1.0/(2*y.size))*(res.T.dot(res)) # 1/2m * sum of squares of (XO - Y) = (XO - Y)t * (XO - Y)
    return cost

def ecuacionNormal(X,y):
    X = numpy.c_[numpy.ones(X.shape[0]), X[:,:]]
    theta = numpy.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

def predicePrecio(X,theta):
    precio = X.dot(theta)
    return precio

X,Y = leerDatos("./ex1data2.txt")
X,mu,s = normalizacionDeCaracteristicas(X)
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
