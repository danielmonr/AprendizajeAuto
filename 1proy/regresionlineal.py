import numpy
import matplotlib.pyplot as plt

def leerDatos(file):
    print 'Parsing File'
    points = numpy.loadtxt(file, delimiter=',') # Save all the file's points in a Matrix
    #print points.shape
    x = numpy.r_['1,2,0', numpy.ones(points.shape[0]), points[:, 0]] # Create Matrix of Xs with 1s
    y = numpy.r_['0,2,0', points[:, 1]] # Create Y Vector, as a Matrix
    #print x
    #print y
    print 'Dim x:'
    print x.shape
    print 'Dim y:'
    print y.shape
    return x,y
    
def graficaDatos(X,y,theta):
    #points = numpy.r_['1,2,0',X[:,1], y[:,0]]
    Opoints = X.dot(theta)
    plt.plot(X,y,'b',X,Opoints,'r')
    plt.show()
    return 0
    
def gradienteDescendente(X,y, theta = numpy.zeros((2,1)),alpha =  0.01,iteraciones = 1500):
    m = y.size # Numero de datos
    for i in xrange(0,iteraciones):
        theta = theta - alpha * (1.0/m) * (numpy.dot(X.T,(numpy.dot(X,theta) - y)))
    
    #print theta
    return theta
    
def calculaCosto(X,y,theta):
    res = X.dot(theta) - y # (XO - Y)
    cost = (1.0/(2*y.size))*(res.T.dot(res)) # 1/2m * sum of squares of (XO - Y) = (XO - Y)t * (XO - Y)
    return cost
    

X,Y = leerDatos("./ex1data1.txt")
O = gradienteDescendente(X,Y) # Save final theta in O
print 'Dim O'
print O.shape
print "\nTheta vector:"
print O
print "\nPrediccion 1:"
print numpy.array([1, 3.5]).dot(O)
print "Prediccion 2:"
print numpy.array([1,7]).dot(O)
print "Cost:"
print calculaCosto(X,Y,O)
#graficaDatos(X,Y,O)