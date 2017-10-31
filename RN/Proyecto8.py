import numpy as np
import random

def leerDatos(file):
    points = np.loadtxt(file, delimiter=' ') # Save all the file's points in a Matrix
    print(points.shape)
    cols = points.shape[1]
    print("cols:", cols)
    x = np.c_[np.ones(points.shape[0]), points[:, np.arange(0,cols-1)]]# Create Matrix of Xs with 1s
    y = np.c_[points[:, cols-1]] # Create Y Vector, as a Matrix
    return x,y

def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y):
    alpha = 2
    iteraciones = 10000
    # Variables inicialization
    m = X.shape[0]
    w1 = randInicializacionPesos(input_layer_size, hidden_layer_size)
    #print("w1.shape:", w1.shape)
    w2 = randInicializacionPesos(hidden_layer_size, num_labels)
    #print("w2.shape:", w2.shape)
    b1 = np.ones((1,hidden_layer_size))
    b2 = np.ones((1,num_labels))
    y_c = transform_y(y, num_labels)

    for i in range(0,iteraciones):
        print(" {0}% completado".format((i / (iteraciones/100))), end="\r")
        # Forward-propagation
        z1 = X.dot(w1) + b1
        a1 = sigmoidal(z1)
        z2 = a1.dot(w2) + b2
        a2 = sigmoidal(z2)

        '''
        print("z1.shape:", z1.shape)
        print("a1.shape:", a1.shape)
        print("z2.shape:", z2.shape)
        print("a2.shape:", a2.shape)
        print("y_c.shape:", y_c.shape)
        '''

        # Back-propagation
        dz2 = a2 - y_c
        #print("dz2.shape:", dz2.shape)
        dw2 = a1.T.dot(dz2)* (1/m)
        #print("dw2.shape:", dw2.shape)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        #print("db2.shape:", db2.shape)
        da1 = w2.dot(dz2.T)
        #print("da1.shape:", da1.shape)
        dz1 = np.multiply(dz2.dot(w2.T), sigmoidalGradiente(z1))
        #print("dz1.shape:", dz1.shape)
        dw1 = (1/m)*dz1.T.dot(X)
        #print("dw1.shape:", dw1.shape)
        db1 = (1/m)*np.sum(dz1, axis=0, keepdims=True)
        #print("db1.shape:", db1.shape)

        # Actualizacion
        w2 = w2 - alpha*dw2
        b2 = b2 - alpha*db2
        w1 = w1 - alpha*dw1.T
        b1 = b1 - alpha*db1


    return w1,b1,w2,b2

def transform_y(y, labels):
    n = np.zeros((y.shape[0], labels))
    cont = 0
    for it in y:
        #print("type:", type(it[0]))
        n[cont, int(it[0])-1] = 1
        cont+=1
    return n

def sigmoidal(x): # validated
    x = -1 * x
    s = np.exp(x)
    return 1/(1+s)

def sigmoidalGradiente(z): # validated
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

def randInicializacionPesos(L_in, L_out): #validated
    eps = 0.12
    w = np.random.uniform(-eps,eps,L_in*L_out).reshape((L_in, L_out))
    return w # shape:(L_in,L_out)

def prediceRNYaEntrenada(X,W1,b1,W2,b2):
    z1 = X.dot(W1) + b1
    a1 = sigmoidal(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoidal(z2)
    y = np.argmax(a2, axis=1)
    y = y+1
    print("prediccion.shape:", y.shape)
    return y

X,y = leerDatos("./RN/newData.txt")

print("X.shape:",X.shape)
print("y.shape", y.shape)

t1,b1,t2,b2 = entrenaRN(X.shape[1], 25, 10, X,y)

res = prediceRNYaEntrenada(X,t1,b1,t2,b2)

print("shape of diff", (res-y.T).shape)

err = np.count_nonzero(res - y.T)/50
print("err:", err)
