import numpy as np
import random

def leerDatos(file):
    points = np.loadtxt(file, delimiter=' ') # Save all the file's points in a Matrix
    print(points.shape)
    x = np.r_['1,2,0', np.ones(points.shape[0]), points[:, 0]] # Create Matrix of Xs with 1s
    y = np.r_['0,2,0', points[:, 1]] # Create Y Vector, as a Matrix
    return x,y
    
def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y):
    
    
X,y = leerDatos("./digitos.txt")

print("X.shape:",X.shape)
print("y.shape", y.shape)