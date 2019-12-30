import numpy as np

def sigmoid(_array, a = 0.6, b = 20, bottom =1, upper = 10):
    yarray = (upper-bottom)*(1./(1.+np.exp(b*(_array-a)))) + bottom
    return yarray

def costfunc1(_ndarray, axis = 0):
    _array = _ndarray[axis]
    costarray = sigmoid(_array)
    return costarray