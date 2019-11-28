import numpy as np
import GPy

import scipy as sp
from numba import jit


def generate_init(bounds, initial_n):
    dim = len(bounds)
    init_x = np.random.rand(initial_n, dim)
    for i in range(dim):
        init_x[:,i]=init_x[:,i]*(bounds[i][1]-bounds[i][0])+bounds[i][0]
    return init_x

@jit
def gpmean(x, bounds, GP_model):
    if len(x.shape) == 1:
        x = np.array([x])
    mu = GP_model.predict(x)[0]
    return mu

@jit
def gauss_hermite(x, GP_model, N_q):
    points, weights = np.polynomial.hermite.hermgauss(N_q)
    mu, sig = GP_model.predict(x)
    _points = mu + np.sqrt(2)*sig*points
    _weights = np.power(np.pi,-1/2)*weights
    return _points, _weights

def fit(X, Y, kernel,noise_var = .01):
    model = GPy.models.GPRegression(X,Y,kernel,noise_var=noise_var)
    model.optimize()
    return model

def minimize(func_acq,bounds,grid=10):
    result_x = sp.optimize.brute(func_acq, ranges=bounds,Ns=grid,finish=None)
    #result_fx = np.atleast_2d(res[1])
    return np.array([result_x])

def gap(values, fmin=0.01147387):
    s0 = values[0][0]
    smin = min(values)[0]
    G=(s0-smin)/(s0-fmin)
    return G