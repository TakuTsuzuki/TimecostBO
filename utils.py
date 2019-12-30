import numpy as np
import GPy

import scipy as sp
import numpy as np
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

@jit
def fit(X, Y, kernel,noise_var = 1e-3):
    model = GPy.models.GPRegression(X,Y,kernel,noise_var=noise_var)
    model.rbf.lengthscale.fix()
    #model.rbf.variance.constrain_bounded(1e-12,4)
    model.optimize()
    return model

def minimize(func_acq,bounds,grid=10):
    _result_x = sp.optimize.brute(func_acq, ranges=bounds,Ns=grid,finish=None)
    #result_fx = np.atleast_2d(res[1])
    return np.array([_result_x])

@jit
def minimize_with_threthold2d(func_acq, func_cost,bounds,threthold,grid=10):
    gridx, gridy = brute_grid(func_acq, bounds)
    nextXor0 = filter_threthold2d(func_cost, threthold, gridx, gridy)
    return nextXor0

def brute_grid(func_acq,bounds,grid=10):
    _x, _y, _gridx, _gridy = sp.optimize.brute(func_acq, ranges=bounds,Ns=grid,full_output=True,finish=None)
    #result_fx = np.atleast_2d(res[1])
    return _gridx, _gridy

@jit
def filter_threthold2d(func_cost,threthold,_gridx,_gridy,grid=10,huristics = 1e+10):
    cost_grid = func_cost(_gridx)
    _flagarray = np.where(cost_grid<=threthold, _gridy, huristics)
    if np.sum(_flagarray) == (grid**2)*huristics:
        return 0
    else:
        _index = np.unravel_index(np.argmin(_flagarray),_flagarray.shape)
        _result_x = _gridx[:,_index[0],_index[1]]
        return np.array([_result_x])

def gap(values, fmin):
    s0 = values[0][0]
    smin = min(values)[0]
    G=(s0-smin)/(s0-fmin)
    return G