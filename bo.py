import numpy as np
import GPy
from numba import jit
from utils import generate_init, fit
from acquisitions import ei, minimize, policy


def bayesianOptimization(func_objective,
                         func_acq,
                         bounds,
                         depth_h,
                         N,
                         initial_n=1,
                         N_q=3,
                         decay_rate=1,
                         ARD_Flag = False,
                         length_scale = None):
    """
    depth_h: num of nest
    N: num of iter
    """
    if depth_h > N:
        print("depth_h > N")
        return 0
    elif initial_n > N:
        print("initial_n > N")
        return 0
    
    _N = N - initial_n
    if initial_n > 0:
        queries = generate_init(bounds, initial_n)
    else:
        queries = np.array([[0.8, 0.8]]) 
    values = func_objective(queries)
    lengtn_scale = (bounds[0][1]-bounds[0][0])/10.
    for i in range(_N):
        print(i)
        kernel = GPy.kern.RBF(len(bounds), ARD=ARD_Flag, lengthscale=length_scale)
        #gp_model = fit(queries, values)
        _h = min({depth_h,_N-i})
        _count_depth = 0
        _gp_list = {}
        _queries_list = {}
        _values_list = {}
        _trajectory = []
        #_idlist = []
        if func_acq == ei:
            GP_model = fit(queries, values, kernel)
            facq = lambda x : -1*ei(x,bounds,GP_model)
        else:
            facq = lambda x : -1*func_acq(x, 
                                      bounds = bounds,
                                      func_policy=policy, 
                                      depth_h = _h, 
                                      _queries = queries,
                                      _values = values,
                                      N_q = N_q,
                                      decay_rate=decay_rate,
                                      ARD_Flag = ARD_Flag,
                                      length_scale = length_scale)
        X = minimize(facq, bounds)
        Y = func_objective(X)
        queries = np.concatenate([queries,X])
        values = np.concatenate([values,Y])
    return queries, values