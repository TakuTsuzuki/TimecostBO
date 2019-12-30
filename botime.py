import numpy as np
import GPy
from numba import jit
from utils import generate_init, fit, brute_grid, minimize_with_threthold2d
from acquisitionstime import ei,ei_per_cost ,minimize, policy_mu
#from cost import costfunc1

def bayesianOptimization(func_objective,
                         func_cost,
                         func_acq,
                         func_policy,
                         bounds,
                         depth_cost,
                         givenCost,
                         initial_n=1,
                         initpoint=None,
                         n_sample = 10,
                         decay_rate=1,
                         length_scale = 0.3,
                         ARD_Flag = False):
    """
    depth_h: num of nest
    N: num of iter
    """
    assert depth_cost <= givenCost, "Error: depth_cost > givenCost"
    #assert initial_n <= N, "Error: initial_n > N" 
    
    _length_scale = length_scale*(bounds[0][1]-bounds[0][0])

    # load/generate init points
    if initial_n > 0:
        queries = generate_init(bounds, initial_n)
    else:
        queries = initpoint


    initial_cost = np.sum(func_cost(queries))
    remainCost = givenCost - initial_cost
    values = func_objective(queries)
    count = 0

    while remainCost > 1 : # assume that min(func_cost) is 1
        count = count + 1
        print(count)
        kernel = GPy.kern.RBF(len(bounds), ARD=ARD_Flag, lengthscale=length_scale)
        _remainC = min({depth_cost,remainCost})
        GP_model = fit(queries, values, kernel)

        # define acquisition function
        if func_acq == ei:
            facq = lambda x : -1*ei(x,bounds,GP_model)
        elif func_acq == ei_per_cost:
            facq = lambda x : -1*ei_per_cost(x,func_cost,bounds,GP_model)
        else:
            facq = lambda x : -1*func_acq(x, 
                                      bounds = bounds,
                                      func_policy=func_policy, 
                                      func_cost = func_cost,
                                      depth_c = _remainC, 
                                      _queries = queries,
                                      _values = values,
                                      n_sample = n_sample,
                                      decay_rate = decay_rate,
                                      ARD_Flag = ARD_Flag,
                                      length_scale = _length_scale)
        
        # compute threthold(lastpoint)
        muquery = policy_mu(GP_model, bounds)
        threthold = _remainC - func_cost(muquery)

        # select next query given that threthold
        X = minimize_with_threthold2d(facq, func_cost,bounds ,threthold)
        Y = func_objective(X)

        # subtract cost from remain cost
        remainCost = remainCost - func_cost(X)
        queries = np.concatenate([queries,X])
        values = np.concatenate([values,Y])
    return queries, values

