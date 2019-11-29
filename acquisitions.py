import numpy as np
import GPy
from numba import jit
import scipy as sp

from utils import fit, gauss_hermite, gpmean, minimize

@jit
def ei(x,bounds,GP_model,jitter=0.):
    if len(x.shape) == 1:
        x = np.array([x])
    mu, sig = GP_model.predict(x)
    fmin = GP_model.predict(GP_model.X)[0].min()
    if isinstance(sig, np.ndarray):
        sig[sig<1e-10] = 1e-10
    elif sig< 1e-10:
        sig = 1e-10
    u = (fmin - mu - jitter)/sig
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * sp.special.erfc(-u / np.sqrt(2))
    next_x = sig * (u * Phi + phi)
    return next_x

def policy(GP_model, depth_h,bounds):
    # caluc policy function
    if depth_h > 1:
        func2minimize = lambda x : -1*ei(x,bounds,GP_model)
    else:
        func2minimize = lambda x : gpmean(x,bounds,GP_model)
    query = minimize(func2minimize, bounds)
    return query

@jit
def rollout_utility_archive(x,
                    bounds,
                    func_policy, 
                    depth_h, 
                    _queries, 
                    _values, 
                    N_q,
                    decay_rate=0.9,
                    ARD_Flag = False,
                    length_scale = None):
    #print(depth_h)
    if len(x.shape) == 1:
        x = np.array([x])
    kernel = GPy.kern.RBF(len(bounds), ARD=ARD_Flag, lengthscale=length_scale)
    gp_model = fit(_queries, _values, kernel) #todo:memo
    if depth_h == 0:
        U = ei(x,bounds ,gp_model)
    else:
        U = ei(x,bounds, gp_model)
        _queries = np.concatenate([_queries,x])
        points, weights = gauss_hermite(x, gp_model, N_q)
        for i in range(N_q):
            val = np.array([[points[0][i]]])
            _values = np.concatenate([_values,val])
            kernel = GPy.kern.RBF(len(bounds), ARD=ARD_Flag, lengthscale=length_scale)
            #print("X",_queries)
            #print("Y",_values)
            _gp_model = fit(_queries, _values, kernel) #todo:memo
            #print(i,"afterfit_afterker")
            x_next = func_policy(_gp_model, depth_h,bounds)
            U = U + weights[i]*decay_rate*rollout_utility_archive(x_next,
                                    bounds,
                                    func_policy,
                                    depth_h-1,
                                    _queries,
                                    _values,
                                    N_q,
                                    decay_rate,
                                    ARD_Flag = ARD_Flag,
                                    length_scale = length_scale)
            _values = _values[:-1,:]
        _queries = _queries[:-1,:]
    return U

@jit
def rollout_mcmc(x,
                    bounds,
                    func_policy, 
                    depth_h, 
                    queries, 
                    values, 
                    n_sample=10,
                    decay_rate=.9,
                    ARD_Flag = False,
                    length_scale = None,
                    remain_h = None
                        ):
    if depth_h>remain_h:
        depth_h = remain_h #maximum depth depends on num of queries that remains
    if len(x.shape) == 1:
        x = np.array([x])
        
    kernel = GPy.kern.RBF(len(bounds), ARD=ARD_Flag, lengthscale=length_scale)
    gp_model = fit(_queries, _values, kernel)
    if depth_h == 0:
        U = ei(x,bounds ,gp_model)
        
    queriesori = np.copy(queries)
    valuesori = np.copy(values)
    for i in range(n_sample):
        _queries = np.copy(queriesori)
        _values = np.copy(valuesori)
        #predict 1st step
        kernel = GPy.kern.RBF(len(bounds), ARD=ARD_Flag, lengthscale=length_scale)
        gp_model = fit(_queries, _values, kernel)
        mu, sig = _gp_model.predict_f(x)
        val = np.array([[np.random.normal(mu, sig)]])
        _queries = np.concatenate([_queries,x])
        _values = np.concatenate([_values,val])
        for j in range(depth_h):
            _remain_h = depth_h - j - 1
            kernel = GPy.kern.RBF(len(bounds), ARD=ARD_Flag, lengthscale=length_scale)
            gp_model = fit(_queries, _values, kernel)
            x_next = func_policy(_gp_model, _h, bounds)
            _queries = np.concatenate([_queries,x])
            mu, sig = _gp_model.predict_f(x)
    
    #kernel = gpflow.kernels.RBF(len(bounds), ARD=True) #todo: fuck!!
    gp_model = fit(_queries, _values, kernel) #todo:memo
    if depth_h == 0:
        U = ei(x,bounds ,gp_model)
    else:
        U = ei(x,bounds, gp_model)
        _queries = np.concatenate([_queries,x])
        points, weights = gauss_hermite(x, gp_model, N_q)
        for i in range(N_q):
            #print("N_q:",i)
            #print(i,"beforfit_beforker")
            val = np.array([[points[0][i]]])
            _values = np.concatenate([_values,val])
            #kernel = gpflow.kernels.RBF(len(bounds), ARD=True)
            #print("X",_queries)
            #print("Y",_values)
            _gp_model = fit(_queries, _values, kernel) #todo:memo
            #print(i,"afterfit_afterker")
            x_next = func_policy(_gp_model, depth_h,bounds)
            U = U + weights[i]*decay_rate*rollout_utility_archive(x_next,
                                    bounds,
                                    func_policy,
                                    depth_h-1,
                                    _queries,
                                    _values,
                                    kernel,
                                    N_q,
                                    decay_rate )
            _values = _values[:-1,:]
        _queries = _queries[:-1,:]
    return U