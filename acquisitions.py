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

def policy_rollout(GP_model, depth_h,bounds):
    # caluc policy function
    if depth_h > 1:
        func2minimize = lambda x : -1*ei(x,bounds,GP_model)
    else:
        func2minimize = lambda x : gpmean(x,bounds,GP_model)
    query = minimize(func2minimize, bounds)
    return query

def policy_naive(GP_model, depth_h,bounds):
    # caluc naive policy function
    func2minimize = lambda x : -1*ei(x,bounds,GP_model)
    query = minimize(func2minimize, bounds)
    return query

U = 0
@jit
def rollout_utility_archive(x,
                    bounds,
                    func_policy, 
                    depth_h, 
                    _queries, 
                    _values, 
                    N_q,
                    n_sample=None,
                    decay_rate=0.9,
                    ARD_Flag = False,
                    length_scale = None):
    #print(depth_h)
    global U
    if len(x.shape) == 1:
        x = np.array([x])
    kernel = GPy.kern.RBF(len(bounds), ARD=ARD_Flag, lengthscale=length_scale)
    gp_model = fit(_queries, _values, kernel) #todo:memo
    if depth_h == 0:
        U += ei(x,bounds ,gp_model)
    else:
        U += ei(x,bounds, gp_model)
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
            _values = np.copy(_values[:-1,:])
        _queries = np.copy(_queries[:-1,:])
    _U = U
    U = 0
    return _U

@jit
#@jit("f8(f8,f8[:,:],char)"" , nopython=True)
def rollout_mcmc(x,
                bounds,
                func_policy, 
                depth_h, 
                _queries, 
                _values, 
                N_q = 5,
                n_sample=10,
                decay_rate=.9,
                ARD_Flag = False,
                length_scale = None
                    ):
    if len(x.shape) == 1:
        x = np.array([x])
        
    kernel = GPy.kern.RBF(len(bounds), ARD=ARD_Flag, lengthscale=length_scale)
    gp_model = fit(_queries, _values, kernel)
    U = ei(x,bounds,gp_model)
    if depth_h == 0:
        return U
    else:        
        queriesori = np.copy(_queries)
        valuesori = np.copy(_values)
        Udelays = np.array([])
        _mu, _sig = gp_model.predict(x)
        for i in range(n_sample):
            _Udelay = 0
            _queriesf = np.copy(queriesori)
            _valuesf = np.copy(valuesori)
            #kernel = GPy.kern.RBF(len(bounds), ARD=ARD_Flag, lengthscale=length_scale)
            #gp_model = fit(_queries, _values, kernel)
            #predict 1st step
            #mu, sig = gp_model.predict(x)
            y_next = np.array(np.random.normal(_mu, _sig))
            _queriesf = np.concatenate([_queriesf,x])
            _valuesf = np.concatenate([_valuesf,y_next])
            for j in range(depth_h):
                _remain_h = depth_h - j - 1
                #kernel = GPy.kern.RBF(len(bounds), ARD=ARD_Flag, lengthscale=length_scale)
                gp_model = fit(_queriesf, _valuesf, kernel)
                x_next = func_policy(gp_model, _remain_h, bounds)
                _Udelay += decay_rate * ei(x_next,bounds,gp_model)

                _queriesf = np.concatenate([_queriesf,x_next])
                mu, sig = gp_model.predict(x_next)
                next_y = np.array(np.random.normal(mu, sig))
                _valuesf = np.concatenate([_valuesf,next_y])
            Udelays = np.append(Udelays, _Udelay)
        U += np.mean(Udelays)
        return U

"""
_h = 0
_count_depth = 0
_gp_list = {}
_queries_list = {}
_values_list = {}
_trajectory = []
U = 0

@jit
def rollout_utility(x,
                    bounds,
                    func_policy, 
                    depth_h, 
                    _queries, 
                    _values, 
                    N_q,
                    n_sample=None,
                    decay_rate=0.9,
                    ARD_Flag = False,
                    length_scale = None):
    global _h 
    global _gp_list
    global _queries_list
    global _values_list
    global _trajectory
    global U
    _h = max([_h, depth_h])


    if len(x.shape) == 1:
        x = np.array([x])
    kernel = GPy.kern.RBF(len(bounds), ARD=ARD_Flag, lengthscale=length_scale)
    if depth_h == 0:
        gp_model = fit(_queries, _values, kernel)
        U += ei(x,bounds ,gp_model)
    else:    
        U += ei(x,bounds, gp_model)
        curr_depth = _h - depth_h #current depth
        _queries = np.concatenate([_queries,x])
        # GaussHermite
        points, weights = gauss_hermite(x, gp_model, N_q)
        for i in range(N_q):
            _trajectory.append(str(i))
            #_queries = np.concatenate([_queries,x])
            _id = str(curr_depth) + "".join(_trajectory)
            if _id in _gp_list:
                #gp_model = _gp_list[_id]
                #_queries = _queries_list[_id]
                _values = _values_list[_id]
            else:
                estimated_value = np.array([[points[0][i]]])
                _values = np.concatenate([_values, estimated_value])
                _gp_model = fit(_queries, _values, kernel) #memo
                x_next = func_policy(_gp_model, depth_h)
                
                _gp_list[_id] = _gp_model
                _values_list[_id] = _values
                
            U += weights[i]*decay_rate*rollout_utility(x_next,
                                            bounds,
                                            func_policy,
                                            depth_h-1,
                                            _queries,
                                            _values,
                                            kernel,
                                            N_q,
                                            decay_rate )
            _values = _values[:-1,:]
            _trajectory.pop(-1)
        _queries = _queries[:-1,:]
    return U(x)
"""