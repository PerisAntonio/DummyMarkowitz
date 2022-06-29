import numpy as np
from scipy import minimize
from time import time
import json


TREASURY_BILL_RATE = 1.5
def random_weights(weight_count):
    weights = np.random.random((weight_count, 1))
    weights /= np.sum(weights)
    return weights.reshape(-1, 1)

def _variance(cov, w):
    return (w.reshape(-1, 1).T @ cov @ w.reshape(-1, 1))[0][0]

def _expected_return(rends, w):
    return (rends.T @ w.reshape(-1, 1))[0]

def run(input_data, solver_params, extra_arguments):
    # This is the core of your algorithm,
    # here you must return the corresponding answer.
    cost = input_data["cost"]
    cov = input_data["cov"]
    current=input_data['current']
    optimal=input_data['optimal']
    rends=input_data['rends']
    solution={}
    start=time()
    res = minimize(
        lambda w: -(_expected_return(r,w)-TREASURY_BILL_RATE/100)/np.sqrt(_variance(cov,w)),
        random_weights(len(data['current'])),
        constraints=[
          {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.},
        ],
        bounds=[(0., 1.) for i in range(len(data['Current']))]
      )
    data['time']=time()-start
    data['success']=res['success']
    data['solution']=res['x']
    data['message']=res['message']
    
    return json.loads(data)
    

