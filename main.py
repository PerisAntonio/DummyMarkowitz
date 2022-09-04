
import numpy as np
from scipy.optimize import minimize,shgo,dual_annealing
from time import time
import json
import math

TREASURY_BILL_RATE = 1.5
DAYS_PER_YEAR=1
def random_weights(weight_count):
    weights = np.random.random((weight_count, 1))
    weights /= np.sum(weights)
    return weights.reshape(-1, 1)
def equal_weights(weight_count):    
    weights = np.ones((weight_count, 1))
    weights /= np.sum(weights)
    return weights.reshape(-1, 1)
def _variance(cov, w):
    return (w.reshape(-1, 1).T @ cov @ w.reshape(-1, 1))[0][0]

def _expected_return(rends, w):
    return (rends.T @ w.reshape(-1, 1))[0]

def safe_func(w,rends,cov,n,r):
    return 1000.*(-(_expected_return(rends,w)*n-r/100)/(np.sqrt(_variance(cov,w)*n)))

def unsafe_func(w,rends,cov,n,risk):
    return 1000.*(_variance(cov,w)*n-risk*_expected_return(rends,w)*n)

def run(input_data, solver_params, extra_arguments):
    # This is the core of your algorithm,
    # here you must return the corresponding answer.
    cost = input_data["cost"]
    cov = input_data["cov"]
    current=input_data['current']   
    rends=np.nan_to_num(np.array(input_data['rends']),nan=-1000)
    budget=float(input_data['budget'])
    prices=np.array(input_data['prices'])
    tickers=np.array(input_data['tickers'])
    start=time()
    options={}
    if('risk_acceptance_level' in extra_arguments):
        acceptance=extra_arguments['risk_acceptance_level']
    i=25
    initial_vector=random_weights(len(input_data['current'])).T[0]
    while(i<=3200):
        options['maxiter']=i
        res = minimize(
            lambda w: unsafe_func(w,rends,cov,DAYS_PER_YEAR,acceptance) if('risk_acceptance_level' in extra_arguments) else safe_func(w,rends,cov,DAYS_PER_YEAR,TREASURY_BILL_RATE),
            initial_vector,
            constraints=[
              {'type': 'eq', 'fun': lambda w: 1-np.sum(w) },
            ],
            bounds=[(0., 2./len(input_data['current'])) for i in range(len(input_data['current']))],
            method='trust-constr',
            options=options

          )
        if(res['success']==True):
            i=10000000
        else:
            i=i*2
            initial_vector=res['x']
            if(i>3200):
                initial_vector=random_weights(len(input_data['current'])).T[0]
                i=25
    np.set_printoptions(suppress=True)
    real_optimization={}
    real_optimization['solution']=np.around(res['x']*budget/prices,decimals=4)
    real_optimization['sharpe_ratio']=safe_func(res['x'],rends,cov,DAYS_PER_YEAR,TREASURY_BILL_RATE)/(-1000.)
    real_optimization['expected_return']=_expected_return(rends,res['x'])*budget
    real_optimization['risk']=np.sqrt(_variance(cov,res['x'])*DAYS_PER_YEAR)
    result=[math.floor(budget*x/y) for x,y in zip(res['x'],prices)]
    budget_res=budget-sum([x*y for x,y in zip(result,prices)])
    while(sum([x>0 and budget_res>=y for x,y in zip(rends,prices)]) >0):
        actual=np.array([x*y/budget for x,y in zip(result,prices)])
        unsafe_func(actual,rends,cov,DAYS_PER_YEAR,acceptance) if('risk_acceptance_level' in extra_arguments) else safe_func(actual,rends,cov,DAYS_PER_YEAR,TREASURY_BILL_RATE)
        m=unsafe_func(actual,rends,cov,DAYS_PER_YEAR,acceptance) if('risk_acceptance_level' in extra_arguments) else safe_func(actual,rends,cov,DAYS_PER_YEAR,TREASURY_BILL_RATE)
        elegido=-1
        candidates=[x>0 and budget_res>=y for x,y in zip(rends,prices)]
        for i in range(len(candidates)):
            if(candidates[i]==True):
                result_part=result.copy()
                result_part[i]=result_part[i]+1
                actual_temp=np.array([x*y/budget for x,y in zip(result_part,prices)])
                m_temp=unsafe_func(actual_temp,rends,cov,DAYS_PER_YEAR,acceptance) if('risk_acceptance_level' in extra_arguments) else safe_func(actual,rends,cov,DAYS_PER_YEAR,TREASURY_BILL_RATE)
                if(m_temp<m):
                    m=m_temp
                    elegido=i
        if(elegido!=-1):
            result[elegido]=result[elegido]+1
            budget_res=budget_res-prices[elegido]
        else:
            budget_res=-1
    data={}
    data['time']=time()-start
    data['real_optimization']=real_optimization
    adjusted_optimization={}
    adjusted_optimization['solution']=[(x,y) for x,y in zip(tickers,result)]
    weights=result*prices/budget
    adjusted_optimization['sharpe_ratio']=safe_func(weights,rends,cov,DAYS_PER_YEAR,TREASURY_BILL_RATE)/(-1000.)
    adjusted_optimization['expected_return']=_expected_return(rends,weights)*budget
    adjusted_optimization['risk']=np.sqrt(_variance(cov,weights)*DAYS_PER_YEAR)
    data['adjusted_optimization']=adjusted_optimization
    
    
    return data
    

