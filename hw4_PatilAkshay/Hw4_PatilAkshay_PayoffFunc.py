"""
                            MATH 6205 - Numerial Methods for Financial Derivatives
                                         Fall 2019 -UNCC Charlotte

                                            Author: AkshayPatil 
                                            Stud ID : 801034919
                                            Email id : apatil17@uncc.edu
                                            
                                            
PURPOSE : To simulate the value of American call option value and American put option value 
 
Numerical Method Used : Monte carlo simulation and Linear Reggression 1 method

Date of Completition : 10/8/2019 

"""

import numpy as np
from scipy.optimize import curve_fit

#=================================================================
#To calculate value of St using MC and Euler discretization method
#=================================================================
def MC(S0,K,r,sigma,t,T,tstep,path,seed):
    np.random.seed(seed=seed) 
    St = np.zeros((path,tstep+1))
    St[:,0]=S0
    dt = (T-t)/tstep
  
    for i in range(1,tstep+1):        
        dWt = (np.random.standard_normal(path))*np.sqrt(dt)
        St[:,i] = St[:,i-1] + r*St[:,i-1]*dt + sigma*St[:,i-1]*dWt
        
    return St

#==============================================================
#Defining continuation Value Function
#==============================================================
def fit_func(x,a,b,c,d):
    return a + b*x + c*(x*x) + d*(x*x*x)

#==============================================================
#Defining maximum payoff function
#==============================================================
def max_payoff(S0,K,r,sigma,t,T,tstep,path,seed):
    St = MC(S0,K,r,sigma,t,T,tstep,path,seed)

    
    # Defining Payoff function 
    V_call = np.zeros(St.shape)
    V_call[:,-1] = np.maximum(St[:,-1] - K, 0)

    V_put = np.zeros(St.shape)
    V_put[:,-1] = np.maximum(K - St[:,-1], 0)
    
    #Approximating continuation value function using curve_fit    
    for j in range(tstep):
        dt = (T-t)/tstep
        x = St[:,tstep-1-j]
        y = np.exp(-r*dt)*V_call[:,tstep-j]
        z = np.exp(-r*dt)*V_put[:,tstep-j]
    
        [coeff_call, cov] = curve_fit(fit_func, x, y)
        c_hat = fit_func(x, coeff_call[0], coeff_call[1], coeff_call[2], coeff_call[3])
    
        [coeff_put, cov] = curve_fit(fit_func, x, z)
        p_hat = fit_func(x, coeff_put[0], coeff_put[1], coeff_put[2], coeff_put[3])
    
       
        V_call[:, tstep-1-j] = np.maximum((np.maximum(x - K, 0)), c_hat)
        V_put[:, tstep-1-j] = np.maximum((np.maximum(K-x, 0)), p_hat)
    
    return V_call,V_put;
    

    

