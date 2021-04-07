

"""
                            MATH 6205 - Numerial Methods for Financial Derivatives
                                         Fall 2019 -UNCC Charlotte

                                            Author: AkshayPatil 
                                            Stud ID : 801034919
                                            Email id : apatil17@uncc.edu
                                            
                                            
PURPOSE : To simulate the value of American call option value and American put option value 
 
Numerical Method Used : Monte carlo simulation and Linear Reggression 2 method

Date of Completition : 10/22/2019 

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
    return a + b*x + c*(x**2) + d*(x*x*x)

#==============================================================
#Defining maximum payoff function
#==============================================================
def max_payoff(S0,K,r,sigma,t,T,tstep,path,seed):
    St = MC(S0,K,r,sigma,t,T,tstep,path,seed)

    
    # Defining Payoff function 
    G_call = np.zeros(path)
    G_call = np.maximum(St[:,-1] - K, 0)

    G_put = np.zeros(path)
    G_put = np.maximum(K - St[:,-1], 0)
    
    #Approximating continuation value function using curve_fit    
    for j in range(tstep-1,0,-1):
        dt = (T-t)/tstep
        tau = np.ones(path)*tstep
        #To find indexes of in-the-money points of call option
        index_1 = np.where(St[:,j] > K)[0]
        #To find indexes of in-the-money points of put option
        index_2 = np.where(K > St[:,j])[0]
        w = St[index_1,j] # St values for in the money call option
        x = St[index_2,j] # St values for in the money put option
        #For American call
        y = np.exp(-r*(tau[index_1]-j)*dt)*G_call[index_1]
        #For American put
        z = np.exp(-r*(tau[index_2]-j)*dt)*G_put[index_2]
        
        #Utilising Curve fit function
        [coeff_call, cov] = curve_fit(fit_func, w, y)
        [coeff_put, cov] = curve_fit(fit_func, x, z)
        
        for k in index_1:
            payoff_call= St[k,j]-K   #Payoff of in the money point for call
            c_hat = fit_func(St[k,j], coeff_call[0], coeff_call[1], coeff_call[2], coeff_call[3])
            
            if payoff_call >= c_hat:
                G_call[k] = payoff_call
                tau[k] = j
           
            
                
        for k in index_2:       
            payoff_put= K - St[k,j]  #Payoff of in the money point for put
            p_hat = fit_func(St[k,j], coeff_put[0], coeff_put[1], coeff_put[2], coeff_put[3])
            if payoff_put >= p_hat:
                G_put[k] = payoff_put
                tau[k] = j
    
    
    call_opt = np.mean(np.exp(-r*tau*dt)*G_call)
    put_opt = np.mean(np.exp(-r*tau*dt)*G_put)
    
            

    
    return call_opt,put_opt;
    

    
