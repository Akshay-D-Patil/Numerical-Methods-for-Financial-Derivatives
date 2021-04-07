#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
                            MATH 6205 - Numerial Methods for Financial Derivatives
                                         Fall 2018 -UNCC Charlotte

                                            Author: AkshayPatil 
                                            Stud ID : 801034919
                                            Email id : apatil17@uncc.edu
                                            
                                            
PURPOSE : To simulate the call option value and put option value and compare it with analytical solution
 
Numerical Method Used : Euler discretization of a Stochastic differential equation and BSM equation for analytical solution

Date of Completition : 09/26/2019 

"""





"""Importing numpy and matplotlib.pyplot libraries"""

from math import *
import numpy as np
from scipy.stats import norm
import pandas as pd



def main(S0,K,r,sigma,t,T,N,path,seed):
    #line 35: make a dataframe
    out_df1 = pd.DataFrame(columns=['N','BSM_C','MC_C' , 'Error_call', 'BSM_P','MC_P','Error_put'] )
    OptVal = MC(S0,K,r,sigma,t,T,N,path,seed)



    """ Call and Put functions by BS equation"""
    call_value = BS_call_option(S0,K,T,t,r,sigma)
    put_value =   BS_put_option(S0,K,T,t,r,sigma)



    Error_call = abs(OptVal['C']-call_value)
    Error_put = abs(OptVal['P']-put_value)
    out_df1 = out_df1.append({'N':N,'BSM_C':call_value,'MC_C':OptVal['C'], 'Error_call':Error_call,'BSM_P':put_value, 'MC_P':OptVal['P'],'Error_put':Error_put},ignore_index=True)
    return out_df1


"""
Defining the function for MC path which calculates the values of St using Euler discretization method
"""

def MC(S0,K,r,sigma,t,T,N,path,seed):
    #To calculate value of St using Euler discretization method
    np.random.seed(seed=seed) 
    # Line 26 creates an matrix named S(t) which is of shape (path * (N+1))
    #In our case it creates a matrix of (5 * 1001)
    St = np.zeros((path,N+1))
    
    # Line 29: There are 1001 columns because we use first column to store the initial value of S(t) at t=0, in our case S90) = 100
    St[:,0]=S0
    dt = (T-t)/N
    #Following for loop runs N times, as it calculates the values of S(t) and stores them in the matrix S(t)
    for i in range(1,N+1):
        
#        Line 40: Computing wiener process 
#        Define N(0,1) to be the standard random variable that is normally distributed 
#        with mean 0 and standard deviation 1. Each random number ∆Wi is computed as 
#        ∆Wi =  Zi* sqrt(∆ti)
#        where zi is chosen from N(0,1)

        dWt = (np.random.standard_normal(path))*np.sqrt(dt)
        
#         Line 44: The Euler discretization of the SDE is given as:
#                   S(t) = S(t-1) + μ*S(t-1)*dt + σ*S(t)*dWt        
        St[:,i] = St[:,i-1] + r*St[:,i-1]*dt + sigma*St[:,i-1]*dWt
    
#    Vc = np.zeros((path,1))
#    for i in range(1,path+1):
#        Vc[:,i] = 
        
    C= np.exp(-r * T) * np.mean(np.maximum(St[:,-1] - K, 0))
    P= np.exp(-r * T) * np.mean(np.maximum(K - St[:,-1], 0))
  
    return {'C': C, 'P': P}
    

""" Defining d1 to calculate value of d1"""
def d1_value(S,K,T,t,r,sigma):
    d1 = (log(S/K)+((r  + ((sigma*sigma)/2.0)) * (T-t))) / (sigma*sqrt(T-t))
    return d1  



""" Defining call option"""
def BS_call_option(S,K,T,t,r,sigma):
    
#    This function computes value of call option using Black-Scholes method.
    
    d1 = d1_value(S,K,T,t,r,sigma)
    d2 = d1 - (sigma*sqrt(T-t))
    return S*norm.cdf(d1) - K * exp(-r*(T-t)) * norm.cdf(d2)



""" Defining put option"""
def BS_put_option(S,K,T,t,r,sigma):
    
#    This function computes value of Put option using Black-Scholes method.
    
    d1 = d1_value(S,K,T,t,r,sigma)
    d2 = d1 - (sigma*sqrt(T-t))
    return (-S*norm.cdf(-d1)) + K * exp(-r*(T-t)) * norm.cdf(-d2)
    
