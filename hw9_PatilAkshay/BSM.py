"""
                             MATH 6205 - Numerical Methods for Financial Derivatives
                                                  Fall 2019


Purpose             : The objective of this Python program is to calculate
                      analytical solution of European Call and European Put
                      option using BSM equation
         
         
Numerical Methods   : Closed form BSM analytical formula
    
    
Author              :Akshay Patil
"""

"""
Importing numpy, scipy libraries

"""
import numpy as np
import scipy.stats as stats


def BSM_value(S0, K, r, q, sigma, T, OptionInd):
    ''' Calculates Black-Scholes-Merton European call & put option value.

    Parameters
    ==========
    St : float
        stock/index level at time t
    K : float
        strike price
    t : float
        valuation date
    T : float
        maturity date
    r : float
        constant, risk-free interest rate
    q : float
        constant, time-continuous dividend yield
    sigma : float
            volatility
    OptionInd : integer
                1 - corresponds to Call Value
                0 - corresponds to Put value
                
    Returns
    =======
    option_value : float
        European call value or put value depending on the Option Indicator
    '''
    
    d1 = (np.log(S0/K) + (r - q + 0.5 * sigma**2) * (T))/sigma * np.sqrt(T)
    
    d2 = d1 - sigma * np.sqrt(T)
       
    if OptionInd == 1: 
    
         option_value = S0 * np.exp (-q * (T)) * stats.norm.cdf(d1) - \
                                 K * np.exp(-r * (T)) * stats.norm.cdf(d2)

    elif OptionInd == 0:
    
         option_value = -S0 * np.exp (-q * (T)) * stats.norm.cdf(-d1) + \
         K * np.exp(-r * (T)) * stats.norm.cdf(-d2)
    
    return round(option_value,6)
       




