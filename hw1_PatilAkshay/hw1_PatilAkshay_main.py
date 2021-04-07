#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

                                          Akshay Patil
                             MATH 6205 - Numerial Methods for Financial Derivatives
                                         Fall 2019 -UNCC Charlotte

Instructor: Dr Hwan Lin
---------- 

Purpose            : The Python code is to compute the prices of European Call and Put options 
-------              using the famous Black-Scholes formulas while allowing for two specifications
                     in the calculations of the standard normal cumulative distribution. The first
                     specification is provided byt the built-in funtion stats.norm.cdf of python. The
                     second specification which is in fact an approximation of the first one is computed
                     using the instructions of this very assignment. The parameters are given by an user
                     directly in the code, so the user must be proficient in Python. All the parameters are
                     constants, with the volatility sigma been a vector of constants.
                     
                     
Results:            : A comparison of results given by both speccification for 1 standard deviation away.
--------              And, a 6*10 table presenting the differents prices given by each specification and
                      the difference in both prices is negligible. 
                      

         
Numerical Methods   : The classical Black-Scholes equation for option pricing is used. 

Date of Completion and submission : September 10th, 2019

Files Included      : Python code (Main and Driver file) and Console output screenshot file
"""


"""                         MAIN FILE                                          """

"""Import important functions from libraries"""

import pandas as pd
from math import *
from scipy.stats import norm


"""                     Importing required functions from driver file                   """

from hw1_PatilAkshay_driver import F

from hw1_PatilAkshay_driver import d1_value

from hw1_PatilAkshay_driver import BS_call_option

from hw1_PatilAkshay_driver import BS_put_option


"""                     Defining required functions                         """


if __name__=="__main__":
    """
    S: stock price
    K: strike price
    T: maturity date
    r: risk-free interest rate
    delta: time-continuous dividend yield
    sigma: stock volatility
    """    
          
    """Define given values """
    S =100
    K =100
    T=1
    t=0
    delta = 0.025
    r =0.05
    
    sigma_array = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    """Create empty dataframes to store the results"""
    out_df1 = pd.DataFrame(columns=['S', 'K', 'T', 't', 'r','delta','sigma',\
                                    'CallValue','PutValue'])
    out_df2 = pd.DataFrame(columns=['S', 'K', 'T', 't', 'r','delta','sigma',\
                                    'CallValue','PutValue'])
    
    
    
    

    """
    1. We use the formula for approximating standard Normal cumulative distribution
      
    2. Run iterations on each elemnts of sigma array
    
   """
    
    
    for sigma in sigma_array:
        """Call call and put functions"""
        call_value = BS_call_option(S,K,T,t,r,sigma, delta, cdf_formula = "y")
        put_value =   BS_put_option(S,K,T,t,r,sigma, delta, cdf_formula = "y")
        
        """ store output in pandas dataframe at each iteration """
        out_df1 = out_df1.append({'S':S, 'K':K, 'T':T, 't':t,'r':r,'delta':delta,\
                                'sigma':sigma,'CallValue': call_value,\
                                'PutValue': put_value},ignore_index=True)
                              
    print('** Values of Call and Put options using classical Black-Scholes method **\n')
    print('******** Outcome using formula for calculating standard NCD ********\n')
    print(out_df1.round(2))






    """                            
    1. We use scipy.stats.norm.cdf function for approximating standard NCD
    2. Run iterations on each elemnts of sigma array
    """
    
    
    for sigma in sigma_array:
        """ Call call and put functions """
        call_value = BS_call_option(S,K,T,t,r,sigma, delta, cdf_formula = "n")
        put_value =   BS_put_option(S,K,T,t,r,sigma, delta, cdf_formula = "n")
        
        """ store output in pandas dataframe at each iteration """
        out_df2 = out_df2.append({'S':S, 'K':K, 'T':T, 't':t,'r':r,'delta':delta,\
                                'sigma':sigma,'CallValue': call_value,\
                                'PutValue': put_value},ignore_index=True)

    print( '\n**** Outcome using scipy.stats.norm.cdf for calculating standard NCD ****\n')
    print( out_df2.round(2))   

"""================================================================================================="""