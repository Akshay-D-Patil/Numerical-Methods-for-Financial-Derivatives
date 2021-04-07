#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

                                          Akshay Patil
                             MATH 6205 - Numerial Methods for Financial Derivatives
                                         Fall 2019 -UNCC Charlotte

"""

"""                         DRIVER FILE                                          """



"""Import important functions from libraries"""

import pandas as pd
from math import *
from scipy.stats import norm


"""                     Defining required functions                         """


"""
    F(x) is the standard Normal cumulative distribution
    This code approximate the standard Normal cumulative distribution.
"""

""" We use the formula for approximating standard Normal cumulative distribution """

def F(x, cdf_formula):
    if cdf_formula == "y":
        f_x = (1.0/ (sqrt(2*pi))) * exp(-(x*x)/2.0)
        z = 1.0/(1.0+ (0.2316419 * x))
        (a1,a2,a3,a4,a5)=(0.31938153,-0.356563782,1.781477937,-1.821255978,1.330274429)
    
        F_x = 1 - (f_x * z * ((((a5*z + a4)*z + a3)*z + a2)*z + a1))
        
        if x<0:
            z = 1.0/(1.0+ (0.2316419 * -x))
            F_x2 = 1 - (f_x * z * ((((a5*z + a4)*z + a3)*z + a2)*z + a1))
            F_x = 1.0 - F_x2
    
    """   We use scipy.stats.norm.cdf for approximating standard NCD   """    
    if cdf_formula == "n":       
        F_x = norm.cdf(x)
        
    return F_x



""" Defining d1 to calculate value of d1"""
def d1_value(S,K,T,t,r,sigma, delta):
    d1 = (log(S/K)+((r - delta + ((sigma*sigma)/2.0)) * (T-t))) / (sigma*sqrt(T-t))
    return d1  



""" Defining call option"""
def BS_call_option(S,K,T,t,r,sigma, delta, cdf_formula):
    """
    This function computes value of call option using Black-Scholes method.
    """
    d1 = d1_value(S,K,T,t,r,sigma, delta)
    d2 = d1 - (sigma*sqrt(T-t))
    return S*exp(-delta*(T-t))*F(d1, cdf_formula) - K * exp(-r*(T-t)) * F(d2, cdf_formula)



""" Defining put option"""
def BS_put_option(S,K,T,t,r,sigma, delta, cdf_formula):
    """
    This function computes value of Put option using Black-Scholes method.
    """
    d1 = d1_value(S,K,T,t,r,sigma, delta)
    d2 = d1 - (sigma*sqrt(T-t))
    return (-S*F(-d1, cdf_formula)*exp(-delta*(T-t))) + K * exp(-r*(T-t)) * F(-d2, cdf_formula)