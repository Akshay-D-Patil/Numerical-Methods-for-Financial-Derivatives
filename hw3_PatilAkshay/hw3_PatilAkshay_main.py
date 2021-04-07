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


"""
        Main file
"""


""" 
Notations:
    
    S0 = stock price(float)
    div = dividend yield
    mu = expected return
    sigma = volatility
    T = Maturity period = 1
    N = Time steps = 1000
    dt = discritized time domain = 1/N
    path = Number of Paths
"""

"""Importing functions from Sample path file"""

from hw3_PatilAkshay_def import *
#import pandas as pd


""" Given Info"""

S0=100
K=100
r = 0.03
delta = 0.025
sigma = 0.75
t = 0
T = 1
path = 1000
seed = 10


#We passed the required given information in the main function which helps us calculating  
#call option value and put option value using  Euler discretization method and compare it with analytical solution
    

N=100    # when delta(t) = 0.01 
output1 = main(S0,K,r,sigma,t,T,N,path,seed)

N=1000   # when delta(t) = 0.001 
output2 = main(S0,K,r,sigma,t,T,N,path,seed)

#Output in table format
output = output1.append(output2,ignore_index=True)
print(output)