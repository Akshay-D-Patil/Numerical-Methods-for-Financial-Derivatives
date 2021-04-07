
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
import pandas as pd
from Hw5_PatilAkshay_def import *

S0 = 100
K = 100
path = 1000
tstep = 100
t=0
T=1
dt = T/tstep
seed = 123
r = 0.03
sigma = 0.75



tstep=100    # when delta(t) = 0.01 
output1 = main(S0,K,r,sigma,t,T,tstep,path,seed)

tstep=1000   # when delta(t) = 0.001 
output2 = main(S0,K,r,sigma,t,T,tstep,path,seed)

#Output in table format
output = output1.append(output2,ignore_index=True)

print(output)


