"""
                             MATH 6205 - Numerical Methods for Financial Derivatives
                                                  Fall 2019


Purpose             : The objective of this Python program is to compute the 
                      prices of European and American calls and puts using 
                      Finite Difference Methods(FDMs) and the closed form 
                      solutions. Every method has its pros and cons. We have 
                      used Explicit, Implicit and Crank-Nicholson discretization
                      methods to solve using differing algorithms such as Thomas, 
                      Brennan-Schwartz, SOR and PSOR. We have seen that American 
                      options are costlier than European counterparts using different 
                      methods. Also, we have calculated the absolute difference 
                      compared to the closed form, Black Scholes solution. 
         
Numerical Methods   : Finite Difference Methods(FDMs) are used to discretize the 
                      heat equation. Explicit Method, Implicit Method and 
                      Crank-Nicholson Methods of FDM are used. Solving 
                      Tridiagonal system of equations, Thomas Algorithm,
                      SOR Algorithm, Brennan-Schwartz Algorithm, PSOR Algorithm.
                      
Author              : Akshay Patil
"""

"""
Importing libraries and functions
"""
import numpy as np
import pandas as pd
from numba import jit
jit(nopython=True) # To enhance the speed of the algorithm

#importing required functions from other python programs


from BSM import *
from explicit import *
from findifmethod import *
from thomas import *
from sor import *
from bren_schw import *
from psor import *

S0 = 100
K  = 100 
T  = 1
r  = 0.02
q = 0.01
sigma = 0.6
dx = 0.05
dt = 0.00125
dtau = 0.00125
xmin = -2.5
xmax = 2.5

#Closed form European Call value
BSM_value_call = BSM_value(S0, K, r, q, sigma, T, 1)


#Closed form European Put value
BSM_value_put = BSM_value(S0, K, r, q, sigma, T, 0)

#=================================================================================

#Calculating European Call by Explicit FDM method
exp_eur_call = expl(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0, 1, 1)


#Calculating European Call by Implicit FDM and Thomas alogorithm
imp_thom_eur_call = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 1, 0, 1)

 
 #Calculating European Call by Implicit FDM and SOR alogorithm
imp_sor_eur_call = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 1, 1, 1)


 #Calculating European Call by Crank Nicholsan FDM and Thomas alogorithm
CN_thom_eur_call = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 1, 0, 1)  
   

#Calculating European Call by Crank Nicholsan FDM and SOR alogorithm
CN_sor_eur_call = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 1, 1, 1)      

#=================================================================================

#Calculating European Put by Explicit FDM method
exp_eur_put = expl(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0, 0, 1)


#Calculating European Put by Implicit FDM and Thomas alogorithm
imp_thom_eur_put = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 0, 0, 1)

 
 #Calculating European Put by Implicit FDM and SOR alogorithm
imp_sor_eur_put = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 0, 1, 1)


 #Calculating European Put by Crank Nicholsan FDM and Thomas alogorithm
CN_thom_eur_put = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 0, 0, 1)  
   

#Calculating European Put by Crank Nicholsan FDM and SOR alogorithm
CN_sor_eur_put = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 0, 1, 1)   

#=================================================================================

#Calculating American Call by Explicit FDM method
exp_amer_call = expl(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0, 1, 0)


#Calculating American Call by Implicit FDM and Brennan alogorithm
imp_bren_amer_call = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 1, 0, 0)

 
 #Calculating American Call by Implicit FDM and PSOR alogorithm
imp_psor_amer_call = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 1, 1, 0)


 #Calculating American Call by Crank Nicholsan FDM and Brennan alogorithm
CN_bren_amer_call = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 1, 0, 0) 
   

#Calculating American Call by Crank Nicholsan FDM and PSOR alogorithm
CN_psor_amer_call = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 1, 1, 0)      

#=================================================================================


#Calculating American Put by Explicit FDM method
exp_amer_put = expl(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0, 0, 0)


#Calculating American Put by Implicit FDM and Brennan alogorithm
imp_bren_amer_put = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 0, 0, 0)

 
 #Calculating American Put by Implicit FDM and PSOR alogorithm
imp_psor_amer_put = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 1, 0, 1, 0)


 #Calculating American Put by Crank Nicholsan FDM and Brennan alogorithm
CN_bren_amer_put = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 0, 0, 0) 
   

#Calculating American Put by Crank Nicholsan FDM and PSOR alogorithm
CN_psor_amer_put = fdm(S0, K, r, q, sigma, T, xmin, xmax, dx, dtau, 0.5, 0, 1, 0)  

#=================================================================================