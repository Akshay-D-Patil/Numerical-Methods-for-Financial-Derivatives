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


print('\n\n================The output will be displayed \
in 3 minutes and 30 seconds=====================')


#importing required functions from other python programs
from BSM import *
from explicit import *
from findifmethod import *
from thomas import *
from sor import *
from bren_schw import *
from psor import *
from Option_valuation import *

"""
main function to call the other functions
"""

if __name__ == '__main__':
    
    # Initializing parameters
    
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

    # Question 1
    
    # Calculating prices of European Call option using FDMs
    
    print('\n==================================================================\n')
    
    fdm_european_call = {'FDM   ':('Explicit','Implicit','Implicit','Crank-Nicholson',\
                                   'Crank-Nicholson'),
           'Algorithm':('Explicit','Thomas','SOR','Thomas','SOR'),
           'European Call':(exp_eur_call, imp_thom_eur_call, imp_sor_eur_call,
                            CN_thom_eur_call,  CN_sor_eur_call),
           'Error_Call':(abs(exp_eur_call-BSM_value_call), 
                        abs(imp_thom_eur_call-BSM_value_call),
                         abs(imp_sor_eur_call-BSM_value_call),
                         abs(CN_thom_eur_call-BSM_value_call),
                         abs(CN_sor_eur_call-BSM_value_call))}
           
            
    fdm_eur_call_output = pd.DataFrame(data = fdm_european_call)
    fdm_eur_call_output = fdm_eur_call_output[['FDM   ','Algorithm',\
                          'European Call','Error_Call']] # rearranging columns
    print('\n  Finite Difference Methods (FDMs) for European Call Option')
    print('\n',fdm_eur_call_output,'\n')    
    
    print('====================================================================')
    
    # Calculating prices of European Put option using FDMs
    
    fdm_european_put = {'FDM   ':('Explicit','Implicit','Implicit','Crank-Nicholson',\
                                  'Crank-Nicholson'),
           'Algorithm':('Explicit','Thomas','SOR','Thomas','SOR'),
           'European Put':(exp_eur_put, imp_thom_eur_put, imp_sor_eur_put,
                            CN_thom_eur_put,  CN_sor_eur_put),
           'Error_Put':(abs(exp_eur_put-BSM_value_put),
                        abs(imp_thom_eur_put-BSM_value_put),
                         abs(imp_sor_eur_put-BSM_value_put),
                         abs(CN_thom_eur_put-BSM_value_put),
                         abs(CN_sor_eur_put-BSM_value_put))}
            
    fdm_eur_put_output = pd.DataFrame(data = fdm_european_put)
    fdm_eur_put_output = fdm_eur_put_output[['FDM   ','Algorithm','European Put'\
                                             ,'Error_Put']] # rearranging columns
    print('\n Finite Difference Methods (FDMs) for European Put Option')
    print('\n',fdm_eur_put_output,'\n') 
    
    # Calculating prices of American options using FDMs

    print('====================================================================')
    
    fdm_amer = {'FDM   ':('Explicit','Implicit','Implicit','Crank-Nicholson',\
                          'Crank-Nicholson'),
           'Algorithm':('Explicit','Brennan','PSOR','Brennan','PSOR'),
           'American Call':(exp_amer_call,  imp_bren_amer_call,
                            imp_psor_amer_call, CN_bren_amer_call, CN_psor_amer_call),
           
           'American Put':(exp_amer_put, imp_bren_amer_put, imp_psor_amer_put,
                            CN_bren_amer_put,CN_psor_amer_put)}
            
    fdm_amer_output = pd.DataFrame(data = fdm_amer)
    fdm_amer_output = fdm_amer_output[['FDM   ','Algorithm','American Call',\
                                       'American Put']] # rearranging columns
    print('\n Finite Difference Methods (FDMs) for American Options')
    print('\n',fdm_amer_output,'\n')         



    
    # Calculating Closed Form solution for European option

    print('\n==================================================================')
    
    bs = {'European Call': BSM_value_call,
          'European Put': BSM_value_put}
    bs_output = pd.DataFrame(data = bs,index=[0])
    print('\n 2. Closed-form solution for European Options')
    print('\n',bs_output,'\n')
    
    print('==================================================================')
    
    

















