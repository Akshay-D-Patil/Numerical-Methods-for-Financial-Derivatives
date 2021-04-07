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
import pandas as pd
from Hw4_PatilAkshay_PayoffFunc import *

#========================================================================================
# Main function to calculate values of american call and put using Regression 1 method
#========================================================================================

def main(S0,K,r,sigma,t,T,tstep,path,seed):
    dt = (T-t)/tstep
    out_df = pd.DataFrame(columns=['delta_t','American_Call','American_Put'])
    
    [V_call,V_put] = max_payoff(S0,K,r,sigma,t,T,tstep,path,seed)
    call_opt = np.exp(-r*dt)*np.mean(V_call[:,0])
    put_opt = np.exp(-r*dt)*np.mean(V_put[:,0])
    out_df = out_df.append({'delta_t':dt,'American_Call':call_opt,'American_Put':put_opt},ignore_index=True)
    
    return out_df





    