
"""
                             MATH 6205 - Numerical Methods for Financial Derivatives
                                                  Fall 2019
                  
         
Numerical Methods   : Pricing Integrals using Fourier Transform and Inverse Fourier Transform are used. Trapezoid rule
                      is used for the summation.Conjugate properties are also used.
                      
Author              : Akshay Patil

"""

"""
Importing libraries and functions
"""
import numpy as np
import pandas as pd

from Hw6_PatilAkshay_def import *

"""
main function to call the function
"""

if __name__ == '__main__':
    
    S0 = 100
    K  = 80
    T  = 1
    r  = 0.05
    B  = 50
    N  = 1000
    sigma = 0.5
    alpha = np.array([2.5,5,10]) # an array of alpha values for call options


    # Creating a tablular format for call values

    call_values = {'alpha':np.array([2.5,5,10]),'Full Frequency Domain': fourier(alpha, S0, K, r, T, sigma, B, N, 1) , 
              'Half Frequency Domain':fourier(alpha, S0, K, r, T, sigma, B, N, 0)}

    call_output = pd.DataFrame(data = call_values,index=np.array([2.5,5,10]))

    call_output.set_index('alpha',inplace=True)

    print('\n Table 1: European Call option values using Fourier Transform for different frequency domains(alpha)')

    print('\n',call_output,'\n')
    
    alpha = np.array([-2.5,-5,-10]) # an array of alpha values for put options

    # Creating a tablular format for put values

    put_values = {'alpha':np.array([-2.5,-5,-10]),'Full Frequency Domain': fourier(alpha, S0, K, r, T, sigma, B, N, 1) , 
              'Half Frequency Domain':fourier(alpha, S0, K, r, T, sigma, B, N, 0)}

    put_output = pd.DataFrame(data = put_values,index=np.array([-2.5,-5,-10]))

    put_output.set_index('alpha',inplace=True)

    print('\n Table 2: European Put option values using Fourier Transform for different frequency domains(alpha)')

    print('\n',put_output)
