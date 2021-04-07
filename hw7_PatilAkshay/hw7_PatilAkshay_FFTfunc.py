
"""
                             MATH 6205 - Numerical Methods for Financial Derivatives
                                                  Fall 2019
                                            Author: AkshayPatil 
                                            Stud ID : 801034919
                                            Email id : apatil17@uncc.edu


Purpose          : The objective of this Python program is to compute the prices of 
                   European calls and puts using Fourier transform techniques. 
                   Geometric Brownian Motion as the underlying diffusion process 
                   for the Stock price. Under risk neutral evaluation, the Brownian 
                   Motion is transformed in such a way that we can make use of fourier 
                   transform provied the characteristic function is known. The pricing 
                   integrals using the Foruier transform and inverse Fourier transform 
                   are derived and then the summation is approximated using the Trapezoidal 
                   rule. Half frequency domain and Full frequency domain are used to compute 
                   the option prices. The algorithm gives us the European call or put option 
                   prices based on the dampening factor(alpha) give. For a positive alpha, 
                   it gives us the European call option price where for negative aplha, we 
                   will get the European put option price.The input parameters required are 
                   provided by the user.
                  
         
Numerical Methods: Pricing Integrals using Fourier Transform and Inverse Fourier Transform 
                   are used. Trapezoid rule is used for the summation.Conjugate properties 
                   are also used.
                      
"""


#Importing numpy, scipy, and pandas libraries


import numpy as np
from scipy import interpolate


#Function to calculate the characteristic function 

def characteristic_fn(alpha, S0, r, T, w, sigma):
    '''

    Parameters
    ==========
    alpha : float
          damping factor
    S0 : float
        stock price at time 0
    T : float
        maturity date
    r : float
        constant, risk free rate
    w : float
        frequency
    sigma : float
            volatility
            
    ''' 
    
    w_new =  w + (alpha+1)*1j
    
    fn = np.exp(-r*T) * np.exp((-1j*(np.log(S0)+\
                                (r - sigma**2/2)*T)*w_new)-sigma**2/2 *T * w_new**2)
    
    return fn


#Function to calculate the denominator of the v function


def denominator(alpha,w):
    
    d = (alpha-1j*w) * (alpha-1j*w+1)
    
    return d  


#Function to calculate the fourier transformation based on the Trapezoid algorithm


def fourier(alpha, S0, K, r, T, sigma, Kmin, B, N):
    '''

    Parameters
    ==========
    alpha : float
          damping factor
    S0 : float
        stock price at time 0
    K : float
        strike price
    Kmin: float
         minimum strike price
    T : float
        maturity date
    r : float
        constant, risk free rate
    B : float
        limit of bandwidth
    N : integer
        # of frequencies
    sigma : float
            volatility
        
    '''
    w = 0
    h = B/(N-1)
    k_min = np.log(Kmin)
    dk = 2*np.pi /(h*N)
    
    k = np.zeros(N,dtype='float128')
    v = np.zeros(N,dtype='float128')
    fou_transf = np.zeros(N,dtype='complex128')
    fou_transf_inv = np.zeros(N,dtype='complex128')
    
        
    for m in range(N):         
        w = h*m
        if m==0:
            dw = h/2
        else:
            dw = h
            
        fou_transf[m] = np.exp(1j*w*k_min) * dw * characteristic_fn(alpha, \
                  S0, r, T, w, sigma) / denominator(alpha,w)
        
    fou_transf_inv = np.fft.ifft(fou_transf)
    
    for i in range(N):
        k[i] = k_min + i*dk
        x = np.exp(-alpha*k[i])/np.pi
        
        for j in range(N):
            fou_transf_inv[i] = fou_transf_inv[i] + \
                                fou_transf[j] * np.exp(2*np.pi*1j*j*i/N) 
            
        v[i] = x * np.real(fou_transf_inv[i])
        
    graph_scale = max(v)/S0
    
    return k,v,graph_scale



def approx(K, k_val, v_val):
    '''

    Parameters
    ==========
    K : float
        strike price
    k_val: array
           array of  strike prices
    v_val : array
           array of option prices
       
    '''
     # Fir Interpolate function
    interp_fun = interpolate.interp1d(np.exp(k_val),v_val)
    
    # Get fitted price at given K    
    v_approx = interp_fun(K)
    
    return v_approx

   


        
        
        