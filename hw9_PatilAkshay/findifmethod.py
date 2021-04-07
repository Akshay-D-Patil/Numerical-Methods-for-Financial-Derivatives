"""
                             MATH 6205 - Numerical Methods for Financial Derivatives
                                                  Fall 2019


Purpose             : The objective of this Python program is to solve a 
                      tridiagonal system of equations using the different 
                      algorithms thereby pricing the european and american 
                      options. We discretize the heat equation. Once we discretize 
                      the heat equation,we create the mesh for space and time and 
                      then use that mesh as a tridiagonal system. The tridiagonal 
                      system can be solved using algorithms such asThomas, 
                      Brennan-Schwartz, SOR and PSOR. The different type of FDMs 
                      is decided by the value of the theta. 
                         
Numerical Methods   : Solving Tridiagonal system of equations, Thomas Algorithm,
                      Brennan-Schwartz Algorithm, SOR Algorithm, PSOR Algorithm. 
                      Finite Difference Methods such as Explicit, Implicit and
                      Crank-Nicholson.
                                           
Author              : Akshay Patil
"""

"""
Importing libraries

"""
import numpy as np
from scipy.sparse import spdiags
from scipy.interpolate import interp1d
from numba import jit
jit(nopython=True) # To enhance the speed of the algorithm

#importing required functions from other python programs
from thomas import *
from sor import *
from bren_schw import *
from psor import *


def fdm(S0,K,r,q,sigma,T,xmin,xmax,dx,dtau,theta,OptionInd,IterationInd,OptionType):
    '''
    Parameters
    ==========
    S0 : float
        stock price at time 0
    K  : float
        strike price
    T  : float
        maturity date
    r  : float
        constant, risk free rate
    q  : float
        constant, time-continuous dividend yield
    dx : float
         space step
    dtau : float
           time step
    sigma : float
            volatility
    theta : float
            0 - corresponds to Explicit FDM discretization
            1 - corresponds to Implicit FDM discretization
            0.5 - corresponds to Crank-Nicholson FDM discretization
    OptionInd : integer
                1 - corresponds to Call option value
                0 - corresponds to Put option value
    IterationInd : integer
                   1 - iterative solvers(SOR for European & PSOR for American options) 
                   0 - direct solvers(Thomas for European & Brennan-Schwartz for American)
    OptionType : integer
                 1 - corresponds to European type of options
                 0 - corresponds to American type of options
                
    Returns
    =======
    v : float
        Prices using different kinds of FDM discretizations for European & American options

    '''      
    
    N = int((xmax-xmin)/dx) #space length
    M = int((0.5*T*sigma**2)/dtau) #time length
    x = xmin + np.arange(N+1)*dx
    tau = np.arange(M+1)*dtau
    lamda = dtau / (dx ** 2)
    
    x_mesh = np.zeros((N+1,M+1)) #space mesh
    w_mesh = np.zeros((N+1,M+1)) #time mesh
    
    r1 = 2*r/sigma**2
    r2= 2*(r-q)/sigma**2
   
             
    if theta == 1: # Implicit FDM

        #setting up tridiagonal matrix         
        alpha = 1 + 2 * theta * lamda * np.ones(N-1)
        beta = -theta * lamda * np.ones(N-1)
        gamma = -theta * lamda * np.ones(N-1)
        mat = tridiagonal(alpha,beta,gamma,N-1)
        
        
        if OptionInd == 1: # Indicator for call option
            
            for i in range(M+1):
                
                c = np.maximum(np.exp(0.5*x*(r2+1))-np.exp(0.5*x*(r2-1)),0)
                
                x_mesh[:,i] = np.exp((0.25*(r2-1)**2+r1)*tau[i]) * c
            
            # boundary conditions
            w_mesh[:,0] = x_mesh[:,0]
            w_mesh[0,:] = x_mesh[0,:]
            w_mesh[N,:] = x_mesh[N,:]
            
            for j in range(M): # time loop
                
                a = x_mesh[1:N,j+1]
                b = np.zeros((N-1,1))
                
                for k in range(1,N):
                    
                    b[k-1] = w_mesh[k,j] + (1-theta)*lamda*(w_mesh[k-1,j]- \
                                                 2*w_mesh[k,j]+w_mesh[k+1,j])
                    
                    if k == N-1:
                        
                        b[k-1] = b[k-1] + theta*lamda*w_mesh[N,j+1]
                        
                    elif k == 1:
                        
                        b[k-1] = b[k-1] + theta*lamda*w_mesh[0,j+1]
                                       
                
                if IterationInd == 0: # Indicator for non-iterative(direct) solver
                    
                    # Indicator for solving european type of options(Thomas algorithm)    
                    if OptionType == 1: 
                        
                        w_mesh[1:N,j+1] = thomas(alpha,gamma,beta,b,N-1)
                        
                    # Indicator for solving american type of options(brennan algorithm)    
                    elif OptionType == 0: 
                        
                        w_mesh[1:N,j+1] = brennan(alpha,gamma,beta,a,b,N-1)
                
                elif IterationInd == 1: # Indicator for iterative solver
                    
                    accuracy = 6 # tolerance level
                    
                    # Indicator for solving european type of options(sor algorithm)    
                    if OptionType == 1: 
                        
                        w_mesh[1:N,j+1] = sor(b,mat,N-1,accuracy)
                        
                        
                     # Indicator for solving american type of options(psor algorithm)    
                    elif OptionType == 0:
                        
                        w_mesh[1:N,j+1] = psor(a,b,mat,N-1,accuracy)
                    
                    
            price = K * w_mesh[:,M] * np.exp(-0.5*(r2-1)*x - \
                                                     (0.25*(r2-1)**2+r1)*tau[M])
            
            price_intra = interp1d(K*np.exp(x),price) # interpolating the option price
            
            v = price_intra(S0).item()
            
            return round(v,6)
                    
        elif OptionInd == 0: # Indicator for put option
            
            for i in range(M+1):
                
                p = np.maximum(np.exp(0.5*x*(r2-1))-np.exp(0.5*x*(r2+1)),0)
                
                x_mesh[:,i] = np.exp((0.25*(r2-1)**2+r1)*tau[i]) * p
            
            # boundary conditions
            w_mesh[:,0] = x_mesh[:,0]
            w_mesh[0,:] = x_mesh[0,:]
            w_mesh[N,:] = x_mesh[N,:]
            
            for j in range(M): # time loop
                
                a = x_mesh[1:N,j+1]
                b = np.zeros((N-1,1))
                
                for k in range(1,N):
                    
                    b[k-1] = w_mesh[k,j] + (1-theta)*lamda*(w_mesh[k-1,j]- \
                                                     2*w_mesh[k,j]+w_mesh[k+1,j])
                    
                    if k == N-1:
                        
                        b[k-1] = b[k-1] + theta*lamda*w_mesh[N,j+1]
                        
                    elif k == 1:
                        
                        b[k-1] = b[k-1] + theta*lamda*w_mesh[0,j+1]
                
                
                if IterationInd == 0: # Indicator for non-iterative(direct) solver
                    
                    # Indicator for solving european type of options(Thomas algorithm)   
                    if OptionType == 1:
                        
                        w_mesh[1:N,j+1] = thomas(alpha,gamma,beta,b,N-1)
                        
                        
                    # Indicator for solving american type of options(brennan algorithm)    
                    elif OptionType == 0: 
                        
                        w_mesh[1:N,j+1] = brennan(alpha,gamma,beta,a,b,N-1)
                
                elif IterationInd == 1: # Indicator for iterative solver
                    
                    accuracy = 6 # tolerance level
                    
                  # Indicator for solving european type of options(sor algorithm)
                    if OptionType == 1: 
                        
                        w_mesh[1:N,j+1] = sor(b,mat,N-1,accuracy)
                        
                  # Indicator for solving american type of options(psor algorithm)     
                    elif OptionType == 0: 
                        
                        w_mesh[1:N,j+1] = psor(a,b,mat,N-1,accuracy)
            
            price = K * w_mesh[:,M] * np.exp(-0.5*(r2-1)*x - \
                                              (0.25*(r2-1)**2+r1)*tau[M])
            
            price_intra = interp1d(K*np.exp(x),price) # interpolating the option price
            
            v = price_intra(S0).item()
            
            return round(v,6)

    elif theta == 0.5: # Crank - Nicholson FDM

        #setting up tridiagonal matrix         
        alpha = 1 + 2 * theta * lamda * np.ones(N-1)
        beta = -theta * lamda * np.ones(N-1)
        gamma = -theta * lamda * np.ones(N-1)
        mat = tridiagonal(alpha,beta,gamma,N-1)
                

        
        if OptionInd == 1: # Indicator for call option
            
            for i in range(M+1):
                
                c = np.maximum(np.exp(0.5*x*(r2+1))-np.exp(0.5*x*(r2-1)),0)
                
                x_mesh[:,i] = np.exp((0.25*(r2-1)**2+r1)*tau[i]) * c
            
            # boundary conditions
            w_mesh[:,0] = x_mesh[:,0]
            w_mesh[0,:] = x_mesh[0,:]
            w_mesh[N,:] = x_mesh[N,:]
            
            for j in range(M): # time loop
                
                a = x_mesh[1:N,j+1]
                b = np.zeros((N-1,1))
                
                for k in range(1,N):
                    
                    b[k-1] = w_mesh[k,j] + (1-theta)*lamda*(w_mesh[k-1,j]- \
                                                 2*w_mesh[k,j]+w_mesh[k+1,j])
                    
                    if k == N-1:
                        
                        b[k-1] = b[k-1] + theta*lamda*w_mesh[N,j+1]
                        
                    elif k == 1:
                        
                        b[k-1] = b[k-1] + theta*lamda*w_mesh[0,j+1]
                    
                  
                if IterationInd == 0: # Indicator for non-iterative(direct) solver
                    
                    # Indicator for solving european type of options(Thomas algorithm)    
                    if OptionType == 1: 
                        
                        w_mesh[1:N,j+1] = thomas(alpha,gamma,beta,b,N-1)
                    
                    
                    # Indicator for solving american type of options(brennan algorithm)
                    elif OptionType == 0: 
                        
                        w_mesh[1:N,j+1] = brennan(alpha,gamma,beta,a,b,N-1)
                
                elif IterationInd == 1:  # Indicator for iterative solver
                    
                    accuracy = 6  # tolerance level
                    
                    # Indicator for solving european type of options(sor algorithm)    
                    if OptionType == 1:
                        
                        w_mesh[1:N,j+1] = sor(b,mat,N-1,accuracy)
                    
                    
                    # Indicator for solving american type of options(psor algorithm)
                    elif OptionType == 0:
                        
                        w_mesh[1:N,j+1] = psor(a,b,mat,N-1,accuracy)
                        
            price = K * w_mesh[:,M] * np.exp(-0.5*(r2-1)*x - (0.25*(r2-1)**2+r1)*tau[M])
            
            price_intra = interp1d(K*np.exp(x),price) # interpolating the option price
            
            v = price_intra(S0).item()
            
            return round(v,6)
        
        # Indicator for put option            
        elif OptionInd == 0:
            
            for i in range(M+1):
                
                p = np.maximum(np.exp(0.5*x*(r2-1))-np.exp(0.5*x*(r2+1)),0)
                
                x_mesh[:,i] = np.exp((0.25*(r2-1)**2+r1)*tau[i]) * p
            
            # boundary conditions
            w_mesh[:,0] = x_mesh[:,0]
            w_mesh[0,:] = x_mesh[0,:]
            w_mesh[N,:] = x_mesh[N,:]
            
            for j in range(M): # time loop
                
                a = x_mesh[1:N,j+1]
                b = np.zeros((N-1,1))
                
                for k in range(1,N):
                    
                    b[k-1] = w_mesh[k,j] + (1-theta)*lamda*(w_mesh[k-1,j]- \
                                                     2*w_mesh[k,j]+w_mesh[k+1,j])
                    
                    if k == N-1:
                        
                        b[k-1] = b[k-1] + theta*lamda*w_mesh[N,j+1]
                        
                    elif k == 1:
                        
                        b[k-1] = b[k-1] + theta*lamda*w_mesh[0,j+1]

                
                if IterationInd == 0: # Indicator for non-iterative(direct) solver
                    
                    # Indicator for solving european type of options(thomas algorithm)    
                    if OptionType == 1:  
                        
                        w_mesh[1:N,j+1] = thomas(alpha,gamma,beta,b,N-1)
                        
                        
                    # Indicator for solving american type of options(brennan algorithm)    
                    elif OptionType == 0: 
                        
                        w_mesh[1:N,j+1] = brennan(alpha,gamma,beta,a,b,N-1)
                
                elif IterationInd == 1:  # Indicator for iterative solver
                    
                    accuracy = 6  # tolerance level
                    
                    # Indicator for solving european type of options(sor algorithm)    
                    if OptionType == 1: 
                        
                        w_mesh[1:N,j+1] = sor(b,mat,N-1,accuracy)
                    
                    
                    # Indicator for solving american type of options(psor algorithm)
                    elif OptionType == 0: 
                        
                        w_mesh[1:N,j+1] = psor(a,b,mat,N-1,accuracy)
            
            price = K * w_mesh[:,M] * np.exp(-0.5*(r2-1)*x - (0.25*(r2-1)**2+r1)*tau[M])
            
            price_intra = interp1d(K*np.exp(x),price) # interpolating the option price
            
            v = price_intra(S0).item()
            
            return round(v,6)
        


