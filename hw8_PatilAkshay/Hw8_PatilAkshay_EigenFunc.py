"""
                             MATH 6205 - Numerical Methods for Financial Derivatives
                                                  Fall 2019
                                            Author: AkshayPatil 
                                            Stud ID : 801034919
                                            Email id : apatil17@uncc.edu


Purpose          : The objective of this Python program is to compute the solution 
                   of a tridiagonal matrix using the finite differences method. 
                   WE use the heat equation to solve for explicit or implicit
                   solution. Based on the given solution, we can discretize 
                   the linear system of difference equations at given point of 
                   time using the finite difference method. The lambda parameter
                   is given by the user. prices of European calls and puts using  
         
Numerical Methods: Heat equation is used to solve for explicit or implicit solution. 
                   Solving Tridiagonal system of equations. Finite Difference method 
                   to solve linear system of equations is used.
                      
"""


import numpy as np

from math import *


def A_matrix(N, l, scheme):
    '''
                The explicit scheme        
    
    w_i,i+1 = lambda * w_j-1,i + (1-2lambda) * w_j,i + lambda * w_j+1,i
    
    In matrix form, 
                    w^(i+1) = A_R * w^i
    
                              AR = (N-1) x (N-1)
                              
   ------------------------------------------------------------------------- 
                The implicit Scheme        
    
    w_j,i+1 - w_j,i     w_j-1,i+1 - 2*w_j,i+1 + w_j+1,i+1
    ---------------  =  ---------------------------------
           dt                         (dx)^2
    
   
                    AL = (N-1) x (N-1)
                    
    where,             
                    
   lambda = dt/(dt)^2
    '''
    
    ## Create zero matrix of (N-1) x (N-1)
    A = np.zeros((N-1, N-1))
        
    if scheme == "explicit":
        ## insert lambda and (1-2lambda) in appropriate postion in A      
        A[ [range(0, N-1), range(0, N-1)] ]  = 1-(2*l)
        A[ [range(1, N-1), range(0, N-1-1)] ]  = l
        A[ [range(0, N-1-1), range(1, N-1)] ]  = l
    
    if scheme == "implicit":
        ## insert lambda and (1-2lambda) in appropriate postion in A      
        A[ [range(0, N-1), range(0, N-1)] ]  = 1+(2*l)
        A[ [range(1, N-1), range(0, N-1-1)] ]  = -l
        A[ [range(0, N-1-1), range(1, N-1)] ]  = -l
        
    return A
    
    
def eign_analy_sol(N, l, scheme):
    '''
    Let G be a KxK tridiagonal matrix:

    The eigenvalues are:
    
    eigenvalues  = alpha + (2*beta*(sqrt(gamma/beta)) * cos( (k*pi)/ (K+1)))    
    
    k = 1,.....,K
    '''
    
    if scheme == "explicit":
        alpha = 1-(2*l)
        beta = gamma = l
    
    if scheme == "implicit":
        alpha = 1+(2*l)
        beta = gamma = -l
        
    K= N-1  
    k = np.arange(1, N)
    
    eig_vec = alpha + (2*beta*(np.sqrt(gamma/beta)) * np.cos( (k*np.pi)/ (K+1)))        
    
    return eig_vec
    

def explicit_scheme(l, N):
    ''' The explicit scheme '''   
    ## Get eigenvalues using analytical formula
    eig_vals1 = eign_analy_sol(N, l, "explicit")
    #eig_vals1 = sorted(eig_vals1)
    
    ## Create AR matrix
    A = A_matrix(N,l, "explicit")
    
    ## Get eigenvalues using numpy function
    eig_vals2 = np.linalg.eigvals(A)
    #eig_vals2 = sorted(eig_vals2)
    
    return eig_vals1, eig_vals2


def implicit_scheme(l, N):
    ''' The implicit scheme '''   
    ## Get eigenvalues using analytical formula
    eig_vals1 = eign_analy_sol(N, l, "implicit")
    #eig_vals1 = sorted(eig_vals1)
    
    ## Create AL matrix
    A = A_matrix(N, l, "implicit")
    A =  np.linalg.inv(A)
    ## Get eigenvalues using numpy function
    eig_vals2 = np.linalg.eigvals(A)
    #eig_vals2 = sorted(eig_vals2)
        
    return eig_vals1, eig_vals2
    

    