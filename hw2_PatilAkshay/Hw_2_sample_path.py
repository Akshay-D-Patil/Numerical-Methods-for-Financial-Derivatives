

"""
                            MATH 6205 - Numerial Methods for Financial Derivatives
                                         Fall 2019 -UNCC Charlotte

                                            Author: AkshayPatil 
                                            Stud ID : 801034919
                                            Email id : apatil17@uncc.edu
                                            
                                            
PURPOSE : To simulate the path of Stock Price  
 
Numerical Method Used : Euler discretization of a Stochastic differential equation. 

Date of Completition : 09/19/2019 

"""

"""
        Sample Path File
"""


"""Importing numpy and matplotlib.pyplot libraries"""

import numpy as np
import matplotlib.pyplot as plt


"""Set the seed as 123 to generate random numbers"""
#np.random.seed(seed=123) 



def main(mu,sigma,t,T,N,dt,path,seed):
  
    sample_path(mu,sigma,t,T,N,dt,path,seed)

"""
Defining the function for sample path which calculates the values of St using 
   Euler discretization method
"""

def sample_path(mu,sigma,t,T,N,dt,path,seed):
    #To calculate value of St using Euler discretization method
    np.random.seed(seed=seed) 
    # Line 49 creates an matrix named S(t) which is of shape (path * (N+1))
    #In our case it creates a matrix of (5 * 1001)
    St = np.zeros((path,N+1))
    
    # There are 1001 columns because we use first column to \
    # store the initial value of S(t) at t=0, in our case S90) = 100
    St[:,0]=100
    
    #Following for loop runs N times, as it calculates the values of S(t) and stores them in the matrix S(t)
    for i in range(1,N+1):
        
#        Line 65: Computing wiener process 
#        Define N(0,1) to be the standard random variable that is normally distributed 
#        with mean 0 and standard deviation 1. Each random number ∆Wi is computed as 
#        ∆Wi =  Zi* sqrt(∆ti)
#        where zi is chosen from N(0,1)

        dWt = (np.random.standard_normal(path))*np.sqrt(dt)
        
#         Line 69: The Euler discretization of the SDE is given as:
#                   S(t) = S(t-1) + μ*S(t-1)*dt + σ*S(t)*dWt        
        St[:,i] = St[:,i-1] + mu*St[:,i-1]*dt + sigma*St[:,i-1]*dWt
    
    #To plot 2-D sample path
    #Y axis ( time line from 0 to 1 into N parts )
    discretization_steps = np.arange(t,T + dt,dt)#.reshape(N+1,1)
    
#    """Output : 2- D graph of given sample paths"""
#    We use the transpose of St, so as to match the first dimensions of x and y passed in plt.plot(x,y)
    plt.plot(discretization_steps,St.T);
    
    # Label path axis  
    plt.xlabel('time(t)')
    plt.ylabel('S(t)')
    plt.title('Sample path of GBM based on Euler discretization method')
    
    #Grid the plot for better visualization 
    plt.grid()
    plt.show()
    

    
