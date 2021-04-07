

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

from Hw_2_sample_path import *



""" Given Info"""
#Model parameter 
mu = 0.1
sigma = 0.5
t = 0
T = 1


#Algorithmic parameter
N=1000
dt = (T-t)/N
path = 5

seed = 123
#We passed the required given information in the main function which helps us calculating and plotting the sample paths of 
#geometric brownian motion using Euler discretization method.

main(mu,sigma,t,T,N,dt,path,seed)



