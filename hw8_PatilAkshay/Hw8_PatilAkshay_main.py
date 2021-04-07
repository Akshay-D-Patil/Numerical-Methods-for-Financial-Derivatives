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


# Import important functions from libraries
from Hw8_PatilAkshay_EigenFunc import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    
    ## Pass function parameters given in question.
    N = 100
    dx_array = [0.050, 0.050, 0.040, 0.040]
    dt_array = [0.0010, 0.0015, 0.0010, 0.0015]
        

        
    for i in range(len(dx_array)):
        dx = dx_array[i]
        dt = dt_array[i]
        l = dt/ (dx**2)
        
        ## Get eigenvalue from functions for both the schemes
        # create empty dataframe to store output in iterations
        out_df = pd.DataFrame()
        
        eig_exp = explicit_scheme(l, N)
        eig_imp = implicit_scheme(l, N)
        
        # store the outputs in dataframe
        out_df["Exp_Analytical"] = eig_exp[0]
        out_df["Exp_Func"] = eig_exp[1]
        out_df["Imp_Analytical"] = eig_imp[0]
        out_df["Imp_Func"] = eig_imp[1]
        print("****************   eigenvalue-based stability analysis *********")
        print("1. Exp_Analytical and Imp_Analytical is calculated using the analytical formula")
        print("2. Exp_Func and Imp_Func using numpy.linalg.eigvals function\n")
        print("       dx = : {0},   dt ={1},      N = 100\n".format(dx_array[i],dt_array[i]))
        print(out_df)
        
        
        ####################### Process to plot the eigenvalues  ##################
        xx = range(1, len(eig_exp[0])+1 )
         

        
        fig = plt.figure()
   
#        Plot explicit scheme     
        
        explicit_fig = fig.add_subplot(211)
        explicit_fig.plot(xx, eig_exp[0],label="Analytical formula", c='b',linewidth=2.0)
        explicit_fig.plot(xx, eig_exp[1],label="numpy eigvals func", c='g')
        explicit_fig.set_title("\n Explicit Scheme"+r" ($\lambda$ = "+str(l)+")")
        explicit_fig.set_ylabel("eigenvalue")
        explicit_fig.set_xlabel(r'N')

        
        print("\n\n")
        
        
#        Plot implicit scheme
        implicit_fig = fig.add_subplot(212)
        implicit_fig.plot(xx, eig_imp[0],label="Analytical formula", c='b',linewidth=2.0)
        implicit_fig.plot(xx, eig_imp[1],label="numpy eigvals func", c='g')
        implicit_fig.legend(loc='upper left')
        implicit_fig.set_title("Implicit Scheme"+r" ($\lambda$ = "+str(l)+")")
        implicit_fig.set_ylabel("eigenvalue")
        implicit_fig.set_xlabel(r'N') 

          
        fig.subplots_adjust(hspace=0.5)

    plt.show()