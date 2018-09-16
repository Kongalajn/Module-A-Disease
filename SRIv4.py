# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 20:09:02 2018

@author: hasee
"""

import RungeKutta4
import numpy as np
import matplotlib.pyplot as plt

#1 represents Malmö
#2 represents Copenhagen
#3 represents Lund

def f1(u1, t,beta=4.5/341457.0,alpha=0.1,gamma=0.3):
    S1, I1, R1 = u1
    
    return [-beta * S1 * I1 - gamma * S1, #+ 0.01 * S2 - 0.001 * S1, 
            beta * S1 * I1- alpha * I1,# + 0.01 * I2 - 0.001 * I1, 
            gamma * S1 + alpha * I1]# + 0.01 * R2 - 0.001 * R1]

def f2(u2,t,beta=4.5/775033.0,alpha=0.1,gamma=0.3):
    
    S2, I2, R2 = u2
    return [-beta * S2 * I2 - gamma * S2, 
            beta * S2 * I2- alpha * I2, 
            gamma * S2 + alpha * I2]    
             

n=60   # number of days

def solve(j):
    if j == 1:
        solution = RungeKutta4.RK4(f1, f2)
        solution.set_initial_condition([341456,1,0],1)# malmö population:341457
    
        time_points = np.linspace(0,60,n) #unit time is 1 day
        u1, t = solution.solve(time_points,1)
    
        S1 = u1[:,0]; I1 = u1[:,1]; R1 = u1[:,2]
        
        plt.plot(t,S1/340000.0,'b-', label='Susceptibles')
        plt.plot(t,I1/340000.0,'r-', label='Infected')
        plt.plot(t,R1/340000.0,'g-', label='Recovered')
    if j == 2:
        solution = RungeKutta4.RK4(f1, f2)
        solution.set_initial_condition([[341456,1,0],[775033,1,0]],2)
        
        time_points = np.linspace(0,60,n) #unit time is 1 day
        u1, t1, u2, t2 = solution.solve(time_points,2)
        
        S1 = u1[:,0]; I1 = u1[:,1]; R1 = u1[:,2]
        
        plt.plot(t1,S1/340000.0,'b-', label='Susceptibles')
        plt.plot(t1,I1/340000.0,'r-', label='Infected')
        plt.plot(t1,R1/340000.0,'g-', label='Recovered')
    
solve(2)

plt.legend(loc='best')
plt.show()