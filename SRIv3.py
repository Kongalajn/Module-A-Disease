# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 12:34:12 2018

@author: hasee
"""
import RungeKutta4
import numpy as np
import matplotlib.pyplot as plt

def f(u,t,beta=0.00001,v=0.1):
    S, I, R = u
    return [-beta*S*I, beta*S*I- v*I, v*I]

n=60

def solve():
    solution = RungeKutta4.RK4(f)
    solution.set_initial_condition([341456,1,0])# malmö population:341457

    time_points = np.linspace(0,60,n)
    u, t = solution.solve(time_points)

    tol = 1e-10
    S = u[:,0]; I = u[:,1]; R = u[:,2]
    for i in range(n):
        if abs(S[i] + I[i] + R[i]) - 341457 > tol: #check if the total population is fixed
            return # function will be terminated

    plt.plot(t,S/340000.0,'b-', label='RK4: susceptibles')
    plt.plot(t,I/340000.0,'r-', label='RK4: infected')
    plt.plot(t,R/340000.0,'g-', label='RK4: recovered')
    
solve()

plt.legend(loc='best')
plt.show()