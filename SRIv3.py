# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 12:34:12 2018

@author: hasee
"""
import RungeKutta4
import numpy as np
import matplotlib.pyplot as plt

def f(u,t,beta=0.000013,alpha=0.1,gamma=0.3):
    S, I, R = u
    return [-beta*S*I - gamma*S, beta*S*I- alpha*I, gamma*S + alpha*I]

n=60

def solve():
    solution = RungeKutta4.RK4(f)
    solution.set_initial_condition([341456,1,0])# malmö population:341457

    time_points = np.linspace(0,60,n)
    u, t = solution.solve(time_points)

    S = u[:,0]; I = u[:,1]; R = u[:,2]
    
    plt.plot(t,S/340000.0,'b-', label='RK4: susceptibles')
    plt.plot(t,I/340000.0,'r-', label='RK4: infected')
    plt.plot(t,R/340000.0,'g-', label='RK4: recovered')
    
solve()

plt.legend(loc='best')
plt.show()
