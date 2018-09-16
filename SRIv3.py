# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 12:34:12 2018

@author: hasee
"""
import RungeKutta4
import numpy as np
import matplotlib.pyplot as plt

T_M=1/(S_M+I_M+R_M)

def f(u,t,beta=0.000013,alpha=0.1,gamma=0.3):
    S, I, R = u
    return [[S_M*(-beta*I_M - gamma + LM*T_M - MK*T_M), I_M*(beta*S_M- alpha + LM*T_M - MK*T_M), R_M(gamma*S_M + alpha*I_M + LM*T_M - MK*T_M)], #Malmö
            [-beta*S*I - gamma*S - LM*T, beta*S*I- alpha*I - LM*T, gamma*S + alpha*I - LM*T], #Lund
            [-beta*S*I - gamma*S + MK*T, beta*S*I- alpha*I + MK*T, gamma*S + alpha*I +MK*T]] #Köpenhamn

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
