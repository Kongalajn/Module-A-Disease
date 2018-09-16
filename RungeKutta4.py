# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 13:08:20 2018

@author: hasee
"""

import numpy as np


class RungeKutta4(object):
    def __init__(self, f1, f2):
        if not callable(f1):
            raise TypeError('f1 is %s, not a function' % type(f1))
        # For ODE systems, f will often return a list, but
        # arithmetic operations with f in numerical methods
        # require that f is an array. Let self.f be a function
        # that first calls f(u,t) and then ensures that the
        # result is an array of floats.
        self.f1 = lambda u1, t: np.asarray(f1(u1, t), float)
        
        self.f2 = lambda u2, t: np.asarray(f2(u2, t), float)

        
    def set_initial_condition(self, U0, c):
        if c == 1:
            U0 = np.asarray(U0)          # (assume U0 is sequence)
            self.size = U0.size
            self.U0 = U0
        
        
        if c == 2:
            U1 = np.asarray(U0[0])
            U2 = np.asarray(U0[1])
            self.size = U1.size
            self.U1 = U1
            self.U2 = U2
        
          
        
            
    def solve(self, time_points, d, terminate=None):
        """
        Compute solution u for t values in the list/array
        time_points, as long as terminate(u,t,step_no) is False.
        terminate(u,t,step_no) is a user-given function
        returning True or False. By default, a terminate
        function which always returns False is used.
        """
        
        if terminate is None:
            terminate = lambda u, t, step_no: False
               
        self.t = np.asarray(time_points)
               
        n = self.t.size
            
        if d == 1:
            self.u1 = np.zeros((n,self.size))
    
            # Assume that self.t[0] corresponds to self.U0
            self.u1[0] = self.U0 #initial numbers
    
            # Time loop
            for k in range(n-1):
                self.k = k
                self.u1[k+1] = self.advance1()
                #if terminate(self.u, self.t, self.k+1):
                   # break  # terminate loop over k
            return self.u1[:k+2], self.t[:k+2]  #[:k=2] means items from the beginning to the k+1's element
        
     
        if d == 2:
            self.u1 = np.zeros((n,self.size))
            self.u2 = np.zeros((n,self.size))
            
            self.u1[0] = self.U1
            self.u2[0] = self.U2
            
            for k in range(n-1):
                self.k = k
                self.u1[k+1] = self.advance1()
                self.u2[k+1] = self.advance2()
                
            return self.u1[:k+2], self.t[:k+2], self.u2[:k+2], self.t[:k+2]
            
      
            
            
            
    
class RK4(RungeKutta4):
    def advance1(self):
        u1, f1, k, t = self.u1, self.f1, self.k, self.t
        h = t[k+1] - t[k]
        dt = h/2.0
        K1 = h*f1(u1[k], t[k])
        K2 = h*f1(u1[k] + 0.5*K1, t[k] + dt)
        K3 = h*f1(u1[k] + 0.5*K2, t[k] + dt)
        K4 = h*f1(u1[k] + K3, t[k] + h)
        u_new = u1[k] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        return u_new
    
    def advance2(self):
        u2, f2, k, t = self.u2, self.f2, self.k, self.t
        h = t[k+1] - t[k]
        dt = h/2.0
        K1 = h*f2(u2[k], t[k])
        K2 = h*f2(u2[k] + 0.5*K1, t[k] + dt)
        K3 = h*f2(u2[k] + 0.5*K2, t[k] + dt)
        K4 = h*f2(u2[k] + K3, t[k] + h)
        u_new = u2[k] + (1/6.0)*(K1 + 2*K2 + 2*K3 + K4)
        return u_new
