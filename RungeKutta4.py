import numpy as np


class RungeKutta4(object):
    def __init__(self, f1, f2):
        if not callable(f1):
            raise TypeError('f1 is %s, not a function' % type(f1))

        self.f1 = lambda u1, t: np.asarray(f1(u1, t), float) #u1, u2 is the odes given by user
        
        self.f2 = lambda u2, t: np.asarray(f2(u2, t), float)
        
        #make the input f as an array

        
    def set_initial_condition(self, U0, c):
        if c == 1:                          #if one city is considered
            U0 = np.asarray(U0)             #in our project U0 will contain 3 elements: number of susceptible, no. of infected & no. of recovered
            self.size = U0.size             #3
            self.U0 = U0
        
        
        if c == 2:                         #if 2 cities are considered
            U1 = np.asarray(U0[0])
            U2 = np.asarray(U0[1])
            self.size = U1.size
            self.U1 = U1
            self.U2 = U2
        
          
        
            
    def solve(self, steps, d, terminate=None): #d is the number of cities
      
        
        if terminate is None:
            terminate = lambda u, t, steps: False #as long as it's false, we can do the calculations
               
        self.t = np.asarray(steps)    #time axis
               
        n = self.t.size
            
        if d == 1:
            self.u1 = np.zeros((n,self.size)) #first put zeros into the array of u1
    
            self.u1[0] = self.U0 #initial numbers
    
            # Time loop
            for k in range(n-1):
                self.k = k
                self.u1[k+1] = self.advance1() #put the results of RK4 method into the next u element
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
