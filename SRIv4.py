import RungeKutta4
import numpy as np
import matplotlib.pyplot as plt

#1 represents Malmö
#2 represents Copenhagen

def f1(u1, t,beta=4.5/341457.0,alpha=0.1,gamma=0.3):  #ODE definition # malmö population:341457, 13800 moving between cities, 93% of them from Malmö to Copenhagen
    S1, I1, R1 = u1
    
    return [-beta * S1 * I1 - gamma * S1 + 966 * S2 * (S2+I2+R2) - 12834 * S1 * (S1+I1+R1),
            beta * S1 * I1- alpha * I1 + 966 * I2 * (S2+I2+R2) - 12834 * I1 * (S1+I1+R1),
            gamma * S1 + alpha * I1 + 966 * R2 * (S2+I2+R2) - 12834 * R1 * (S1+I1+R1)]

def f2(u2,t,beta=4.5/775033.0,alpha=0.1,gamma=0.3): # copenhagen population:341457
    
    S2, I2, R2 = u2
    return [-beta * S2 * I2 - gamma * S2 - 966 * S2 * (S2+I2+R2) - 12834 * S1 * (S1+I1+R1),
            beta * S2 * I2- alpha * I2 - 966 * I2 * (S2+I2+R2) - 12834 * I1 * (S1+I1+R1),
            gamma * S2 + alpha * I2 - 966 * R2 * (S2+I2+R2) - 12834 * R1 * (S1+I1+R1)]
             

n = 60   # number of days

def solve(j):
    if j == 1:                             #if only one city included
        solution = RungeKutta4.RK4(f1, f2)
        solution.set_initial_condition([341456,1,0],1)
    
        steps = np.linspace(0,60,n) #unit time is 1 day
        u1, t = solution.solve(steps,1)
    
        S1 = u1[:,0]; I1 = u1[:,1]; R1 = u1[:,2]  #put the results into array of S1, I1 and R1
        
        plt.plot(t,S1/340000.0,'b-', label='Susceptible')
        plt.plot(t,I1/340000.0,'r-', label='Infected')
        plt.plot(t,R1/340000.0,'g-', label='Recovered')
   
    if j == 2:                             #if two cities are included
        solution = RungeKutta4.RK4(f1, f2)
        solution.set_initial_condition([[341456,1,0],[775033,1,0]],2)
        
        steps = np.linspace(0,60,n) #unit time is 1 day
        u1, t1, u2, t2 = solution.solve(steps,2)
        
        S1 = u1[:,0]; I1 = u1[:,1]; R1 = u1[:,2]
        S2 = u2[:,0]; I2 = u2[:,1]; R2 = u2[:,2]
        
        plt.figure(1)
        plt.title('SIR model for measles spreading in Malmö')
        plt.xlabel('Days')
        plt.ylabel('Proportion')       
        plt.plot(t1,S1/340000.0,'b-', label='Susceptible')
        plt.plot(t1,I1/340000.0,'r-', label='Infected')
        plt.plot(t1,R1/340000.0,'g-', label='Recovered')
        plt.legend(loc='best')
        
        plt.figure(2)
        plt.title('SIR model for measles spreading in Copenhagen')
        plt.xlabel('Days')
        plt.ylabel('Proportion')        
        plt.plot(t1,S2/775033.0,'b-', label='Susceptible')
        plt.plot(t1,I2/775033.0,'r-', label='Infected')
        plt.plot(t1,R2/775033.0,'g-', label='Recovered')
        plt.legend(loc='best')
    
solve(2)

#error estimation for malmö

def solve2():
    
    
    n = 60
    
    solution = RungeKutta4.RK4(f1, f2)
    solution.set_initial_condition([341456,1,0],1)
       
    steps = np.linspace(0,60,n) #unit time is 1 day
    u1, t = solution.solve(steps,1)
    

    
    S1 = u1[:,0]; I1 = u1[:,1]; R1 = u1[:,2]
    
    S12 = S1[::2]; I12 = I1[::2]; R12 = R1[::2] #extract odd elements in S1 I1 and R1
    
    n = 30 #the time interval h will *2
    
    solution = RungeKutta4.RK4(f1, f2)
    solution.set_initial_condition([341456,1,0],1)
    
    steps = np.linspace(0,60,n) 
    u1, t = solution.solve(steps,1)
    
            
    S11 = u1[:,0]; I11 = u1[:,1]; R11 = u1[:,2]
    S11 = np.asarray(S11)
    S12 = np.asarray(S12)
    I11 = np.asarray(I11)
    I12 = np.asarray(I12)
    R11 = np.asarray(R11)
    R12 = np.asarray(R12)
    
        
    ETS = (S11 - S12) / 15.0    #the truncation error = (y(2h)-y(h))/15, because we are using 4th-order RK method
    ETI = (I11 - I12) / 15.0
    ETR = (R11 - R12) / 15.0
    
    #print(ETS)
    
   
    plt.figure(3)
    plt.title('Truncation error estimation on Malmö')
    plt.xlabel('Days')
    plt.ylabel('Number of people')        
    plt.plot(t,ETS/340000.0,'b-', label='ETS')
    plt.plot(t,ETI/340000.0,'r-', label='ETI')
    plt.plot(t,ETR/340000.0,'g-', label='ETR')
    plt.legend(loc='best')

    
solve2()


plt.show()
