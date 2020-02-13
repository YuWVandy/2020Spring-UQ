# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:18:08 2020

@author: wany105
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:05:21 2020

@author: wany105
"""

import numpy as np 
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import scipy.stats
import math

##Physics Model
def Physics(data, P):
    pred = np.zeros(len(data))
    
    for i in range(len(data)):
        E = data[i, 0:4]
        delta = 0
    
        for j in range(len(E)):
            delta += 1/1*P/E[j]
        
        pred[i] = delta
    
    return pred


def DataSample(low, high, Num, P):
    data = np.zeros([Num, 5])
    
    for i in range(Num):
        
        E1 = np.random.uniform(low[0], high[0])
    
        E2 = np.random.uniform(low[1], high[1])
    
        E3 = np.random.uniform(low[2], high[2])
    
        E4 = np.random.uniform(low[3], high[3])
    
        noi = np.random.uniform(low[4], high[4])

        data[i, :] = [E1, E2, E3, E4, noi]
        
    return data

def likelihood(obs, pred, Num, data):
    Like = np.zeros(len(pred))
    
    for i in range(len(pred)):
        Like[i] = scipy.stats.norm.pdf(obs, pred[i], data[i, -1])
    
    return Like

def like2prob(likelihood):
    prob = np.zeros(len(likelihood))
    
    temp = 0
    prob = likelihood/np.sum(likelihood)
    for i in range(len(likelihood)):
        prob[i] = temp + prob[i]
        temp = prob[i]
        
    return prob

def DataSample2(prob, Num, data):
    sdata = np.zeros([Num, 5])
    for i in range(Num):
        erfa = np.random.rand()
        temp = 0
        while(erfa > prob[temp]):
            temp += 1
        sdata[i] = data[temp]
    
    return sdata
    

c = ['red', 'blue', 'green', 'orange', 'purple']        
Xlabel = ['E1', 'E2', 'E3', 'E4', 'P']


#uniform distribution
P_low = [25000, 22000, 15000, 16000, 0.015]
P_high = [35000, 28000, 18000, 20000, 0.05]


#Generate Observed value
P = 2000
Num1 = 20
data= DataSample(P_low, P_high, Num1, P)
Obser = Physics(data, P)
    
#Generate Initial Data
Num = 5000
data= DataSample(P_low, P_high, Num, P)
for i in range(5):
    fig = plt.figure()
    plt.hist(data[:, i], color = c[i])
    plt.xlabel(Xlabel[i])
    plt.ylabel('Frequency')

Temp = 0
while(Temp <= Num1 - 1):
    Pred = Physics(data, P) 
    Like = likelihood(Obser[Temp], Pred, Num, data)
    Sprob = like2prob(Like)
    data = DataSample2(Sprob, Num, data)
    Temp += 1

    print(Temp)



for i in range(5):
    fig = plt.figure()
    plt.hist(data[:, i], color = c[i])
    plt.xlabel(Xlabel[i])
    plt.ylabel('Frequency')




    