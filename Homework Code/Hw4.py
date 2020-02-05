# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:15:44 2020

@author: wany105
"""

import numpy as np 
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import scipy.stats
import math

#Normalize and Unnormalized Model
def Unnormal(Mean, Sig, Data):
    Data = Data*Sig + Mean
    return Data

def Normalize(Mean, Sig, Data):
    Data = (Data - Mean)/Sig
    return Data

##Physics Model
def PhyModel(data):
    E = data[0:4]
    P = data[4]
    
    delta = 0
    for i in range(len(E)):
        delta += 1/1*P/E[i]
    
    return delta

def DataSample(Num, sigma, u):
    data = np.zeros([Num, 6])
    f0 = np.zeros(Num) #prior probability
    
    for i in range(Num):
        E1 = np.random.normal(u[0], sigma[0])
        pE1 = scipy.stats.norm.pdf(E1, u[0], sigma[0])
        
        E2 = np.random.normal(u[1], sigma[1])
        pE2 = scipy.stats.norm.pdf(E2, u[1], sigma[1])
        
        E3 = np.random.normal(u[2], sigma[2])
        pE3 = scipy.stats.norm.pdf(E3, u[2], sigma[2])
        
        E4 = np.random.normal(u[3], sigma[3])
        pE4 = scipy.stats.norm.pdf(E4, u[3], sigma[3])
        
        P = np.random.normal(u[4], sigma[4])
        pP = scipy.stats.norm.pdf(P, u[4], sigma[4])
        
        noi = np.random.normal(u[5], sigma[5])
        pnoi = scipy.stats.norm.pdf(noi, u[5], sigma[5])
        
        data[i, :] = [E1, E2, E3, E4, P, noi]
        f0[i] = pE1*pE2*pE3*pE4*pP*pnoi
    
    return data, f0

def likelihood(data, Num):
    f1 = np.zeros(Num) #likelihood vector
    for i in range(Num):
        obs = PhyModel(data[i]) + data[i, -1]
        pred = PhyModel(data[i])
        sigma = data[i, -1]
        
        f1[i] = scipy.stats.norm.pdf(obs, pred, sigma)
    
    return f1

def count(M, f0, f1, Num):
    Temp = 0
    erfa = np.zeros(Num)
    for i in range(Num):
        erfa[i] = f1[i]/(M*f0[i])
        if(erfa[i] >= 0 and erfa[i] <= 1):
            Temp += 1
            
    print("M = {}, Temp = {}-----------------".format(M, Temp))
    print(erfa)
    
    return 

def accep(M, f0, f1, Num):
    erfa = np.zeros(Num)
    data_up = []
    for i in range(Num):
        erfa[i] = f1[i]/(M*f0[i])
        if(erfa[i] <= np.random.rand()):
            data_up.append(i)
    return data_up

def paracount(E1, E2, E3, E4, P, noi, data_up, data):
    for i in range(len(data_up)):
        E1.append(data[data_up[i], 0])
        E2.append(data[data_up[i], 1])
        E3.append(data[data_up[i], 2])
        E4.append(data[data_up[i], 3])
        P.append(data[data_up[i], 4])
        noi.append(data[data_up[i], 5])
    
    return E1, E2, E3, E4, P, noi


##MCMC
Num = 100
sigma = [3000, 2700, 3200, 2500, 400, 10] #the sigma value for the variables: E1-E4, P, noise
mean = [30000, 25000, 32000, 27000, 2000, 200] #the mean value for the variables: E1-E4, P, noise
E1, E2, E3, E4, P, noi = [], [], [], [], [], []


data, f0 = DataSample(Num, sigma, mean)
f1 = likelihood(data, Num)
count(1e20, f0, f1, Num)
count(1e21, f0, f1, Num)
count(1e19, f0, f1, Num)
count(1e18, f0, f1, Num)
data_up = accep(1e19, f0, f1, Num)
E1, E2, E3, E4, P, noi = paracount(E1, E2, E3, E4, P, noi, data_up, data)

data, f0 = DataSample(Num, sigma, mean)
f1 = likelihood(data, Num)
count(1e20, f0, f1, Num)
count(1e21, f0, f1, Num)
count(1e19, f0, f1, Num)
count(1e18, f0, f1, Num)
data_up = accep(1e19, f0, f1, Num)
E1, E2, E3, E4, P, noi = paracount(E1, E2, E3, E4, P, noi, data_up, data)

def PlotDistri(var, c, X, Y):
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    # Plot Histogram on x
    plt.hist(var, bins = 20, color = c)
    plt.gca().set(title='', xlabel = X, ylabel= Y)
    plt.grid(True)

#plot the distribution of the variable after selection    
PlotDistri(np.array(E1), 'red', X = 'The variable of E1', Y = 'the Frequency')
PlotDistri(np.array(E2), 'blue', X = 'The variable of E2', Y = 'the Frequency')
PlotDistri(np.array(E3), 'green', X = 'The variable of E3', Y = 'the Frequency')
PlotDistri(np.array(E4), 'brown', X = 'The variable of E4', Y = 'the Frequency')

#compare with the distribution of the variable with the prior distribution
color = ['red', 'blue', 'green', 'brown', 'black', 'orange']
X_label = ['the variable of E1', 'the variable of E2', 'the variable of E3', 'the variable of E4', 'P', 'noise']
Name = ['E1', 'E2', 'E3', 'E4', 'P', 'Noise']
for i in range(6):
    fig = plt.figure()
    PlotDistri(np.array(data[:, i]), color[i], X = X_label[i], Y = 'the Frequency')
    plt.savefig('Sampling Distribution {}'.format(Name[i]))









