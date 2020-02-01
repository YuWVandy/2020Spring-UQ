# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 13:58:19 2020

@author: wany105
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.integrate import quad
import math
import numpy as np
from matplotlib import pyplot as plt

#----------------------------------Spectral Simulation
def S_gg(w):
    return 1/(math.pi*(1+w**2))

def Gt(W, t, dW, fi):
    gt = 0
    for i in range(len(W)):
        #fi = 2*math.pi*np.random.rand()
        Sgg = S_gg(W[i])
        Temp = (2*Sgg*dW)**0.5*math.cos(W[i]*t+fi[i])
        gt += Temp
    gt = gt*(2**0.5)
    return gt

w = 20*math.pi
N = 1000
dW = w/N
Num = 300
dT = 0.01
T1 = 1
W = np.arange(0, w+dW, dW)
T = np.arange(0, T1 + dT, dT)
fi = np.zeros([Num, len(W)])
for i in range(Num):
    for j in range(len(W)):
        fi[i, j] = 2*math.pi*np.random.rand()

G = np.zeros([len(T), Num])
for i in range(len(T)):
    for j in range(Num):
        G[i, j] = Gt(W, T[i], dW, fi[j, :])
CovList1 = []
TimeInter = []    
Base = G[0, :]
for i in range(len(T)):
    Comp = G[i, :]
    CovList1.append(pearsonr(Base, Comp)[0])
    TimeInter.append(T[i] - T[0])

fig1 = plt.figure()
plt.plot(np.array(TimeInter), np.array(CovList1), label = 'Simulate', linestyle = '--')
plt.plot(np.array(TimeInter), np.exp(-np.array(TimeInter)), label = 'Target', linestyle = '-')
plt.xlabel('Time Interval t(s)')
plt.ylabel('Correlation Coefficient R(t)')
plt.xticks(np.arange(0, T1+0.1, 0.1))
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)
plt.grid(True)


##----------------------------KL-expansion
dT = 0.01
T = np.arange(0, 1 + dT, dT)
Mtr = np.zeros([len(T), len(T)])
for i in range(len(T)):
    for j in range(len(T)):
        Mtr[i, j] = np.exp(-np.abs(T[i] - T[j]))
eigvalue = np.linalg.eig(Mtr)[0]
eigvector = np.linalg.eig(Mtr)[1]

Num = 100
lamb = np.zeros([Num, len(eigvalue)])
for i in range(Num):
    for j in range(len(eigvalue)):
        lamb[i, j] = np.random.normal()
        
G = np.zeros([len(T), Num])
for i in range(len(T)):
    for j in range(Num):
        Temp = 0
        for k in range(len(eigvalue)):
            Temp += eigvalue[k]**0.5*lamb[j, k]*eigvector[i, k]
        G[i, j] = Temp

CovList2 = []
TimeInter = []    
Base = G[0, :]
for i in range(len(T)):
    Comp = G[i, :]
    CovList2.append(pearsonr(Base, Comp)[0])
    TimeInter.append(T[i] - T[0])
    
fig2 = plt.figure()
plt.plot(np.array(TimeInter), np.array(CovList2), label = 'Simulate', linestyle = '--')
plt.plot(np.array(TimeInter), np.exp(-np.array(TimeInter)), label = 'Target', linestyle = '-')
plt.xlabel('Time Interval t(s)')
plt.ylabel('Correlation Coefficient R(t)')
plt.xticks(np.arange(0, 1.1, 0.1))
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)
plt.grid(True)