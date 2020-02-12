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
    Like = np.zeros([len(pred), len(obs)])
    
    for i in range(len(pred)):
        for j in range(len(obs)):
            Like[i, j] = scipy.stats.norm.pdf(obs[j], pred[i], data[i, -1])
    
    return Like

def like2prob(likelihood):
    prob = np.zeros([likelihood.shape[0], likelihood.shape[1]])
    
    for i in range(likelihood.shape[1]):
        Temp = np.sum(likelihood[:, i])
        temp = 0
        for j in range(likelihood.shape[0]):
            prob[j, i] = likelihood[j, i]/Temp + temp
            temp = prob[j, i]
            
    return prob

def DataSample2(prob, Num, data):
    sdata = np.zeros([Num, 5])
    for i in range(Num):
        erfa = np.random.rand()
        for j in range(5):
            temp = 0
            while(erfa > prob[temp, j]):
                temp += 1
            sdata[i, j] = data[temp, j]
    
    return sdata

def Plotdist(data, c, X_label, Name):#plot the distribution of the variable and calculate the discrete value
    Frequency, interval = plt.hist(data, color = c)[0], plt.hist(data, color = c)[1]
    plt.xlabel(X_label)
    #plt.savefig('Sampling Distribution {}'.format(Name))    
    
    pdf = np.zeros(len(Frequency))
    Sum = np.sum(Frequency)
    for i in range(len(Frequency)):
        pdf[i] = Frequency[i]/Sum
    
    return pdf, interval
                
#uniform distribution
P_low = [20000, 20000, 20000, 15000, 0.03]
P_high = [30000, 30000, 30000, 25000, 0.09]


#Generate Observed value
P = 2000
Num = 5
data= DataSample(P_low, P_high, Num, P)
Obser = Physics(data, P)

#Generate Initial Data
Num = 10000
data= DataSample(P_low, P_high, Num, P)

Temp = 0
while(Temp <= 20):
    Pred = Physics(data, P) 
    Like = likelihood(Obser, Pred, Num, data)
    Sprob = like2prob(Like)
    data = DataSample2(Sprob, Num, data)
    Temp += 1
    print(Temp)







#Plot the KL value to see whether converge
X = np.arange(0, len(klE1), 1)
fig3 = plt.figure()
plt.plot(X, klE1/np.max(klE1), label = 'E1', color = 'red', marker = 'o', markersize=0.3)
plt.plot(X, klE2/np.max(klE2), label = 'E2', color = 'blue', marker = 'o', markersize=0.3)
plt.plot(X, klE3/np.max(klE3), label = 'E3', color = 'orange', marker = 'o', markersize=0.3)
plt.plot(X, klE4/np.max(klE4), label = 'E4', color = 'green', marker = 'o', markersize=0.3)
plt.plot(X, np.abs(klnoi)/np.max(np.abs(klnoi)), label = 'Noi_sig', color = 'brown', marker = 'o', markersize=0.3)
plt.xticks(np.arange(0, len(klE1), 20))
plt.xlabel('The number of samples (cumulative)*1000',  fontweight='bold')
plt.ylabel('KL divergence value (normalized)',  fontweight='bold')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)
plt.grid(True)
#plt.savefig("InterStrength2.png", dpi = 2000, bbox_inches='tight') 
plt.show()

###Generate the samples from the prior distribution so that we can plot and compare
#psE1, psE2, psE3, psE4, psnoi = [], [], [], [], []
#for i in range(len(E1)):
#    psE1.append(np.random.normal(P_mean[0], P_sigma[0]))
#    
#for i in range(len(E2)):
#    psE2.append(np.random.normal(P_mean[1], P_sigma[1]))
#    
#for i in range(len(E3)):
#    psE3.append(np.random.normal(P_mean[2], P_sigma[2]))
#    
#for i in range(len(E4)):
#    psE4.append(np.random.normal(P_mean[3], P_sigma[3]))
#    
#for i in range(len(Noise)):
#    psnoi.append(np.random.normal(P_mean[4], P_sigma[4]))
    
#Generate the samples from the prior distribution_uniform
psE1, psE2, psE3, psE4, psnoi = [], [], [], [], []
for i in range(len(E1)):
    psE1.append(np.random.uniform(P_low[0], P_high[0]))
    
for i in range(len(E2)):
    psE2.append(np.random.uniform(P_low[1], P_high[1]))
    
for i in range(len(E3)):
    psE3.append(np.random.uniform(P_low[2], P_high[2]))
    
for i in range(len(E4)):
    psE4.append(np.random.uniform(P_low[3], P_high[3]))
    
for i in range(len(Noise)):
    psnoi.append(np.random.uniform(P_low[4], P_high[4]))
    
#Plot the Prior Distribution and the final posterior distribution 
PlotDistri2(psE1, E1, 'red', 'blue', 'The value of the variable E1', 'E1_Prior', 'E1', 'the Frequency', 0.4, 0.3)
PlotDistri2(psE2, E2, 'red', 'blue', 'The value of the variable E2', 'E2_Prior', 'E2', 'the Frequency', 0.4, 0.3)
PlotDistri2(psE3, E3, 'red', 'blue', 'The value of the variable E3', 'E3_Prior', 'E3', 'the Frequency', 0.4, 0.3)
PlotDistri2(psE4, E4, 'red', 'blue', 'The value of the variable E4', 'E4_Prior', 'E4', 'the Frequency', 0.4, 0.3)
PlotDistri2(psnoi, Noise, 'red', 'blue', 'The value of the variable noise', 'Noise_Prior', 'Noise', 'the Frequency', 0.4, 0.3)
    