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

#Normalize and Unnormalized Model
def Unnormal(Mean, Sig, Data):
    Data = Data*Sig + Mean
    return Data

def Normalize(Mean, Sig, Data):
    Data = (Data - Mean)/Sig
    return Data

##Physics Model
def PhyModel(data, P):
    E = data[0:4]
    delta = 0
    
    for i in range(len(E)):
        delta += 1/1*P/E[i]
    
    return delta

def DataSample(sigma, u):
    data = np.zeros(5)
    
    E1 = np.random.normal(u[0], sigma[0])
    pE1 = scipy.stats.norm.pdf(E1, u[0], sigma[0])
    
    E2 = np.random.normal(u[1], sigma[1])
    pE2 = scipy.stats.norm.pdf(E2, u[1], sigma[1])
    
    E3 = np.random.normal(u[2], sigma[2])
    pE3 = scipy.stats.norm.pdf(E3, u[2], sigma[2])
    
    E4 = np.random.normal(u[3], sigma[3])
    pE4 = scipy.stats.norm.pdf(E4, u[3], sigma[3])
    
    noi = np.random.normal(u[4], sigma[4])
    pnoi = scipy.stats.norm.pdf(noi, u[4], sigma[4])

    data = [E1, E2, E3, E4, noi]
    sampleprob = [pE1, pE2, pE3, pE4, pnoi]
    
    return data, sampleprob

def likelihood(data, obs, P):
    pred = PhyModel(data, P)
    sigma = data[-1]

    f1 = scipy.stats.norm.pdf(obs[0], pred, sigma)

    f2 = scipy.stats.norm.pdf(obs[1], pred, sigma)
    
    f3 = scipy.stats.norm.pdf(obs[2], pred, sigma)
    
    f4 = scipy.stats.norm.pdf(obs[3], pred, sigma)
    
    f5 = scipy.stats.norm.pdf(obs[4], pred, sigma)
    
    likeprob = [f1, f2, f3, f4, f5]
    
    return likeprob

#def Prior_nor(sigma, u, data):
#    pE1 = scipy.stats.norm.pdf(data[0], u[0], sigma[0])
#    
#    pE2 = scipy.stats.norm.pdf(data[1], u[1], sigma[1])
#    
#    pE3 = scipy.stats.norm.pdf(data[2], u[2], sigma[2])
#    
#    pE4 = scipy.stats.norm.pdf(data[3], u[3], sigma[3])
#    
#    pnoi = scipy.stats.norm.pdf(data[4], u[4], sigma[4])
#
#    priorprob = [pE1, pE2, pE3, pE4, pnoi]
#    
#    return priorprob

def Prior_uni(P_low, P_high, data):
    pE1 = scipy.stats.uniform.pdf(data[0], P_low[0], P_high[0] - P_low[0])
    
    pE2 = scipy.stats.uniform.pdf(data[1], P_low[1], P_high[1] - P_low[1])
    
    pE3 = scipy.stats.uniform.pdf(data[2], P_low[2], P_high[2] - P_low[2])
    
    pE4 = scipy.stats.uniform.pdf(data[3], P_low[3], P_high[3] - P_low[3])
    
    pnoi = scipy.stats.uniform.pdf(data[4], P_low[4], P_high[4] - P_low[4])

    priorprob = [pE1, pE2, pE3, pE4, pnoi]
    
    return priorprob

def accep(plike, pprior, psample, M):
    alpha = np.zeros(len(plike))
    flag = np.zeros(len(alpha))
    for i in range(len(plike)):
        alpha[i] = plike[i]*pprior[i]/psample[i]*M
        temp = np.random.rand()
        if(temp <= alpha[i]):
            flag[i] = 1
        else:
            flag[i] = 0
            
    return flag

def Plotdist(data, c, X_label, Name):#plot the distribution of the variable and calculate the discrete value
    Frequency, interval = plt.hist(data, color = c)[0], plt.hist(data, color = c)[1]
    plt.xlabel(X_label)
    #plt.savefig('Sampling Distribution {}'.format(Name))    
    
    pdf = np.zeros(len(Frequency))
    Sum = np.sum(Frequency)
    for i in range(len(Frequency)):
        pdf[i] = Frequency[i]/Sum
    
    return pdf, interval

def Dkl(Ppdf, Pinterval, meanQ, sigmaQ, div): #Numerical integration to calculate the KL divergence value
    left = Pinterval[0]
    right = Pinterval[-1]
    
    Inter = np.arange(left, right, (right-left)/div)
    Sum = 0
    for i in range(len(Inter)):
        for j in range(len(Pinterval) - 1):
            if(Inter[i] >= Pinterval[j] and Inter[i] <= Pinterval[j + 1]):
                temp1 = j
    
        pdfP = Ppdf[temp1]
        pdfQ = scipy.stats.norm.pdf(Inter[i], meanQ, sigmaQ)
        Sum += pdfP*np.log(pdfP/pdfQ)
    
    return Sum*(right - left)/div


        
color = ['red', 'blue', 'green', 'brown', 'black', 'orange']
X_label = ['the variable of E1', 'the variable of E2', 'the variable of E3', 'the variable of E4', 'P', 'noise']
Name = ['E1', 'E2', 'E3', 'E4', 'P', 'Noise']

#1 normalized, loglikelihood
#Sampling Distribution - normal distribution
S_sigma = [3000, 2700, 3200, 2500, 0.02]
S_mean = [30000, 25000, 32000, 27000, 0.05]

##Prior Distribution - normal distribution
#P_sigma = [1000, 2000, 3000, 2000, 0.02]
#P_mean = [25000, 20000, 28000, 22000, 0.07]

#uniform distribution
P_low = [20000, 20000, 20000, 15000, 0.03]
P_high = [30000, 30000, 30000, 25000, 0.09]

E1, E2, E3, E4, Noise = [], [], [], [], []
klE1, klE2, klE3, klE4, klnoi = [], [], [], [], []

#Generate Observed value
P = 2000
obs = np.zeros(5)
for i in range(5):
    data, Sprob = DataSample(S_sigma, S_mean)
    obs[i] = PhyModel(data, P)

##Sampling
Temp = 0
while(Temp <= 200):
    temp = 0
    while(temp <= 1000):
        data, Sprob = DataSample(S_sigma, S_mean)
        Lprob = likelihood(data, obs, P)
#        Pprob = Prior_nor(P_sigma, P_mean, data)
        Pprob = Prior_uni(P_low, P_high, data)
        flag = accep(Lprob, Pprob, Sprob, 0.1)
        if(flag[0] == 1):
            E1.append(data[0])
            S_mean[0] = data[0]
        if(flag[1] == 1):
            E2.append(data[1])
            S_mean[1] = data[1]
        if(flag[2] == 1):
            E3.append(data[2])
            S_mean[2] = data[2]
        if(flag[3] == 1):
            E4.append(data[3])
            S_mean[3] = data[3]
        if(flag[4] == 1):
            Noise.append(data[4])
            S_mean[4] = data[4]
        
        temp += 1
    
    pdfE1, intervalE1 = Plotdist(E1, color[0], X_label[0], Name[0])
    pdfE2, intervalE2 = Plotdist(E2, color[1], X_label[1], Name[1])
    pdfE3, intervalE3 = Plotdist(E3, color[2], X_label[2], Name[2])
    pdfE4, intervalE4 = Plotdist(E4, color[3], X_label[3], Name[3])
    pdfnoi, intervalnoi = Plotdist(Noise, color[4], X_label[4], Name[4])
    
    klE1.append(Dkl(pdfE1, intervalE1, P_mean[0], P_sigma[0], 1000))
    klE2.append(Dkl(pdfE2, intervalE2, P_mean[1], P_sigma[1], 1000))
    klE3.append(Dkl(pdfE3, intervalE3, P_mean[2], P_sigma[2], 1000))
    klE4.append(Dkl(pdfE4, intervalE4, P_mean[3], P_sigma[3], 1000))
    klnoi.append(Dkl(pdfnoi, intervalnoi, P_mean[4], P_sigma[4], 1000))

    Temp += 1

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
    