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

def PlotDistri2(var1, var2, c1, c2, X, label1, label2, Y, erfa1, erfa2): #Function to plot the sampling distribution and the distribution after Metroplis Algorithm
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    # Plot Histogram on x
    plt.hist(var1, bins = 20, color = c1, label = label1, alpha = erfa1)
    plt.hist(var2, bins = 20, color = c2, label = label2, alpha = erfa2)
    plt.gca().set(title='', xlabel = X, ylabel= Y)
    plt.legend()
    plt.grid(True)

##Physics Model
def PhyModel(data):
    E = data[0:4]
    P = data[4]
    
    delta = 0
    for i in range(len(E)):
        delta += 1/1*P/E[i]
    
    return delta

def DataSample(Num, sigma, u): #Randomly generate the sample from the sampling distribution
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

def likelihood(data, Num): #calculate the pdf of the target distribution for each sampling point
    f1 = np.zeros(Num) #likelihood vector
    for i in range(Num):
        obs = PhyModel(data[i]) + data[i, -1]
        pred = PhyModel(data[i])
        sigma = data[i, -1]
        
        f1[i] = scipy.stats.norm.pdf(obs, pred, sigma)
    
    return f1

def count(M, f0, f1, Num): #To visualize the ratio so as to set a optimal alpha
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

def Plotdist(data, c, X_label, Name):#plot the distribution of the variable and calculate the discrete value
    Frequency, interval = plt.hist(data, color = c)[0], plt.hist(data, color = c)[1]
    plt.xlabel(X_label)
    plt.savefig('Sampling Distribution {}'.format(Name))    
    
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
        print(Inter[i])
        Sum += pdfP*np.log(pdfP/pdfQ)
    
    return Sum*(right - left)/div

##MCMC
Num = 100
sigma = [3000, 2700, 3200, 2500, 400, 10] #the sigma value for the variables: E1-E4, P, noise
mean = [30000, 25000, 32000, 27000, 2000, 200] #the mean value for the variables: E1-E4, P, noise
E1, E2, E3, E4, P, noi = [], [], [], [], [], []
klE1, klE2, klE3, klE4, klP, klnoi = [], [], [], [], [], []

color = [['blue', 'maroon'], ['blue', ], 'green', 'brown', 'black', 'orange']
X_label = ['the variable of E1', 'the variable of E2', 'the variable of E3', 'the variable of E4', 'P', 'noise']
Name = ['E1', 'E2', 'E3', 'E4', 'P', 'Noise']

Iter = 15
for i in range(Iter):
    data, f0 = DataSample(Num, sigma, mean)
    f1 = likelihood(data, Num)
    count(1e20, f0, f1, Num)
    count(1e21, f0, f1, Num)
    count(1e19, f0, f1, Num)
    count(1e18, f0, f1, Num)
    data_up = accep(1e19, f0, f1, Num)
    E1, E2, E3, E4, P, noi = paracount(E1, E2, E3, E4, P, noi, data_up, data)

    pdfE1, intervalE1 = Plotdist(E1, color[0], X_label[0], Name[0])
    pdfE2, intervalE2 = Plotdist(E2, color[1], X_label[1], Name[1])
    pdfE3, intervalE3 = Plotdist(E3, color[2], X_label[2], Name[2])
    pdfE4, intervalE4 = Plotdist(E4, color[3], X_label[3], Name[3])
    pdfP, intervalP = Plotdist(P, color[4], X_label[4], Name[4])
    pdfnoi, intervalnoi = Plotdist(noi, color[5], X_label[5], Name[5])

    klE1.append(Dkl(pdfE1, intervalE1, mean[0], sigma[0], 1000))
    klE2.append(Dkl(pdfE2, intervalE2, mean[1], sigma[1], 1000))
    klE3.append(Dkl(pdfE3, intervalE3, mean[2], sigma[2], 1000))
    klE4.append(Dkl(pdfE4, intervalE4, mean[3], sigma[3], 1000))
    klP.append(Dkl(pdfP, intervalP, mean[4], sigma[4], 1000))
    klnoi.append(Dkl(pdfnoi, intervalnoi, mean[5], sigma[5], 1000))
    


#Plot the kl value
X = np.arange(0, len(klE1), 1)
fig3 = plt.figure()
plt.plot(X, klE1/np.max(klE1), label = 'E1', color = 'red', marker = 'o')
plt.plot(X, klE2/np.max(klE2), label = 'E2', color = 'blue', marker = 'o')
plt.plot(X, klE3/np.max(klE3), label = 'E3', color = 'orange', marker = 'o')
plt.plot(X, klE4/np.max(klE4), label = 'E4', color = 'green', marker = 'o')
plt.plot(X, klP/np.max(klP), label = 'P', color = 'black', marker = 'o')
plt.plot(X, klnoi/np.max(klnoi), label = 'Noi_sig', color = 'brown', marker = 'o')
plt.xticks(X)
plt.xlabel('The number of samples (cumulative)*100',  fontweight='bold')
plt.ylabel('KL divergence value (normalized)',  fontweight='bold')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)
plt.grid(True)
#plt.savefig("InterStrength2.png", dpi = 2000, bbox_inches='tight') 
plt.show()


data, f0 = DataSample(len(E1), sigma, mean)
#Plot the Prior Distribution(Sampling Distribution)  
PlotDistri2(data[:, 0], E1, 'red', 'blue', 'The value of the variable E1', 'E1_Prior', 'E1', 'the Frequency', 0.4, 0.3)
PlotDistri2(data[:, 1], E2, 'red', 'blue', 'The value of the variable E2', 'E2_Prior', 'E2', 'the Frequency', 0.4, 0.3)
PlotDistri2(data[:, 2], E3, 'red', 'blue', 'The value of the variable E3', 'E3_Prior', 'E3', 'the Frequency', 0.4, 0.3)
PlotDistri2(data[:, 3], E4, 'red', 'blue', 'The value of the variable E4', 'E4_Prior', 'E4', 'the Frequency', 0.4, 0.3)
PlotDistri2(data[:, 4], P, 'red', 'blue', 'The value of the variable P', 'P_Prior', 'P', 'the Frequency', 0.4, 0.3)
PlotDistri2(data[:, 5], noi, 'red', 'blue', 'The value of the variable noise', 'Noise_Prior', 'Noise', 'the Frequency', 0.4, 0.3)









