# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 13:59:42 2020

@author: wany105
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.integrate import quad
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns


def CorFun(lamb, X1, X2):
    return np.exp(-lamb*np.abs(X1 - X2))

def Klexpan(X, lamb, Num):
    Mtr = np.zeros([len(X), len(X)])
    
    #Generate Correlation Matrix
    for i in range(len(Mtr)):
        for j in range(len(Mtr)):
            Mtr[i, j] = CorFun(lamb, X[i], X[j])
    
    #Calculate the eigenvalue and eigenvector
    eigvalue = np.linalg.eig(Mtr)[0]
    eigvector = np.linalg.eig(Mtr)[1]
    
    #Sampling normal distribtuion
    fi = np.zeros([Num, len(eigvalue)])
    for i in range(Num):
        for j in range(len(eigvalue)):
            fi[i, j] = np.random.normal()
    
    #Sampling Random process
    G = np.zeros([len(X), Num])
    for i in range(len(X)):
        for j in range(Num):
            Temp = 0
            for k in range(len(eigvalue)):
                Temp += eigvalue[k]**0.5*fi[j, k]*eigvector[i, k]
            G[i, j] = Temp
            
    return G

def CorrCheck(G, X):
    Covlist = []
    Inter = []
    Base = G[0, :]
    for i in range(len(X)):
        Compare = G[i, :]
        Covlist.append(pearsonr(Base, Compare)[0])
        Inter.append(X[i] - X[0])
    
    fig1 = plt.figure()
    plt.plot(np.array(Inter), np.array(Covlist), label = 'Simulate', linestyle = '--')
    plt.plot(np.array(Inter), np.exp(-np.array(Inter)), label = 'Target', linestyle = '-')
    plt.xlabel('Interval t')
    plt.ylabel('Correlation Coefficient R(t)')
    plt.xticks(X)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)
    plt.grid(True)

dL = 1
L1 = 0
L2 = 3
L = np.arange(L1, L2 + dL, dL)
lamb = 1
Num = 1000
G = Klexpan(L, lamb, Num)
CorrCheck(G, L)

#Sampling parameter to generate training data
#Unnormalized
def Unnormal(Mean, Sig, Data):
    Data = Data*Sig + Mean
    return Data

def Normalize(Mean, Sig, Data):
    Data = (Data - Mean)/Sig
    return Data

def SampleProcess(X, Num, G, Num2):
    Sample = np.zeros([len(X), Num2])
    for i in range(Num2):
        Index = np.random.randint(0, Num)
        Sample[:, i] = G[:, i]
    return Sample

def SampleNormal(Mean, Sig, Num):
    P = np.zeros(Num)
    for i in range(Num):
        P[i] = np.random.normal(Mean, Sig)
    
    return P

#Calculate Physical Result
def PhysicalModel(E, P):
    Y = 0
    for i in range(len(E)):
        Y += 1/1*P/E[i]
    
    return Y
        
def CalElong(E, P, X, Num2):
    X_Data = np.zeros([Num2, len(E) + 1])
    Y = []
    Temp = 0
    for i in range(len(P)):
        X_Data[Temp, :] = [P[i], E[0, i], E[1, i], E[2, i], E[3, i]]
        Y.append(PhysicalModel([E[0, i], E[1, i], E[2, i], E[3, i]], P[i]))
        Temp += 1
    return X_Data, np.array(Y)


def PlotDistri(Y, c):
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})
    # Plot Histogram on x
    plt.hist(Y, bins = 20, color = c)
    plt.gca().set(title='', xlabel = 'The Elongation Value', ylabel='The Frequency')
    plt.grid(True)
    
    

Mean_E = 30000
Sig_E = 3000

Mean_P = 2000
Sig_P = 400
Num2 = 50
P = SampleNormal(Mean_P, Sig_P, Num2)
E = SampleProcess(L, Num, G, Num2)
E = Unnormal(Mean_E, Sig_E, E)
X_Data, Y = CalElong(E, P, L, Num2)
PlotDistri(Y, c = 'orange')

#Surrogate Model
#PCE Parameter
def TrainData(RegNum, Num2, E, P, DataNum):
    E_Norm = Normalize(Mean_E, Sig_E, E)
    P_Norm = Normalize(Mean_P, Sig_P, P)
    X = np.zeros([DataNum, RegNum])
    
    Temp = 0
    for i in range(Num2):
        f1, f2, f3, f4, f5 = P_Norm[i], E_Norm[0, i], E_Norm[1, i], E_Norm[2, i], E_Norm[3, i]
        f1f2, f1f3, f1f4, f1f5 = f1*f2, f1*f3, f1*f4, f1*f5
        f2f3, f2f4, f2f5 = f2*f3, f2*f4, f2*f5
        f3f4, f3f5 = f3*f4, f3*f5
        f4f5 = f4*f5
        f1_2, f2_2, f3_2, f4_2, f5_2 = f1**2-1, f2**2-1, f3**2-1, f4**2-1, f5**2-1
        
        X[Temp, 0], X[Temp, 1], X[Temp, 2], X[Temp, 3], X[Temp, 4], X[Temp, 5] = 1, f1, f2, f3, f4, f5
        X[Temp, 6], X[Temp, 7], X[Temp, 8], X[Temp, 9], X[Temp, 10] = f1_2, f2_2, f3_2, f4_2, f5_2
        X[Temp, 11], X[Temp, 12], X[Temp, 13], X[Temp, 14] = f1f2, f1f3, f1f4, f1f5
        X[Temp, 15], X[Temp, 16], X[Temp, 17] = f2f3, f2f4, f2f5
        X[Temp, 18], X[Temp, 19] = f3f4, f3f5
        X[Temp, 20] = f4f5
        Temp += 1
    return X
def Regression(X, Y):
    Coeff = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), Y.reshape((len(Y), 1)))
    return Coeff

ParaNum = 5
Order = 2
RegNum = 21
DataNum = len(Y)
X = TrainData(RegNum, Num2, E, P, DataNum)
Coeff = Regression(X, Y)


#Experiment using our surrogate model
Y2 = np.matmul(X, Coeff)
PlotDistri(Y2, c = 'orange')
PlotDistri(Y, c = 'blue')