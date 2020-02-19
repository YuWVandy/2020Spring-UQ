# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:41:54 2020

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

##----------------------------KL-expansion-Problem 2
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

#---------------------------------------------------------------------Training our RGP model
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import scipy.stats

tdata = np.zeros([X_Data.shape[0], X_Data.shape[1] + 1])

for i in range(len(X_Data)):
    for j in range(X_Data.shape[1] + 1):
        if(j == 0):
            tdata[i, j] = 1
        elif(j == 1):
            tdata[i, j] = Normalize(Mean_P, Sig_P, X_Data[i, j - 1])
        else:
            tdata[i, j] = Normalize(Mean_E, Sig_E, X_Data[i, j - 1])

yphy = Y #Physical Model
ysur = Y2 #Surrogate Model
##Maximum likelihood to estimate the beta

def CalLikelihood(beta, x, sigma, y):
    predict = np.matmul(x, beta)
    prob = scipy.stats.norm.pdf(predict, y, sigma)
    return prob

def objective_fun(parameter):
    beta = np.array(parameter[0:6]).reshape(6, 1)
    sigma = parameter[6]
    
    obj = 0
    
    for i in range(len(tdata)):
        logprob = np.log(CalLikelihood(beta, tdata[i, :], sigma, Y[i]))
        obj -= logprob
    
    print(obj)
    return obj

parameter = [1, 1, 1, 1, 1, 1, 1]
sol = minimize(objective_fun, parameter, tol = 1e-6, method = 'Nelder-Mead')

##-----------------RGP
def relement(theta, sigma, x, y):
    temp = 0
    for i in range(len(theta)):
        temp += theta[i]*(x[i] - y[i])**2
    temp = np.exp(-temp)
    return temp*sigma**2

def rmatrix(X, sigma, theta):
    R = np.zeros([len(X), len(X)])
    for i in range(len(X)):
        for j in range(len(X)):
            R[i, j] = relement(theta, sigma, X[i, :], X[j, :])
            R[j, i] = R[i, j]
    
    return R

def obj(parameter):
    theta = np.array(parameter[0:5]).reshape(5, 1)
    sigma = parameter[5]

    #R = rmatrix(X[:, 1:6], sigma, theta)
    R = rmatrix(tdata[:, 1:6], sigma, theta)
    g = Y.reshape(len(Y), 1)
    Fbeta = np.matmul(tdata, beta)
    
    sigma2 = 1/len(tdata)*np.matmul(np.matmul((g - Fbeta).transpose(), np.linalg.inv(R)), g - Fbeta)
    
    obj = 1/len(tdata)*np.log(np.linalg.det(R)) + np.log(sigma2)
    print(obj)
    
    return obj

def ru(sigma, theta, x, X):
    r = np.zeros(len(X))
    
    for i in range(len(r)):
        r[i] = relement(theta, sigma, x, X[i, :])
    
    return r

def usigu(beta, R, x, X, Y, sigma, r):
    g = Y.reshape(len(Y), 1)
    Fbeta = np.matmul(X, beta)
    hu = x
    
    ##Mean Value
    ug = np.matmul(hu, beta) + np.matmul(np.matmul(r, np.linalg.inv(R)), g - Fbeta)
    ##Sig Value
    temp = np.zeros([len(x)+len(R), len(x)+len(R)])
    temp[0:len(x), 0:len(x)] = 0
    temp[0:len(x), len(x):len(temp)] = X.transpose()
    temp[len(x):len(temp), 0:len(x)] = X
    temp[len(x):len(temp), len(x):len(temp)] = R
    
    huru = np.concatenate((hu, r))
    sigu = np.sqrt(np.abs(sigma**2 - np.matmul(np.matmul(huru, np.linalg.inv(temp)), huru)))

    return ug, sigu

#-------------------------------------------------------Homework 6 
#RGP model training to fix the parameters
parameter = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
sol2 = minimize(obj, parameter, tol = 1e-2, method = 'Nelder-Mead')

Para2 = sol2.x
theta = np.array(Para2[0:5]).reshape(5,1)
sigma = Para2[5]
R = rmatrix(tdata[:, 1:6], sigma, theta)

#Double Loop Sampling
#Totally 5 varaibles, each variable can take 100 values, for each value, there are 1000 samples for the other 4 variables
def SampleNormal(Mean, Sig, Num):
    P = np.zeros(Num)
    for i in range(Num):
        P[i] = np.random.normal(Mean, Sig)
    
    P = (P-Mean)/Sig
    return P

def Sample2PCE(x):
    X = np.zeros(21)
    f1, f2, f3, f4, f5 = x[0], x[1], x[2], x[3], x[4]
    f1f2, f1f3, f1f4, f1f5 = f1*f2, f1*f3, f1*f4, f1*f5
    f2f3, f2f4, f2f5 = f2*f3, f2*f4, f2*f5
    f3f4, f3f5 = f3*f4, f3*f5
    f4f5 = f4*f5
    f1_2, f2_2, f3_2, f4_2, f5_2 = f1**2-1, f2**2-1, f3**2-1, f4**2-1, f5**2-1
    
    X[0], X[1], X[2], X[3], X[4], X[5] = 1, f1, f2, f3, f4, f5
    X[6], X[7], X[8], X[9], X[10] = f1_2, f2_2, f3_2, f4_2, f5_2
    X[11], X[12], X[13], X[14] = f1f2, f1f3, f1f4, f1f5
    X[15], X[16], X[17] = f2f3, f2f4, f2f5
    X[18], X[19] = f3f4, f3f5
    X[20] = f4f5
    
    return X


#Mean = [25000, 22000, 32000, 30000, 2000]
#Sig = [3000, 2700, 3200, 2500, 100]

Up_E1, Up_E2, Up_E3, Up_E4 = 30000, 28000, 27000, 29000
Low_E1, Low_E2, Low_E3, Low_E4 = 25000, 24000, 23000, 20000

Up_P = 2000
Low_P = 400

Up = [Up_P, Up_E1, Up_E2, Up_E3, Up_E4]
Low = [Low_P, Low_E1, Low_E2, Low_E3, Low_E4]

Num1, Num2, Num3, Num4 = 5, 100, 100, 4
SampleMatrix1 = np.zeros([Num1, Num2, Num3, Num4])
RGPYmean = np.zeros([Num1, Num2, Num3])
RGPYsigma = np.zeros([Num1, Num2, Num3])
PCEYmean = np.zeros([Num1, Num2, Num3])
SampleMatrix2 = np.zeros([Num1, Num2])

def SampleUniform2(Up, Low, Num2):
    X = np.zeros(Num2)
    Xnorm = np.zeros(Num2)
    erfa = (Up + Low)/2
    beta = (Up - Low)/2
    for i in range(Num2):
        X[i] = np.random.uniform(Low, Up)
        Xnorm[i] = (X[i] - erfa)/beta
    return Xnorm    
    
    
    
for i in range(Num1):
    SampleMatrix2[i, :] = SampleUniform2(Up[i], Low[i], Num2)
    for j in range(Num2):
        for k in range(Num3):
            Temp = 0
            x = np.zeros(Num1)
            x2 = np.zeros(Num1 + 1)
            x[i] = SampleMatrix2[i, j]
            for l in range(Num4 + 1):
                if(l != i):
                    SampleMatrix1[i, j, k, Temp] = SampleUniform2(Mean[l], Sig[l], 1)
                    x[l] = SampleMatrix1[i, j, k, Temp]
                    Temp += 1
            
            x2[0] = 1
            x2[1:] = x
            r = ru(sigma, theta, x, tdata)
            RGPYmean[i, j, k], RGPYsigma[i, j, k] = usigu(beta, R, x2, tdata, Y, sigma, r)
            
            X = Sample2PCE(x)
            PCEYmean[i, j, k] = np.matmul(X, Coeff)


def SensiVar(mean, Num1, Num2):
    VarSum = np.var(mean)
    Mean = np.zeros([Num1, Num2])
    Var = np.zeros(Num1)
    for i in range(Num1):
        for j in range(Num2):
            Mean[i, j] = np.mean(mean[i, j, :])
        Var[i] = np.var(Mean[i, :])
    return VarSum, Var

RGPv, RGPvi = SensiVar(RGPYmean, Num1, Num2)
PCEv, PCEvi = SensiVar(PCEYmean, Num1, Num2)

#------------------------------------------------------------------------------Analytical sensitivity analysis for PCE
Up_E1, Up_E2, Up_E3, Up_E4 = 30000, 28000, 27000, 29000
Low_E1, Low_E2, Low_E3, Low_E4 = 25000, 24000, 23000, 20000

Up_P = 2000
Low_P = 400
Num2 = 50

def SampleUniform(Up, Low, Num2):
    X = np.zeros(Num2)
    Xnorm = np.zeros(Num2)
    erfa = (Up + Low)/2
    beta = (Up - Low)/2
    for i in range(Num2):
        X[i] = np.random.uniform(Low, Up)
        Xnorm[i] = (X[i] - erfa)/beta
    return X, Xnorm, erfa, beta
    


P, Pnorm, Perfa, Pbeta = SampleUniform(Up_P, Low_P, Num2)
E1, E1norm, E1erfa, E1beta = SampleUniform(Up_E1, Low_E1, Num2)
E2, E2norm, E2erfa, E2beta = SampleUniform(Up_E2, Low_E2, Num2)
E3, E3norm, E3erfa, E3beta = SampleUniform(Up_E3, Low_E3, Num2)
E4, E4norm, E4erfa, E4beta = SampleUniform(Up_E4, Low_E4, Num2)

E = np.stack((E1, E2, E3, E4), axis = 0)
Data = np.stack((P, E1, E2, E3, E4), axis = 0)
Data_norm = np.stack((Pnorm, E1norm, E2norm, E3norm, E4norm), axis = 0)

X_Data, Y = CalElong(E, P, L, Num2)

#Surrogate Model
#PCE Parameter
def TrainData(RegNum, Data_Norm, DataNum):
    X = np.zeros([DataNum, RegNum])
    
    Temp = 0
    for i in range(DataNum):
        f1, f2, f3, f4, f5 = Data_Norm[0, i], Data_Norm[1, i], Data_Norm[2, i], Data_Norm[3, i], Data_Norm[4, i]
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
X = TrainData(RegNum, Data_norm, DataNum)
Coeff = Regression(X, Y)


Index = [[1, 6, 11, 12, 13, 14], [2, 7, 11, 15, 16, 17], [3, 8, 12, 15, 18, 19], [4, 9, 13, 16, 18, 20], [5, 10, 14, 17, 19, 20]]
Sobol = np.zeros(5)
fi = np.square(X)
Sum = np.sum((Coeff[1:]**2)*(np.mean(X[1:]**2, axis = 0)))
#Sum = np.sum((Coeff[1:]**2))
for i in range(5):
    Sobol[i] = np.sum((Coeff[Index[i]]**2)*(np.mean(X[Index[i]]**2, axis = 0)))/Sum









