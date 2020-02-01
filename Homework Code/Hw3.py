# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:00:14 2020

@author: wany105
"""

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

Para = sol.x
beta = np.array(Para[0 : 6]).reshape(6, 1)
sigma = Para[6]

#Compared the linear regression with maximum likelihood
beta2 = Regression(tdata, Y)

ylinear = np.matmul(tdata, beta2)
ymaximum = np.matmul(tdata, beta)

PlotDistri(ymaximum, c = 'gold')


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

    

parameter = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2]
sol2 = minimize(obj, parameter, tol = 1e-2, method = 'Nelder-Mead')

Para2 = sol2.x
theta = np.array(Para2[0:5]).reshape(5,1)
sigma = Para2[5]
R = rmatrix(tdata[:, 1:6], sigma, theta)

ug = np.zeros(len(tdata))
sigu = np.zeros(len(tdata))
for i in range(len(tdata)):
    r = ru(sigma, theta, tdata[i, 1:6], tdata[:, 1:6])
    ug[i], sigu[i] = usigu(beta, R, tdata[i, :], tdata, Y, sigma, r)
    
PlotDistri(ug, c = 'red')    
    
    
    

##Standard package for GPR
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(tdata[:, 1:6], Y)

ugstand = gpr.predict(tdata[:,1:6], return_std = True)[0]
sigustand = gpr.predict(tdata[:,1:6], return_std = True)[1]

PlotDistri(ugstand, c= 'purple')




#--------------------------------------------------------------------Validation
Num2 = 10
pval = SampleNormal(Mean_P, Sig_P, Num2)
Eval = Unnormal(Mean_E, Sig_E, SampleProcess(L, Num, G, Num2))

#Physical Model
xval, yval1 = CalElong(Eval, pval, L, Num2)

#Surrogate Model
xval = TrainData(RegNum, Num2, Eval, pval, Num2)
yval2 = np.matmul(xval, Coeff)

#linear regression 
yval4 = np.matmul(xval2, beta2)

#GPR Model
pvalnorm = Normalize(Mean_P, Sig_P, pval)
evalnorm = Normalize(Mean_E, Sig_E, Eval)
xval = np.zeros([Num2, 5])
xval2 = np.zeros([Num2, 6])
Temp = 0

for i in range(Num2):
        xval[Temp, :] = pvalnorm[i], evalnorm[0, i], evalnorm[1, i], evalnorm[2, i], evalnorm[3, i]
        xval2[Temp, :] = 1, pvalnorm[i], evalnorm[0, i], evalnorm[1, i], evalnorm[2, i], evalnorm[3, i]
        Temp += 1

yval3 = np.zeros(Num2)
sigyval3 = np.zeros(Num2)
for i in range(Num2):
    r = ru(sigma, theta, xval2[i, 1:6], tdata[:, 1:6])
    yval3[i], sigyval3[i] = usigu(beta, R, xval2[i, :], tdata, Y, sigma, r)
    
yvalstand = gpr.predict(xval2[:, 1:6], return_std = True)[0]
sigyvalstand = gpr.predict(xval2[:, 1:6], return_std = True)[1]
    
PlotDistri(yval1, c= 'blue') 
PlotDistri(yval2, c= 'orange') 
PlotDistri(yval4, c= 'green') 
PlotDistri(yval3, c= 'red') 
PlotDistri(yvalstand, c= 'purple')


##Plot
X_Num = np.arange(1, Num2 + 1, 1)
size = 3
plt.scatter(X_Num, yval1, c = 'blue', label = 'Physical(Real) Model', s = size)
plt.scatter(X_Num, yval2, c = 'orange', label = 'PCE Model', s = size)
plt.scatter(X_Num, yval4, c = 'green', label = 'Least Square Model', s = size)
plt.scatter(X_Num, yval3, c = 'red', label = 'Maximum Likelihood + GPR (Manually)', s = size)
plt.scatter(X_Num, yvalstand, c = 'purple', label = 'GPR (Using existed package)', s = size)
plt.xlabel('Data seriel number')
plt.ylabel('Predict value of each model')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)
plt.grid(True)
plt.show()


plt.scatter(X_Num, yval3, c = 'green', label = 'Maximum Likelihood + GPR (Manually)', s = size)
plt.scatter(X_Num, yvalstand, c = 'purple', label = 'GPR (Using existed package)', s = size)
plt.fill_between(X_Num, yval3 - sigyval3, yval3 + sigyval3, color = 'green', alpha = 0.2)
plt.fill_between(X_Num, yvalstand - sigyvalstand, yvalstand + sigyvalstand, color = 'purple', alpha = 0.2)
plt.xlabel('Data seriel number')
plt.ylabel('Predict value of each model')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, frameon = 0)
plt.grid(True)
plt.show()