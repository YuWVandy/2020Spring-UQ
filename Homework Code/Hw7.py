# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:41:54 2020

@author: wany105
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import svd
import pandas as pd
import seaborn as sns

Data = np.array(pd.read_csv(r"C:\Users\wany105\Desktop\UQ_Code\Homework Code\hw7.csv"))
#Standardize the data
mean1 = np.mean(Data, axis = 0)
std1 = np.std(Data, axis = 0)
Data_standard = (Data - mean1)/std1

Train_Num = 450
Verify_Num = 50


Train_Data = Data[0:Train_Num, :]
Mean = np.mean(Train_Data, axis = 0)
std = np.std(Train_Data, axis = 0)

##Correlation matrix
Corr_mat = np.zeros([Data.shape[1], Data.shape[1]])
for i in range(Data.shape[1]):
    for j in range(Data.shape[1]):
        Data1 = Train_Data[i, :]
        Data2 = Train_Data[j, :]
        EXY = np.mean(Data1*Data2)
        EXEY = np.mean(Data1)*np.mean(Data2)
        Corr_mat[i, j] = (EXY - EXEY)/(std[i]*std[j])

eig_value, eig_vector = np.linalg.eig(Corr_mat)

##Select the 1st 5 principal components
print(eig_vector)
eig_vec_prin = eig_vector[:, :5]

##Test the performance of using only 1st 5 components
Y = np.matmul(Data[Train_Num:(Train_Num + Verify_Num)], eig_vec_prin)
T2 = np.matmul(Y, eig_vec_prin[:, :5].transpose())

sns.heatmap((T2 - Data[Train_Num:(Train_Num + Verify_Num)])/Data[Train_Num:(Train_Num + Verify_Num)])
#Variability explaination ability
std_validateset = np.std(Data[Train_Num:(Train_Num + Verify_Num)], axis = 0)
std_T2 = np.std(T2, axis = 0)
var_percent = std_T2/std_validateset

import numpy as np
import matplotlib.pyplot as plt

X = np.arange(0, 81, 1)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, std_validateset, color = 'b', width = 0.25)
ax.bar(X + 0.5, std_T2, color = 'g', width = 0.25)



    



