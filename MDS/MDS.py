'''
Create on: 2022.5.11
@author  : ivy
@filr    : MDS.py
@describe: use MDS to draw map
'''

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import pandas as pd


# 读取数据

data=pd.read_excel('citi_distance.xlsx',index_col=0,header=0)
print(data)
type(data)
asarray(data)

# MDS降维算法
def MDS(dataMat, d):
    dataMat=data
    d=2
    dataMatrix = asarray(dataMat)
    dataMatSqua = dataMatrix ** 2

    #计算矩阵B的方法1
    J=eye(10)-1/10*ones(10)
    #print(J)
    j=mat(J)
    q=mat(dataMatSqua)
    B = -0.5*j*q*j
    #print(B)

    # 计算矩阵B的方法2
    distI = mean(dataMatSqua, axis=1)
    distJ = mean(dataMatSqua, axis=0)
    distAll = mean(dataMatSqua)
    #B = zeros(dataMatSqua.shape)
    #for i in range(B.shape[0]):
    #    for j in range(B.shape[1]):
    #        B[i][j] = -0.5 * (dataMatSqua[i][j] - distI[i] - distJ[j] + distAll)

    # 特征值分解
    eigA, eigV = linalg.eig(B)
    X = dot(eigV[:, :d], sqrt(diag(eigA[:d])))
    print(X)
    label = ['beijing', 'shanghai', 'shenzhen', 'chengdu', 'fuzhou', 'haerbin', 'lasa', 'haikou', 'lanzhou', 'kunming']
    #通过调整x0和x1的位置和正负号来进行图像翻转
    plt.plot(X[:, 0], X[:, 1], 'o')
    for i in range(X.shape[0]):
        plt.text(X[i, 0] + 25, X[i, 1] - 15, label[i])
    plt.axis('off')
    plt.show()


    plt.plot(-X[:, 0], -X[:, 1], 'o')
    for i in range(X.shape[0]):
        plt.text(-X[i,0] + 25, -X[i, 1] - 15, label[i],rotation=-35)
    plt.axis('off')
    plt.show()

MDS(data,2)