'''
Create on :2022.4.28
@author   :ivy
@file     :Laplacian_Embedding
@describe :try to use laplacian embedding
'''
# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#算出距离平方矩阵
def cal_pairwise_dist(x):
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist

#算出权重
def rbf(dist, t=1.0):
    return np.exp(-(dist / t))


#权重矩阵W
def cal_rbf_dist(data, n_neighbors=10, t=1):
    dist = cal_pairwise_dist(data)
    dist[dist < 0] = 0
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)
    #print(rbf_dist)

    W = np.zeros((n, n))
    for i in range(n):\
        #找出最近的k个点给这些距离赋予权重，其他的是0
        index_ = np.argsort(dist[i])[1:1 + n_neighbors]
        W[i, index_] = rbf_dist[i, index_]
        W[index_, i] = rbf_dist[index_, i]

    return W


def le(data,n_dims=2,n_neighbors=5, t=1.0):
    '''
    :param data: (n_samples, n_features)
    :param n_dims: target dim
    :param n_neighbors: k nearest neighbors
    :param t: a param for rbf
    :return:
    '''
    N = data.shape[0]
    W = cal_rbf_dist(data, n_neighbors, t)
    D = np.zeros_like(W)#形状和W一致但是元素全为0
    for i in range(N):
        D[i, i] = np.sum(W[i])
    #矩阵求逆
    D_inv = np.linalg.inv(D)
    #拉普拉斯矩阵
    L = D - W
    eig_val, eig_vec = np.linalg.eig(np.dot(D_inv, L))
    #eig_val, eig_vec = np.linalg.eig( L)
    sort_index_ = np.argsort(eig_val)
    eig_val = eig_val[sort_index_]
    print("eig_val[:10]: ", eig_val[:10])

    j = 0
    while eig_val[j] < 1e-6:
        j += 1

    print("j: ", j)

    sort_index_ = sort_index_[j:j + n_dims]
    eig_val_picked = eig_val[j:j + n_dims]
    print(eig_val_picked)
    eig_vec_picked = eig_vec[:, sort_index_]

    print("YDY: ")
    print(np.dot(np.dot(eig_vec_picked.T, D), eig_vec_picked))

    X_ndim = eig_vec_picked
    return X_ndim


if __name__ == '__main__':
    X = datasets.load_iris().data
    y = datasets.load_iris().target


    dist = cal_pairwise_dist(X)
    max_dist = np.max(dist)
    print("max_dist", max_dist)
    k=[1,2,3,5,10,15,20,25]
    fig=plt.figure()
    q=1
    for i in k[0:4]:
        plt.subplot(2, 2, q)
        q=q+1
        X_ndim = le(X, n_neighbors=i, t=max_dist * 0.1)
        plt.scatter(X_ndim[:, 0], X_ndim[:, 1], c=y)
        plt.title('k={}'.format(i))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    plt.show()

    fig2 = plt.figure()
    q = 1
    for i in k[4:]:
        plt.subplot(2, 2, q)
        q = q + 1
        X_ndim = le(X, n_neighbors=i, t=max_dist * 0.1)
        plt.scatter(X_ndim[:, 0], X_ndim[:, 1], c=y)
        plt.title('k={}'.format(i))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

    
