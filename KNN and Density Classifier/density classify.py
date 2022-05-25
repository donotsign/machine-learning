# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:10:15 2022

@author: DELL
"""
import scipy.io as sio
import sklearn
import numpy as np
import pandas as pd



#读取数据
face_test_x=sio.loadmat('C:\\Users\\DELL\\Desktop\\5310 proj1\\ATNT face\\testX.mat')
face_test_y=sio.loadmat('C:\\Users\\DELL\\Desktop\\5310 proj1\\ATNT face\\testY.mat')
face_train_x=sio.loadmat('C:\\Users\\DELL\\Desktop\\5310 proj1\\ATNT face\\trainX.mat')
face_train_y=sio.loadmat('C:\\Users\\DELL\\Desktop\\5310 proj1\\ATNT face\\trainY.mat')

hand_test_x=sio.loadmat('C:\\Users\\DELL\\Desktop\\5310 proj1\\Binalpha handwritten\\testX.mat')
hand_test_y=sio.loadmat('C:\\Users\\DELL\\Desktop\\5310 proj1\\Binalpha handwritten\\testY.mat')
hand_train_x=sio.loadmat('C:\\Users\\DELL\\Desktop\\5310 proj1\\Binalpha handwritten\\trainX.mat')
hand_train_y=sio.loadmat('C:\\Users\\DELL\\Desktop\\5310 proj1\\Binalpha handwritten\\trainY.mat')
#使用十折交叉验证


hand_test_x_ar=hand_test_x['testX']
hand_test_y_ar=hand_test_y['testY']
hand_train_x_ar=hand_train_x['trainX']
hand_train_y_ar=hand_train_y['trainY']
face_test_x_ar=face_test_x['testX']
face_test_y_ar=face_test_y['testY']
face_train_x_ar=face_train_x['trainX']
face_train_y_ar=face_train_y['trainY']
face_test=np.concatenate((face_test_x_ar,face_test_y_ar),axis=0)
hand_test=np.concatenate((hand_test_x_ar,hand_test_y_ar),axis=0)
face_train=np.concatenate((face_train_x_ar,face_train_y_ar),axis=0)
hand_train=np.concatenate((hand_train_x_ar,hand_train_y_ar),axis=0)



def DensityClassify(newInput, dataSet, labels, r=1000):
    #newInput=face_test_x_ar[:,0]
    #dataSet_xy=face_train
    #dataSet=face_train_x_ar
    #labels=face_train_y_ar
    #r=1200
    #1.step 1：计算出到每个点的距离
    
    numSamples = dataSet.shape[1]   # 算出列数
    dataSet_xy=np.concatenate((dataSet,labels),axis=0)
    newInput=newInput.T
    dataSet=dataSet.T
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # 按元素求差值
    squaredDiff = diff ** 2  # 将差值平方
    squaredDist = np.sum(squaredDiff, axis = 1)   # 按行累加
    distance = squaredDist ** 0.5  # 将差值平方和求开方，即得距离
    data_train_y=dataSet_xy[-1,:]
    distance=pd.DataFrame(distance,columns=['distance'])
    data_train_y=pd.DataFrame(data_train_y,columns=['Class'])
    distance=pd.concat([distance,data_train_y],axis=1)
    minn=min(distance['distance'])
    #print(minn)
    # # step 2: 选取在r范围内的点并计算各个分类的个数
    
    in_range=pd.DataFrame()
    for i in distance.index:
        if distance['distance'][i]<r:
            in_range=in_range.append(pd.DataFrame(distance.loc[i]).T)
            
            
            
    count={}
    distance_total={}
    for i in in_range.index:
        if int(in_range['Class'][i]) not in count.keys():
            count[int(in_range['Class'][i])]=1
            distance_total[int(in_range['Class'][i])]=in_range['distance'][i]
        else:
            count[int(in_range['Class'][i])] = count[int(in_range['Class'][i])]+1
            distance_total[int(in_range['Class'][i])]=distance_total[int(in_range['Class'][i])]+in_range['distance'][i]
    #print('k ',len(count))
    #找到出现最多次的那个类
    #当有几个类出现的次数都为最大值，取他们总距离最短的。
    max_count=max(count.values())
    max_class=[]
    for i in count.keys():
        if count[i]==max_count:
            max_class.append(i)
    
    min_dis=100000000000
    target=0
    for i in max_class:
        if min_dis>distance_total[i]:
            target=i
            min_dis=distance_total[i]
        else:
            pass
    
    return target




def fold10_cross_validation(data,r):
    fold=10
    #data=face_train.T
    
    num_train = data.shape[0]
    num_validation = int(num_train/fold)
    
    index = [i for i in range(num_train)] # test_data为测试数据
    np.random.seed(3)
    np.random.shuffle(index) # 打乱索引
    test_data=data[index]
    corr_rate=0
    
    for i in range(fold):
        validation=test_data[i*num_validation:(i+1)*num_validation]
        test_test=np.delete(test_data,np.s_[i*num_validation:(i+1)*num_validation],axis=0)
        
        
        test_test=test_test.T
        validation=validation.T
        
        true_class=validation[-1]
        true_class=true_class.tolist()
        
        test_test_x=test_test[:-1]
        test_test_y=test_test[-1:]
        new_input=validation[:-1]
        
        estimate=[]
        for i in range(num_validation):
            est=DensityClassify(newInput=new_input[:,i], dataSet=test_test_x, labels=test_test_y, r=r)
            estimate.append(est)
        
        #计算正确率
        corr=0
        for i in range(num_validation):
            if int(estimate[i])==int(true_class[i]):
                corr=corr+1
            else:
                pass
        corr_rate=corr_rate+corr/num_validation
    avg_corr_rate=corr_rate/10
        
    return avg_corr_rate
#------------------------------------------------------------------------------
#if we don't want to miss some point，find the test R
#face=============================================================
for r in range(930,1200,10):
    print('r=',r, '  ',fold10_cross_validation(face_train.T,r=r))
#write============================================================
r1=[10.5,10.6,10.7,10.8,11,11.2,11.4,11.6,11.8,12]
for r in r1:
    print('r=',r,'   ',fold10_cross_validation(data=hand_train.T, r=10.7))
    
#-----------------------------------------------------------------------------
#将验证集与测试集用我们找到的最好的r=930来进行验证获得最终的正确率
a=[]

for i in range(80):
    newInput=face_test_x_ar[:,i]
    dataSet=face_train_x_ar
    labels=face_train_y_ar
    
    qq=DensityClassify(newInput, dataSet, labels, r=930)
    a.append(int(qq))
    #print(qq)
b=face_test_y_ar.tolist()   

true=0
for i in range(80):
    if int(a[i])==int(b[0][i]):
        true=true+1
    else:
        pass
print('the correct rate of face test=',true/face_test_x_ar.shape[1])
#-----------------------------------------------------------------------------    
#将验证集与测试集用我们找到的最好的r=10.5来进行验证获得最终hand的正确率
a=[]

for i in range(234):
    newInput=hand_test_x_ar[:,i]
    dataSet=hand_train_x_ar
    labels=hand_train_y_ar
    
    qq=DensityClassify(newInput, dataSet, labels, r=10.5)
    a.append(int(qq))
    #print(qq)
b=hand_test_y_ar.tolist()   

true=0
for i in range(234):
    if int(a[i])==int(b[0][i]):
        true=true+1
    else:
        pass
print(true/hand_test_x_ar.shape[1])

#-----------------------------------------------------------------------------   
    
    
#忽略圆内无临近点的测试点的分类器
def DensityClassify_2(newInput, dataSet, labels, r=1000):
    #newInput=face_test_x_ar[:,0]
    #dataSet_xy=face_train
    #dataSet=face_train_x_ar
    #labels=face_train_y_ar
    #r=1200
    #1.step 1：计算出到每个点的距离
    
    numSamples = dataSet.shape[1]   # 算出列数
    dataSet_xy=np.concatenate((dataSet,labels),axis=0)
    newInput=newInput.T
    dataSet=dataSet.T
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # 按元素求差值
    squaredDiff = diff ** 2  # 将差值平方
    squaredDist = np.sum(squaredDiff, axis = 1)   # 按行累加
    distance = squaredDist ** 0.5  # 将差值平方和求开方，即得距离
    data_train_y=dataSet_xy[-1,:]
    distance=pd.DataFrame(distance,columns=['distance'])
    data_train_y=pd.DataFrame(data_train_y,columns=['Class'])
    distance=pd.concat([distance,data_train_y],axis=1)
    minn=min(distance['distance'])
    #print(minn)
    # # step 2: 选取在r范围内的点并计算各个分类的个数
    
    in_range=pd.DataFrame()
    for i in distance.index:
        if distance['distance'][i]<r:
            in_range=in_range.append(pd.DataFrame(distance.loc[i]).T)
            
    if in_range.shape[0]==0:
        return 0
    else:        
        count={}
        distance_total={}
        for i in in_range.index:
            if int(in_range['Class'][i]) not in count.keys():
                count[int(in_range['Class'][i])]=1
                distance_total[int(in_range['Class'][i])]=in_range['distance'][i]
            else:
                count[int(in_range['Class'][i])] = count[int(in_range['Class'][i])]+1
                distance_total[int(in_range['Class'][i])]=distance_total[int(in_range['Class'][i])]+in_range['distance'][i]
        #print('k ',len(count))
        #找到出现最多次的那个类
        #当有几个类出现的次数都为最大值，取他们总距离最短的。
        max_count=max(count.values())
        max_class=[]
        for i in count.keys():
            if count[i]==max_count:
                max_class.append(i)
        
        min_dis=100000000000
        target=0
        for i in max_class:
            if min_dis>distance_total[i]:
                target=i
                min_dis=distance_total[i]
            else:
                pass
        
        return target

# 忽略圆内无临近点的验证点
def fold10_cross_validation_2(data,r):
    fold=10
   # data=face_train.T
   # r=700
    
    num_train = data.shape[0]
    num_validation = int(num_train/fold)
    
    index = [i for i in range(num_train)] # test_data为测试数据
    np.random.seed(2022)
    np.random.shuffle(index) # 打乱索引
    test_data=data[index]
    true_rate=0
    
    for i in range(fold):
       # i=0
        validation=test_data[i*num_validation:(i+1)*num_validation]
        test_test=np.delete(test_data,np.s_[i*num_validation:(i+1)*num_validation],axis=0)
        
        
        test_test=test_test.T
        validation=validation.T
        
        true_class=validation[-1]
        true_class=true_class.tolist()
        
        test_test_x=test_test[:-1]
        test_test_y=test_test[-1:]
        new_input=validation[:-1]
        
        estimate=[]
        count_err_point=0
        for i in range(num_validation):
            est=DensityClassify_2(newInput=new_input[:,i], dataSet=test_test_x, labels=test_test_y, r=r)
            if est==0:
                count_err_point=count_err_point+1
            estimate.append(est)
       # print(count_err_point)
        
        #计算误差率
        true=0
        for i in range(num_validation):
            if int(estimate[i])==int(true_class[i]):
                true=true+1
            else:
                pass
        #print('true',true)   
        true_rate=true_rate+(true)/(num_validation)
    avg_err_rate=true_rate/10
        
    return avg_err_rate
#===================================================================================================================
#find best r
#-----------------------------------------------------------------------------
#face
for r in range(500,930,20):
    print('r=',r, '  ',fold10_cross_validation_2(face_train.T,r=r))
#------------------------------------------------------------------------------
#handwriting
r1=[7,7.2,7.4,7.6,7.8,8,8.2,8.4,8.6,8.8,9,9.2,9.4,9.6,9.8,10]
for r in r1:
    print('r=',r, '  ',fold10_cross_validation_2(hand_train.T, r=r))
#====================================================================================================
#-----------------------------------------------------------------------------
#将验证集与测试集用我们找到的最好的r=780来进行验证获得最终的正确率
a=[]

for i in range(80):
    newInput=face_test_x_ar[:,i]
    dataSet=face_train_x_ar
    labels=face_train_y_ar
    
    qq=DensityClassify_2(newInput, dataSet, labels, r=800)
    a.append(int(qq))
    #print(qq)
b=face_test_y_ar.tolist()   

true=0
for i in range(80):
    if int(a[i])==int(b[0][i]):
        true=true+1
    else:
        pass
print('the correct rate of face test=',true/face_test_x_ar.shape[1])
#-----------------------------------------------------------------------------    
#将验证集与测试集用我们找到的最好的r=9来进行验证获得最终hand的正确率
a=[]

for i in range(234):
    newInput=hand_test_x_ar[:,i]
    dataSet=hand_train_x_ar
    labels=hand_train_y_ar
    
    qq=DensityClassify_2(newInput, dataSet, labels, r=9.2)
    a.append(int(qq))
    #print(qq)
b=hand_test_y_ar.tolist()   

true=0
for i in range(234):
    if int(a[i])==int(b[0][i]):
        true=true+1
    else:
        pass
print('the correct rate of handwriting test=',true/hand_test_x_ar.shape[1])

#-----------------------------------------------------------------------------   