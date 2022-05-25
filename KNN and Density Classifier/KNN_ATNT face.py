import scipy.io as scio
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

# # 1 import images
dataFile1 = 'D:/! ZhouHuiz/CUHK-FE/5310 MLearning/proj1/ATNT face/trainX.mat'
data1 = scio.loadmat(dataFile1)
trainX = data1['trainX'].transpose()

dataFile2 = 'D:/! ZhouHuiz/CUHK-FE/5310 MLearning/proj1/ATNT face/trainY.mat'
data2 = scio.loadmat(dataFile2)
trainY = data2['trainY'][0,:]

dataFile3 = 'D:/! ZhouHuiz/CUHK-FE/5310 MLearning/proj1/ATNT face/testX.mat'
data3 = scio.loadmat(dataFile3)
testX = data3['testX'].transpose()

dataFile4 = 'D:/! ZhouHuiz/CUHK-FE/5310 MLearning/proj1/ATNT face/testY.mat'
data4 = scio.loadmat(dataFile4)
testY = data4['testY'][0,:]


# # 2 define KNN classifier
def kNNClassify(newInput, dataSet, labels, k):  # dataSet:trainX  labels:trainY  newInput:testX
    numSamples = dataSet.shape[0]

    # step 1: 计算距离
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # 按元素求差值
    squaredDiff = diff ** 2  # 将差值平方
    squaredDist = np.sum(squaredDiff, axis = 1)   # 按行累加
    distance = squaredDist ** 0.5  # 将差值平方和求开方，即得距离

    # step 2: 对距离排序
    sortedDistIndices = np.argsort(distance)
    classCount = {}
    for i in range(k):
        # step 3: 选择k个最近邻
        voteLabel = labels[sortedDistIndices[i]]

        # step 4: 计算k个最近邻中各类别出现的次数
        # when the key voteLabel is not in dictionary classCount, get()
        # will return 0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # step 5: 返回出现次数最多的类别标签
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key

    return maxIndex


# # 3 Cross-validation
# define k to be tested
ks = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

kf = KFold(n_splits=10, shuffle=True)  # 10 folds, random grouping
best_k = ks[0]
best_score = 0  # accuracy of prediction

# choose k with best average accuracy
for k in ks:
    curr_score = 0
    for train_index, valid_index in kf.split(trainX):
        train_X = trainX[train_index]
        valid_X = trainX[valid_index]
        train_Y = trainY[train_index]
        valid_Y = trainY[valid_index]

        # train and predict for a fold
        predict_label = []
        for i in range(valid_X.shape[0]):
            outputLabel = kNNClassify(valid_X[i, :], train_X, train_Y, k)
            predict_label.append(outputLabel)
        curr = np.array(predict_label) == valid_Y
        curr_score = curr_score + np.sum(curr) / curr.shape
    avg_score = curr_score / 10
    print("k:", k, "avg_score:", avg_score)
    if avg_score > best_score:
        best_k = k
        best_score = avg_score

print("------------------------")

print("after cross validation, the final best k is: %d" % best_k)
print("after cross validation, the final best average score is: ", best_score)

# # 4 predict for test images
predict_label = []
for i in range(testX.shape[0]):
    outputLabel = kNNClassify(testX[i, :], trainX, trainY, best_k)
    predict_label.append(outputLabel)

ccur = np.array(predict_label) == testY
print("predicted class labels:", predict_label)
print("Accuracy of prediction:", np.sum(ccur)/ccur.shape)



