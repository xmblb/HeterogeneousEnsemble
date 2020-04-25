import numpy as np
import csv
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import cross_val_score,KFold

MIN_NUM = 1
MAX_NUM = [6,9,5,6,12,5,3,3,5,4,4,7,7,5,5,5]

def mean_data(data):
    data = data.tolist()
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = (data[i][j]-MIN_NUM)/(MAX_NUM[j]-MIN_NUM)
    return data


def readData(train_file_path,test_file_path,num_factors):
    def transform_x(data):
        data_x = []
        for row in data:
            data_i = []
            for i in row:
                data_i.append(int(i))
            data_x.append(data_i)
        return data_x

    def transform_y(data):
        data_y = []
        for i in data:
            data_y.append(int(i))
        return data_y

    data1 = csv.reader(open(train_file_path,'r'))
    train_data_x = []
    train_data_y = []
    for row in data1:
        train_data_x.append(row[:num_factors])
        train_data_y.append(row[num_factors])
    train_data_x = train_data_x[1:]
    train_data_y = train_data_y[1:]
    train_data_x = np.array(transform_x(train_data_x))
    train_data_y = np.array(transform_y(train_data_y))


    data2 = csv.reader(open(test_file_path,'r'))
    test_data_x = []
    test_data_y = []
    for row in data2:
        test_data_x.append(row[:num_factors])
        test_data_y.append(row[num_factors])
    test_data_x = test_data_x[1:]
    test_data_y = test_data_y[1:]
    test_data_x = np.array(transform_x(test_data_x))
    test_data_y = np.array(transform_y(test_data_y))
    return train_data_x,train_data_y,test_data_x,test_data_y

train_data_x = np.load('train_x.npy')
test_data_x = np.load('test_x.npy')

train_data_y = np.append(np.ones(266,dtype=int),np.zeros(266,dtype=int))
# train_y_1D = train_data_y
test_data_y = np.append(np.ones(114,dtype=int),np.zeros(114,dtype=int))
test_y_1D = test_data_y


# train_data_x = mean_data(train_data_x)
# train_data_x = np.array(train_data_x)

data_input_x = test_data_x
# print(data_input_x,type(data_input_x))
# data_input_x = mean_data(data_input_x)
data_input_x = np.array(data_input_x)
# print(data_input_x,type(data_input_x))
data_input_y = test_data_y
# print(data_input_y,type(data_input_y))

gamma_set = [10, 1, 0.1, 0.01, 0.001, 0.0001]
# gamma_set = []
# gamma_min = 2**-10
# for i in range(16):
#     gamma_set.append(gamma_min)
#     gamma_min *= 2

c_set = [0.1, 1, 10, 100, 1000]
# c_set = []
# c_min = 2**-5
# for c in range(16):
#     c_set.append(c_min)
#     c_min *= 2
# kernel = ['linear', 'poly', 'rbf', 'sigmoid']
kFold = KFold(n_splits=5)

best_score = 0.0

for j,gamma in enumerate(gamma_set):
    for k,c in enumerate(c_set):
        model = svm.SVC (kernel='rbf',C=c, gamma=gamma)
        score = cross_val_score(model, train_data_x, train_data_y, cv=5, scoring="roc_auc").mean()
        number = j*len(c_set) + k
        print('loop',number,' ','score:',score,' ','gamma='+str(gamma),' ','c='+str(c))
        if score > best_score:
            best_score = score
            best_parameters = {'c':c,'gamma':gamma}

print("best score: ", best_score)
print("best parameters: ", best_parameters)

SVM = svm.SVC(C=best_parameters['c'],gamma = best_parameters['gamma'],probability=True)

SVM.fit(train_data_x,train_data_y)
data_input_x = test_data_x
data_input_y = test_data_y
y_pred = SVM.predict(data_input_x)                            #得到输出标签值
accuracy = SVM.score(data_input_x,data_input_y)                           #得到分类正确率
y_probability = SVM.predict_proba(data_input_x)               #得到分类概率值
y_probability_first = [x[1] for x in y_probability]
print(y_probability_first,type(y_probability_first))
test_auc = metrics.roc_auc_score(data_input_y,y_probability_first)    #得到AUC值

fpr, tpr, thresholds = metrics.roc_curve(data_input_y, y_probability_first)
print (y_probability_first)
print ('accuracy = %f' %accuracy)
print ('AUC = %f'%test_auc)
#输出混淆矩阵
print (confusion_matrix(data_input_y,y_pred))