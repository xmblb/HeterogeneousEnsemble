import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import cross_val_score,KFold



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

penalty = ['l1', 'l2']
C = [0.001, 0.01, 0.1, 1, 10, 100]
kFold = KFold(n_splits=5)

best_score = 0.0

for j,gamma in enumerate(penalty):
    for k,c in enumerate(C):
        model = LogisticRegression(penalty=gamma, C=c)
        score = cross_val_score(model, train_data_x, train_data_y, cv=5, scoring="roc_auc").mean()
        number = j*len(C) + k
        print('loop',number,' ','score:',score,' ','gamma='+str(gamma),' ','c='+str(c))
        if score > best_score:
            best_score = score
            best_parameters = {'penalty':gamma,'C':c}

print("best score: ", best_score)
print("best parameters: ", best_parameters)

SVM = LogisticRegression(penalty=best_parameters['penalty'], C=best_parameters['C'])

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