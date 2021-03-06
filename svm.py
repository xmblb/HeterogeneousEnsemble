from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from common_func import evaluate_method

#read train data
train_data_x = np.load('train_x.npy')
test_data_x = np.load('test_x.npy')

train_data_y = np.append(np.ones(266,dtype=int),np.zeros(266,dtype=int))
# train_y_1D = train_data_y
test_data_y = np.append(np.ones(114,dtype=int),np.zeros(114,dtype=int))
test_y_1D = test_data_y
# test_y_1D = test_data_y
# train_y = np_utils.to_categorical(train_data_y,2)
# test_y = np_utils.to_categorical(test_data_y,2)

# best_parameters = svm_para_opt.svm_opt(train_data_x,train_data_y)

SVM = svm.SVC(probability=True, C=1000, gamma=0.001)
# SVM = GaussianNB()
# SVM = MLPClassifier()
# SVM = LogisticRegression()

# optimal_acu = 0.0
# number = 0
# for i in range(10, 100, 10):
#     SVM = RandomForestClassifier(random_state=1, n_estimators=i)
# # SVM = svm.SVC(probability=True)
#     SVM.fit(train_data_x,train_data_y)
#     y_probability = SVM.predict_proba(test_data_x)  # 得到分类概率值
#     y_probability_first = [x[1] for x in y_probability]
#     test_auc = metrics.roc_auc_score(test_y_1D,y_probability_first)
#     if test_auc>optimal_acu:
#         optimal_acu = test_auc
#         number = i
#
# print(optimal_acu, number)

# SVM = RandomForestClassifier(random_state=1, n_estimators=20)
SVM.fit(train_data_x,train_data_y)
# test_data_x = train_data_x
# test_y_1D = train_data_y

y_pred = SVM.predict(test_data_x)                            #得到输出标签值
accuracy = SVM.score(test_data_x,test_y_1D)                           #得到分类正确率
y_probability = SVM.predict_proba(test_data_x)               #得到分类概率值
y_probability_first = [x[1] for x in y_probability]

test_auc = metrics.roc_auc_score(test_y_1D,y_probability_first)
kappa = evaluate_method.get_kappa(test_y_1D, y_probability_first)
IOA = evaluate_method.get_IOA(test_y_1D, y_probability_first)
MCC = evaluate_method.get_mcc(test_y_1D, y_probability_first)
recall = evaluate_method.get_recall(test_y_1D, y_probability_first)
precision = evaluate_method.get_precision(test_y_1D, y_probability_first)
f1 = evaluate_method.get_f1(test_y_1D, y_probability_first)
# MAPE = evaluate_method.get_MAPE(test_y_1D,y_probability_first)

# evaluate_method.get_ROC(test_y_1D,y_probability_first,save_path='roc_svm_train.txt')

print("ACC = " + str(accuracy))
print("AUC = " + str(test_auc))
print(' kappa = '+ str(kappa))
print("IOA = " + str(IOA))
print("MCC = " + str(MCC))
print(' precision = '+ str(precision))
print("recall = " + str(recall))
print("f1 = " + str(f1))