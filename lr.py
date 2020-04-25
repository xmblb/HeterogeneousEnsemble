from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import numpy as np
from common_func import evaluate_method

#read train data
train_data_x = np.load('train_x.npy')
test_data_x = np.load('test_x.npy')

train_data_y = np.append(np.ones(266,dtype=int),np.zeros(266,dtype=int))
test_data_y = np.append(np.ones(114,dtype=int),np.zeros(114,dtype=int))
test_y_1D = test_data_y

SVM = LogisticRegression(penalty='l2', C=0.1)
# SVM = GradientBoostingClassifier(random_state=6)
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

# evaluate_method.get_ROC(test_y_1D,y_probability_first,save_path='roc_lr_test.txt')

print("ACC = " + str(accuracy))
print("AUC = " + str(test_auc))
print(' kappa = '+ str(kappa))
print("IOA = " + str(IOA))
print("MCC = " + str(MCC))
print(' precision = '+ str(precision))
print("recall = " + str(recall))
print("f1 = " + str(f1))