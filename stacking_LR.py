import numpy as np
from common_func import evaluate_method
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from keras.utils import np_utils

#read data
train_x = np.load('train_stacking_x_new.npy')
train_y = np.load('train_stacking_y_new.npy')

test_x = np.load('test_stacking_x_new.npy')
train_y = [x[1] for x in train_y]
train_y = np.array(train_y)
print(train_y.shape,train_x.shape)
# test_x, test_y_1D = read_data.read_data('test_data_sanxia.csv')
# train_x = np.load('train_x_new_90percent.npy')
# test_x = np.load('train_x_new_90percent.npy')
# train_y = np.load('train_y_new_90percent.npy')


# test_y_1D = train_y_1D
test_y_1D = np.append(np.ones(114,dtype=int),np.zeros(114,dtype=int))


# SVM = svm.SVC(probability=True)
SVM = LogisticRegression()
SVM.fit(train_x,train_y)

# test_x = np.load('test_stacking_x_new_train.npy')
# test_y_1D = np.append(np.ones(266,dtype=int),np.zeros(266,dtype=int))

proba_pred = SVM.predict(test_x)     #output predict probability
accuracy = SVM.score(test_x, test_y_1D)
y_probability = SVM.predict_proba(test_x)
y_probability_first = [x[1] for x in y_probability]
# np.savetxt("Predict_pro_3.txt", y_probability_first)

test_auc = metrics.roc_auc_score(test_y_1D,y_probability_first)
kappa = evaluate_method.get_kappa(test_y_1D, y_probability_first)
IOA = evaluate_method.get_IOA(test_y_1D, y_probability_first)
MCC = evaluate_method.get_mcc(test_y_1D, y_probability_first)
recall = evaluate_method.get_recall(test_y_1D, y_probability_first)
precision = evaluate_method.get_precision(test_y_1D, y_probability_first)
f1 = evaluate_method.get_f1(test_y_1D, y_probability_first)
# MAPE = evaluate_method.get_MAPE(test_y_1D,y_probability_first)

evaluate_method.get_ROC(test_y_1D,y_probability_first,save_path='roc_stacking_test.txt')

print("ACC = " + str(accuracy))
print("AUC = " + str(test_auc))
print(' kappa = '+ str(kappa))
print("IOA = " + str(IOA))
print("MCC = " + str(MCC))
print(' precision = '+ str(precision))
print("recall = " + str(recall))
print("f1 = " + str(f1))
# print("MAPE = " + str(MAPE))