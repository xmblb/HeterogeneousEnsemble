import numpy as np
from keras.models import Sequential, load_model
from sklearn import svm
from keras.layers import Dense,Conv3D,MaxPool3D,Flatten,Activation,Dropout
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from common_func import evaluate_method
from sklearn import metrics


train_x = np.load('train_x.npy')
train_data_x = train_x
test_x = np.load('test_x.npy')
test_data_x = test_x
train_x = np.expand_dims(train_x,axis=2)
test_x = np.expand_dims(test_x,axis=2)

train_data_y = np.append(np.ones(266,dtype=int),np.zeros(266,dtype=int))
train_y_1D = train_data_y
test_data_y = np.append(np.ones(114,dtype=int),np.zeros(114,dtype=int))
test_y_1D = test_data_y
train_y = np_utils.to_categorical(train_data_y,2)
test_y = np_utils.to_categorical(test_data_y,2)

LR = LogisticRegression(penalty='l2', C=0.1)
LR.fit(train_data_x,train_y_1D)
SVM = svm.SVC(probability=True, C=1000, gamma=0.001)
SVM.fit(train_data_x,train_y_1D)
model_cnn = load_model('my_model_CNN.h5')
model_rnn = load_model('my_model_RNN.h5')

# test_data_x = train_data_x
# test_x = train_x
# test_y_1D = train_data_y

y_pred_cnn = model_cnn.predict(test_x)
probability_cnn = [prob[1] for prob in y_pred_cnn]
probability_cnn = np.array(probability_cnn)

y_pred_rnn = model_rnn.predict(test_x)
probability_rnn = [prob[1] for prob in y_pred_rnn]
probability_rnn = np.array(probability_rnn)

y_pred_LR = LR.predict_proba(test_data_x)
y_probability_LR = np.array([x[1] for x in y_pred_LR])

y_pred_SVM = SVM.predict_proba(test_data_x)
y_probability_SVM = np.array([x[1] for x in y_pred_SVM])

y_probability_first = (probability_cnn+probability_rnn+y_probability_LR+y_probability_SVM)/4

acc = evaluate_method.get_acc(test_y_1D, y_probability_first)  # AUC value
test_auc = metrics.roc_auc_score(test_y_1D,y_probability_first)
kappa = evaluate_method.get_kappa(test_y_1D, y_probability_first)
IOA = evaluate_method.get_IOA(test_y_1D, y_probability_first)
MCC = evaluate_method.get_mcc(test_y_1D, y_probability_first)
recall = evaluate_method.get_recall(test_y_1D, y_probability_first)
precision = evaluate_method.get_precision(test_y_1D, y_probability_first)
f1 = evaluate_method.get_f1(test_y_1D, y_probability_first)
# MAPE = evaluate_method.get_MAPE(test_y_1D,y_probability_first)

# evaluate_method.get_ROC(test_y_1D,y_probability_first,save_path='roc_Saveraging_test.txt')

print("ACC = " + str(acc))
print("AUC = " + str(test_auc))
print(' kappa = '+ str(kappa))
print("IOA = " + str(IOA))
print("MCC = " + str(MCC))
print(' precision = '+ str(precision))
print("recall = " + str(recall))
print("f1 = " + str(f1))