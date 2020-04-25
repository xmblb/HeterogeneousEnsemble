#!/usr/bin/python
# # -*- coding=utf-8 -*-
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Activation,Dropout,Dense,LSTM,Conv1D,MaxPool1D,Flatten
from common_func import loss_history,evaluate_method,read_data
from keras import optimizers
from sklearn.model_selection import KFold
from tensorflow import set_random_seed
set_random_seed(6)
#read train data
np.random.seed(6)


train_x = np.load('train_x.npy')
test_x = np.load('test_x.npy')
train_x = np.expand_dims(train_x,axis=2)
test_x = np.expand_dims(test_x,axis=2)

train_data_y = np.append(np.ones(266,dtype=int),np.zeros(266,dtype=int))
train_y_1D = train_data_y
test_data_y = np.append(np.ones(114,dtype=int),np.zeros(114,dtype=int))
test_y_1D = test_data_y
train_y = np_utils.to_categorical(train_data_y,2)
test_y = np_utils.to_categorical(test_data_y,2)
kfold = KFold(n_splits=5, shuffle=True, random_state=6)
cvscores = []
for train, test in kfold.split(train_x, train_y_1D):
  # create model
    model = Sequential()
    model.add(SimpleRNN(batch_input_shape=(None, 12, 1), activation='relu', output_dim=30, unroll=True))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    # model.add(BatchNormalization())
    model.add(Activation('softmax'))
    optimizer = optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # Fit the model
    model.fit(train_x[train], np_utils.to_categorical(train_y_1D[train], 2), epochs=200,  verbose=2)
    # evaluate the model
    y_prob_test = model.predict(train_x[test])     #output predict probability
    probability = [prob[1] for prob in y_prob_test]
    auc = evaluate_method.get_auc(train_y_1D[test],probability)    # ACC value
    print("AUC: ", auc)
    cvscores.append(auc)
print((np.mean(cvscores), np.std(cvscores)))




