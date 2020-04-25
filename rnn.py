import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN,Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn import metrics
from keras import optimizers
from common_func import loss_history,evaluate_method
from tensorflow import set_random_seed
set_random_seed(6)
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


model = Sequential()
model.add(SimpleRNN(batch_input_shape=(None, 12, 1), activation='relu',output_dim = 30, unroll=True))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))

optimizer = optimizers.RMSprop()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

history = loss_history.LossHistory()
model.fit(train_x, train_y,  verbose=2, callbacks=[history], shuffle=True,validation_data=(test_x, test_y),epochs=150)


# test_x = train_x
# test_y_1D = train_y_1D
y_prob_test = model.predict(test_x)     #output predict probability
y_probability_first = [prob[1] for prob in y_prob_test]

acc = evaluate_method.get_acc(test_y_1D, y_probability_first)  # AUC value
test_auc = metrics.roc_auc_score(test_y_1D,y_probability_first)
kappa = evaluate_method.get_kappa(test_y_1D, y_probability_first)
IOA = evaluate_method.get_IOA(test_y_1D, y_probability_first)
MCC = evaluate_method.get_mcc(test_y_1D, y_probability_first)
recall = evaluate_method.get_recall(test_y_1D, y_probability_first)
precision = evaluate_method.get_precision(test_y_1D, y_probability_first)
f1 = evaluate_method.get_f1(test_y_1D, y_probability_first)
# MAPE = evaluate_method.get_MAPE(test_y_1D,y_probability_first)

# evaluate_method.get_ROC(test_y_1D,y_probability_first,save_path='roc_stacking.txt')
print("ACC = " + str(acc))
print("AUC = " + str(test_auc))
print(' kappa = '+ str(kappa))
print("IOA = " + str(IOA))
print("MCC = " + str(MCC))
print(' precision = '+ str(precision))
print("recall = " + str(recall))
print("f1 = " + str(f1))

model.save('my_model_RNN1.h5')
# history.loss_plot('epoch')