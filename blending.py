import numpy as np
from sklearn import svm
import gdal
from sklearn.linear_model import LogisticRegression
# from common_func import read_data,evaluate_method
from keras.utils import np_utils
from keras.models import Model, Sequential,load_model
from keras import optimizers
from keras.layers import Conv1D, MaxPool1D, Activation, Average, Dropout,Dense,Flatten,SimpleRNN,LSTM
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow import set_random_seed
set_random_seed(6)
np.random.seed(6)

def read_tif(file):
    dem = gdal.Open(file)
    col = dem.RasterXSize
    row = dem.RasterYSize
    band = dem.RasterCount
    geotransform = dem.GetGeoTransform()
    proj = dem.GetProjection()
    data = np.zeros([row, col, band])
    for i in range(band):
        sing_band = dem.GetRasterBand(i+1)
        data[:,:,i] = sing_band.ReadAsArray()
    return col, row, geotransform, proj, data
#read data
train_x_1D = np.load('train_x.npy')
test_x_1D = np.load('test_x.npy')
train_x = np.expand_dims(train_x_1D,axis=2)
test_x = np.expand_dims(test_x_1D,axis=2)

train_y_1D = np.append(np.ones(266,dtype=int),np.zeros(266,dtype=int))
test_y_1D = np.append(np.ones(114,dtype=int),np.zeros(114,dtype=int))
train_y = np_utils.to_categorical(train_y_1D, 2)
test_y = np_utils.to_categorical(test_y_1D, 2)

# total_data = pd.read_csv('total_data_yanshan12.csv')
# total_data = total_data.values
# total_data_x = total_data[:,:-1]
# total_data_x_DL = np.expand_dims(total_data_x,axis=2)


def get_blending(models, train_x, train_y, test_x):
    train_x_new, test_x_new, train_y_new, test_y_new = train_test_split(train_x,train_y, test_size=0.5, shuffle=True, random_state=1)
    # learnRate = [0.01, 0.01]
    epochs = [150, 150]
    optimizer = ['Adamax','RMSprop']

    result_train_x = np.zeros(shape = (len(models), len(test_x_new)))
    result_test_x = np.zeros(shape = (len(models), len(test_x)))
    j = 0
    for i in range(len(models)):
        # learningR = learnRate[i]
        epoch = epochs[i]
        model = models[i]
        # optimizer = optimizers.Adagrad()
        model.compile(loss='categorical_crossentropy',optimizer=optimizer[i], metrics=['accuracy'])
        model.fit(train_x_new, train_y_new, validation_data=(test_x_new, test_y_new),
                  verbose=2, epochs=epoch)
        y_prob_train = model.predict(test_x_new)
        probability_train = [prob_test[1] for prob_test in y_prob_train]
        result_train_x[j] = probability_train

        y_prob_test = model.predict(test_x)
        probability_test = [prob_test[1] for prob_test in y_prob_test]
        result_test_x[j] = probability_test
        j += 1
    return result_train_x.T, test_y_new,  result_test_x.T

def get_blending_ML(models, train_x, train_y, test_x):
    train_x_new, test_x_new, train_y_new, test_y_new = train_test_split(train_x, train_y, test_size=0.5, shuffle=True, random_state=1)
    result_train_x = np.zeros(shape = (len(models), len(test_x_new)))
    result_test_x = np.zeros(shape = (len(models), len(test_x)))
    j = 0
    for i in range(len(models)):
        model = models[i]
        model.fit(train_x_new,train_y_new)
        y_prob_train = model.predict_proba(test_x_new)
        probability_train = [prob_test[1] for prob_test in y_prob_train]
        result_train_x[j] = probability_train

        y_prob_test = model.predict_proba(test_x)
        probability_test = [prob_test[1] for prob_test in y_prob_test]
        result_test_x[j] = probability_test
        j += 1
    return result_train_x.T, test_y_new, result_test_x.T

def cnn():
    model = Sequential()
    # 添加输入层
    model.add(Conv1D(30, 3, activation='relu', input_shape=(12, 1)))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    return model

def rnn():
    model = Sequential()
    model.add(SimpleRNN(batch_input_shape=(None, 12, 1), activation="relu", output_dim=30, unroll=True))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

cnn_model = cnn()
rnn_model = rnn()
LR = LogisticRegression(penalty='l2', C=0.1)
SVM = svm.SVC(probability=True, C=1000, gamma=0.001)
models = [cnn_model, rnn_model]

col, row, geotransform, proj, total_data = read_tif('yanshan12.tif')
total_data_DL = np.reshape(total_data, [col*row, total_data.shape[2], 1])
total_data = np.reshape(total_data, [col*row, total_data.shape[2]])

# new_train_x_first, new_train_y_first, new_test_x_first = get_blending(models, train_x, train_y, train_x)
# new_train_x_second, new_train_y_second, new_test_x_second = get_blending_ML([LR, SVM], train_x_1D,
#                                                                             train_y_1D,train_x_1D)

new_train_x_first, new_train_y_first, new_total_x_first = get_blending(models, train_x, train_y, total_data_DL)
new_train_x_second, new_train_y_second, new_total_x_second = get_blending_ML([LR, SVM], train_x_1D, train_y_1D,
                                                                             total_data)
# meta_train = np.concatenate((new_train_x_first, new_train_x_second), axis=1)
# meta_test = np.concatenate((new_test_x_first, new_test_x_second), axis=1)

meta_total = np.concatenate((new_total_x_first, new_total_x_second), axis=1)
print(meta_total.shape)

# np.save('train_belending_x_new.npy', meta_train)
# np.save('train_blending_y_new.npy', new_train_y_first)
# np.save('test_blending_x_new_orig_train.npy', meta_test)

np.save('train_blending_total.npy', meta_total)