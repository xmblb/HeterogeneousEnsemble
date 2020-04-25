import numpy as np
from sklearn import svm
from common_func import read_data,evaluate_method,Stacking
from keras.utils import np_utils
from keras.models import Model, Sequential,load_model
from keras import optimizers
from keras.layers import Conv1D, MaxPool1D, Activation, Average, Dropout,Dense,Flatten,SimpleRNN,LSTM
from sklearn.model_selection import KFold
import gdal
from sklearn.linear_model import LogisticRegression
from tensorflow import set_random_seed
set_random_seed(6)
#read data
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
# total_data_x = np.expand_dims(total_data_x,axis=2)

def get_stacking(models, train_x, train_y, test_x, n_folds):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    new_train_y = np.zeros((1,2))
    for train_index, val_index in kf.split(train_x):
        y_test = train_y[val_index]
        new_train_y = np.append(new_train_y, y_test, axis=0)
    new_train_y = new_train_y[1:]

    new_train_x = np.zeros(shape = (len(models), len(train_x)))
    new_test_x = np.zeros(shape = (len(models), len(test_x)))
    epochs = [150, 150]
    optimizer = ['Adamax','RMSprop']
    for i in range(len(models)):
        # learningR = learnRate[i]
        epoch = epochs[i]
        model = models[i]
        new_single_features = np.array([])
        new_single_test_features = np.zeros(shape=(n_folds, len(test_x)))
        j = 0
        for train_index, val_index in kf.split(train_x):

            x_train, y_train = train_x[train_index], train_y[train_index]
            x_val, y_val = train_x[val_index], train_y[val_index]
            # optimizer = optimizers.Adagrad(lr=learningR)
            model.compile(loss='categorical_crossentropy',
                        optimizer=optimizer[i], metrics=['accuracy'])
            model.fit(x_train,y_train,validation_data= (x_val,y_val),verbose=2,epochs=epoch)
            y_prob_val = model.predict(x_val)  # output predict probability
            probability = [prob[1] for prob in y_prob_val]
            new_single_features = np.append(new_single_features, probability)

            y_prob_test = model.predict(test_x)
            probability_test = [prob_test[1] for prob_test in y_prob_test]
            new_single_test_features[j] = probability_test
            j += 1
        new_train_x[i] = new_single_features
        new_test_x[i] = np.mean(new_single_test_features, axis=0)
    return new_train_x.T, new_train_y, new_test_x.T


def get_stacking_prob(models, train_data_x, train_data_y, test_data_x, n_folds):
    # np.random.seed(6)
    train_num, test_num = train_data_x.shape[0], test_data_x.shape[0]
    meta_train_x = np.zeros((len(models), train_num))
    meta_test_x = np.zeros((len(models), test_num))

    for i in range(len(models)):
        model = models[i]
        meta_data_train_x = np.array([])
        meta_data_train_y = np.array([])

        new_test_nfolds = np.zeros((n_folds,test_num))

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
        for j, (train_index, test_index) in enumerate(kf.split(train_data_x)):
            train_x, train_y = train_data_x[train_index], train_data_y[train_index]
            test_x, test_y = train_data_x[test_index], train_data_y[test_index]

            model.fit(train_x, train_y)
            y_probability_train = model.predict_proba(test_x)
            meta_data_train_x = np.append(meta_data_train_x,[x[1] for x in y_probability_train])
            meta_data_train_y = np.append(meta_data_train_y, test_y)

            y_probability_test = model.predict_proba(test_data_x)
            new_test_nfolds[j] = [x[1] for x in y_probability_test]


        meta_train_x[i] = meta_data_train_x
        meta_test_x[i] = new_test_nfolds.mean(axis=0)
    return meta_train_x.T, meta_data_train_y, meta_test_x.T
# def get_stacking(models, train_x, n_models):
#     new_train_x = np.zeros(shape = (n_models, len(train_x)))
#     for i in range(len(models)):
#         model = models[i]
#         y_prob = model.predict(train_x)
#         y_new_features = [prob[1] for prob in y_prob]
#         new_train_x[i] = y_new_features
#     return new_train_x.T

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


LR = LogisticRegression(penalty='l2', C=0.1)
SVM = svm.SVC(probability=True, C=1000, gamma=0.001)
cnn_model = cnn()
rnn_model = rnn()
models = [cnn_model, rnn_model]

col, row, geotransform, proj, total_data = read_tif('yanshan12.tif')
total_data_DL = np.reshape(total_data, [col*row, total_data.shape[2], 1])
total_data = np.reshape(total_data, [col*row, total_data.shape[2]])
CV = 3
# new_train_x_first, new_train_y_first, new_test_x_first = get_stacking(models, train_x, train_y, test_x, CV)
# new_train_x_second, new_train_y_second, new_test_x_second = Stacking.get_stacking_prob([LR, SVM], train_x_1D, train_y_1D,
#                                                                                       test_x_1D, CV)
new_train_x_first, new_train_y_first, new_total_x_first = get_stacking(models, train_x, train_y, total_data_DL, CV)
new_train_x_second, new_train_y_second, new_total_x_second = Stacking.get_stacking_prob([LR, SVM], train_x_1D, train_y_1D,
                                                                                        total_data, CV)

# meta_train = np.concatenate((new_train_x_first, new_train_x_second), axis=1)
# meta_test = np.concatenate((new_test_x_first, new_test_x_second), axis=1)

meta_total = np.concatenate((new_total_x_first, new_total_x_second), axis=1)

# np.save('train_stacking_x_new.npy', meta_train)
# np.save('test_stacking_x_new.npy', meta_test)
# np.save('train_stacking_y_new.npy', new_train_y_first)

# np.save('train_stacking_total_first.npy', new_total_x_first)
np.save('train_stacking_total.npy', meta_total)

