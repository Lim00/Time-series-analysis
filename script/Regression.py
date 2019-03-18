import tensorflow as tf
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential,load_model, Model
from keras.layers import Dense, Dropout, LSTM, TimeDistributed, Input, Masking, LeakyReLU, BatchNormalization, Conv2D, MaxPooling2D, Permute, Flatten
from keras.regularizers import L1L2

import numpy as np

def get_optimizer(optimizer_name, learning_rate):
    if (optimizer_name == "SGD"):
        return keras.optimizers.sgd(lr=learning_rate)
    elif (optimizer_name == "ADAM"):
        return keras.optimizers.adam(lr=learning_rate)
    elif (optimizer_name == "RMS"):
        return keras.optimizers.rmsprop(lr=learning_rate)
    else:
        return keras.optimizers.sgd(lr=learning_rate)


def model_dl_lstm1(sequence_length, num_features, learning_rate, optimizer_name, class_num):

    model = Sequential()

    model.add(LSTM(input_shape=(sequence_length, num_features), units=64, return_sequences=False))

    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))
    model.add(Activation("linear"))

    optimizer = get_optimizer(optimizer_name, learning_rate)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    print(model.summary())

    return model


def model_dl_lstm2(sequence_length, num_features, learning_rate, optimizer_name, class_num):
    model = Sequential()

    model.add(LSTM(input_shape=(sequence_length, num_features), units=128, return_sequences=True))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.1))

    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))
    model.add(Activation("linear"))

    optimizer = get_optimizer(optimizer_name, learning_rate)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    print(model.summary())

    return model


def model_dl_cnn_fullFilter(sequence_length, num_features, learning_rate, optimizer_name, class_num):

    model = Sequential()

    model.add(Conv2D(16, (num_features, 4), padding='valid', input_shape=(sequence_length, num_features, 1)))
    model.add(Activation('relu'))
    model.add(Permute((3, 2, 1)))
    model.add(MaxPooling2D(pool_size=(1, 2)))

    second_feature = model.output_shape[1]

    model.add(Conv2D(32, (second_feature, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Permute((3, 2, 1)))
    model.add(MaxPooling2D(pool_size=(1, 2)))

    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1))
    model.add(Activation("linear"))

    optimizer = get_optimizer(optimizer_name, learning_rate)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    print(model.summary())

    return model


def model_dl_cnn(sequence_length, num_features, learning_rate, optimizer_name, class_num):

    model = Sequential()

    # Layer 1
    model.add(Conv2D(16, (3, 3), padding='valid', input_shape=(sequence_length, num_features, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))

    # Layer 2
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))

    # Fully connection layer
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1))
    model.add(Activation("linear"))

    optimizer = get_optimizer(optimizer_name, learning_rate)

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    print(model.summary())

    return model

def model_ml_svm(sequence_length, num_features, learning_rate, optimizer_name, class_num):
    from sklearn import svm

    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    return svm.SVR()

def model_ml_knn(sequence_length, num_features, learning_rate, optimizer_name, class_num):
    from sklearn.neighbors import KNeighborsRegressor

    # https://scikit-learn.org/stable/modules/neighbors.html
    return KNeighborsRegressor(5)

def model_ml_tree(sequence_length, num_features, learning_rate, optimizer_name, class_num):
    from sklearn.tree import DecisionTreeRegressor

    # https://scikit-learn.org/stable/modules/tree.html
    return DecisionTreeRegressor()
