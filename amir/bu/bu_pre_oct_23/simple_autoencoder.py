#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
======================== No condition bit =====================

"""
import numpy as np
import tensorflow as tf
import pickle
import keras
from keras.datasets  import mnist
from keras.models    import Model, Sequential
from keras.layers    import Dense, Input, Reshape, Lambda, Concatenate
from keras.callbacks import Callback
from keras           import backend as K
from keras           import objectives , optimizers, callbacks
import datetime
import os
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# (X_train, _), (X_test, _) = mnist.load_data()

# X_train = X_train.astype('float32') / 255.
# X_test = X_test.astype('float32') / 255.
# X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
# X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
# print(X_train.shape)
# print(X_test.shape)

X_train, Y_train = pickle.load(open('plane_dataset_train', 'rb'))
X_train = X_train.reshape([X_train.shape[0], -1])
X_test, Y_test = pickle.load(open('plane_dataset_test', 'rb'))
X_test = X_test.reshape([X_test.shape[0], -1])

input_dim = X_train.shape[1]
default_act_func = 'relu'
batch_size = 256
latent_dim = 32 # 4


def encoder():
    model = Sequential()
    model.add(Dense(latent_dim, input_shape=(input_dim,), activation=default_act_func))
    return model

def decoder():
    model = Sequential()
    model.add(Dense(input_dim, input_shape=(latent_dim,), activation='sigmoid')) ############# input should be normalized
    return model

def encoder_decoder(e,d):
    model = Sequential()
    model.add(e)
    model.add(d)
    return model

encoder = encoder()
decoder = decoder()
encoder_decoder = encoder_decoder(encoder, decoder)

encoder_decoder_lr = 0.001
encoder_decoder_optimizer = optimizers.Adam(lr=encoder_decoder_lr, beta_1=0.1)


# scenario 1
# encoder_decoder.compile(encoder_decoder_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# scenario 2
encoder_decoder.compile(encoder_decoder_optimizer, loss='mean_squared_error', metrics=['accuracy'])

h = encoder_decoder.fit(X_train, X_train,
                epochs=5,
                batch_size=256,
                shuffle=True)


