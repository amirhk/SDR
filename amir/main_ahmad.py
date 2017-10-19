#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:58:50 2017

======================== No condition bit =====================

"""
import numpy as np
import pickle
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Reshape, Lambda, Concatenate
from keras import backend as K
import tensorflow as tf
from keras import objectives , optimizers, callbacks
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import datetime
from keras.callbacks import Callback
import os
import matplotlib.pyplot as plt

# from tensorflow.python import debug as tf_debug

# from keras.datasets import mnist
# import numpy as np
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


# img_size = [40,40]
input_dim = X_train.shape[1]
default_act_func = 'relu'
batch_size = 100
latent_dim = 2 # 32


def encoder():
    model = Sequential()
    model.add(Dense(latent_dim, input_shape=(input_dim,), activation=default_act_func))
    return model

def decoder():
    model = Sequential()
    model.add(Dense(input_dim, input_shape=(latent_dim,), activation='sigmoid')) ############# input should be normalized
    return model


# def encoder():
#     model = Sequential()
#     model.add(Dense(500, input_shape=(input_dim,), activation=default_act_func))
#     model.add(Dense(200, activation=default_act_func))
#     model.add(Dense(100, activation=default_act_func))
#     model.add(Dense(latent_dim))
#     return model

# def decoder():
#     model = Sequential()
#     model.add(Dense(100, input_shape=(latent_dim,), activation=default_act_func))
#     model.add(Dense(200, activation=default_act_func))
#     model.add(Dense(500, activation=default_act_func))
#     model.add(Dense(input_dim, activation='sigmoid')) ############# input should be normalized
#     return model

def encoder_decoder(e,d):
    model = Sequential()
    model.add(e)
    model.add(d)
    return model

def mixing(z):
    z_1 = z[:,:int(latent_dim/2)]
    z_2 = z[:,int(latent_dim/2):]
    idx_1 = np.int32(np.floor(np.random.uniform(0,batch_size - 0.000001, batch_size)))
    idx_2 = np.int32(np.floor(np.random.uniform(0,batch_size - 0.000001, batch_size)))
    return K.concatenate([K.gather(z_1,idx_1), K.gather(z_2,idx_2)], axis=1)

def discriminator():
    model = Sequential()
    model.add(Dense(20, input_shape=(latent_dim,), activation=default_act_func))
    model.add(Dense(20, activation=default_act_func))
    model.add(Dense(20, activation=default_act_func))
    model.add(Dense(1, activation='sigmoid'))
    return model

def discriminator_2(d):
    z = Input(shape=(latent_dim,))
    mixed_z = Lambda(mixing, output_shape=(latent_dim,))(z)
    d1 = d(z)
    d2 = d(mixed_z)
    c = Concatenate()([d1,d2])
    model = Model(z, c)
    return model

def encoder_discriminator(encoder,discriminator):
    model = Sequential()
    model.add(encoder)
    model.add(discriminator)
    return model

encoder = encoder()
decoder = decoder()
discriminator = discriminator()
discriminator_2 = discriminator_2(discriminator)
encoder_decoder = encoder_decoder(encoder, decoder)
encoder_discriminator_2 = encoder_discriminator(encoder, discriminator_2)


# encoder_decoder_optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9)
# # encoder_decoder_lr = 0.0001
# # encoder_decoder_optimizer = optimizers.Adam(lr=encoder_decoder_lr, beta_1=0.1)
# #encoder_decoder_optimizer = optimizers.SGD(lr=encoder_decoder_lr, momentum=0.9, nesterov=True)
# encoder_decoder.compile(encoder_decoder_optimizer, loss='binary_crossentropy') # , metrics=['accuracy'])


# discriminator_2_optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9)
# # discriminator_2_lr = 0.0001
# # discriminator_2_optimizer = optimizers.Adam(lr=discriminator_2_lr, beta_1=0.1)
# # discriminator_2_optimizer = optimizers.SGD(lr=discriminator_2_lr, momentum=0.1, nesterov=True)
# discriminator_2.compile(discriminator_2_optimizer, loss='binary_crossentropy')

# def negative_binary_crossentropy(c_true, c_est):
#     return -1 * keras.losses.binary_crossentropy(c_true, c_est)


# encoder_discriminator_2_optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9)
# # encoder_discriminator_2_lr = 0.0001
# # encoder_discriminator_2_optimizer = optimizers.Adam(lr=encoder_discriminator_2_lr, beta_1=0.1)
# # encoder_discriminator_2_optimizer = optimizers.SGD(lr=encoder_discriminator_2_lr, momentum=0.9, nesterov=True)
# encoder_discriminator_2.compile(encoder_discriminator_2_optimizer, loss=negative_binary_crossentropy)

encoder_decoder_lr = 0.001
encoder_decoder_optimizer = optimizers.Adam(lr=encoder_decoder_lr, beta_1=0.1)
#encoder_decoder_optimizer = optimizers.SGD(lr=encoder_decoder_lr, momentum=0.9, nesterov=True)
encoder_decoder.compile(encoder_decoder_optimizer, loss='mean_squared_error')

discriminator_2_lr = 0.001
#discriminator_2_optimizer = optimizers.Adam(lr=discriminator_2_lr, beta_1=0.1)
discriminator_2_optimizer = optimizers.SGD(lr=discriminator_2_lr, momentum=0.1, nesterov=True)
discriminator_2.compile(discriminator_2_optimizer, loss='binary_crossentropy')

def negative_binary_crossentropy(c_true, c_est):
    return -1 * keras.losses.binary_crossentropy(c_true, c_est)

encoder_discriminator_2_lr = 0.001
#encoder_discriminator_2_optimizer = optimizers.Adam(lr=encoder_discriminator_2_lr, beta_1=0.1)
encoder_discriminator_2_optimizer = optimizers.SGD(lr=encoder_discriminator_2_lr, momentum=0.9, nesterov=True)
encoder_discriminator_2.compile(encoder_discriminator_2_optimizer, loss=negative_binary_crossentropy)













# encoder.summary()
# decoder.summary()
# encoder_decoder.summary()


# encoder_decoder.summary()
# discriminator_2.summary()
# encoder_discriminator_2.summary()


# def scheduler(epoch):
#     if epoch == 5:
#         model.lr.set_value(.02)
#     return model.lr.get_value()

# change_lr = LearningRateScheduler(scheduler)

# model.fit(x_embed, y, nb_epoch=1, batch_size = batch_size, show_accuracy=True,
#        callbacks=[chage_lr])

# encoder_decoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

# h = encoder_decoder.fit(X_train, X_train,
#                 epochs=5,
#                 batch_size=256,
#                 shuffle=True,
#                 # verbose=0,
#                 validation_data=(X_test, X_test))

# encoder_decoder_optimizer.lr = 0.001
# h = encoder_decoder.fit(X_train, X_train,
#                 epochs=10,
#                 batch_size=256,
#                 shuffle=True)

# encoder_decoder_optimizer.lr = 0.0003
# h = encoder_decoder.fit(X_train, X_train,
#                 epochs=10,
#                 batch_size=256,
#                 shuffle=True)

# encoder_decoder_optimizer.lr = 0.0001
# h = encoder_decoder.fit(X_train, X_train,
#                 epochs=10,
#                 batch_size=256,
#                 shuffle=True)




y_batch = np.concatenate( [np.zeros((batch_size,1)), np.ones((batch_size,1))], axis=1)
for epoch in range(300):

    # if epoch == 0:
    #     encoder_decoder_optimizer.lr = 0.001
    # elif epoch == 5:
    #     encoder_decoder_optimizer.lr = 0.0005
    # elif epoch == 10:
    #     encoder_decoder_optimizer.lr = 0.0001

    if epoch == 0:
        encoder_decoder_optimizer.lr = 0.001
        discriminator_2_optimizer.lr = 0.001
        encoder_discriminator_2_optimizer.lr = 0.001
    elif epoch == 100-90:
        encoder_decoder_optimizer.lr = 0.0001
        discriminator_2_optimizer.lr = 0.0005
        encoder_discriminator_2_optimizer.lr = 0.0001
    elif epoch == 170-90:
        encoder_decoder_optimizer.lr = 0.0001
        discriminator_2_optimizer.lr = 0.0005
        encoder_discriminator_2_optimizer.lr = 0.0001
    elif epoch == 200-90:
        encoder_decoder_optimizer.lr = 0.0001
        discriminator_2_optimizer.lr = 0.0005
        encoder_discriminator_2_optimizer.lr = 0.0001
    elif epoch == 250-90:
        encoder_decoder_optimizer.lr = 0.0001
        discriminator_2_optimizer.lr = 0.0005
        encoder_discriminator_2_optimizer.lr = 0.0001
    elif epoch == 300-90:
        encoder_decoder_optimizer.lr = 0.00001
        discriminator_2_optimizer.lr = 0.00005
        encoder_discriminator_2_optimizer.lr = 0.00001
    elif epoch == 350-90:
        encoder_decoder_optimizer.lr = 0.00001
        discriminator_2_optimizer.lr = 0.00005
        encoder_discriminator_2_optimizer.lr = 0.00001

    # lr = encoder_decoder.optimizer.lr
    # print(lr)
    # encoder_decoder.optimizer.lr.set_value(99)
    # lr = encoder_decoder.optimizer.lr
    # print(lr)
    # lr = K.eval(encoder_decoder_optimizer.lr * (1. / (1. + encoder_decoder_optimizer.decay * encoder_decoder_optimizer.iterations)))
    # print('\nLR: {:.6f}\n'.format(lr))

    reconstruction_losses = []
    discriminator_2_losses = []
    encoder_discriminator_2_losses = []
    # print("### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###")
    print("Epoch #%d" % (epoch))
    for index in range(int(X_train.shape[0]/batch_size)):
        x_batch = X_train[index*batch_size:(index+1)*batch_size]
        # y_batch = Y_train[index*batch_size:(index+1)*batch_size]
        z = encoder.predict(x_batch)

        reconstruction_loss = encoder_decoder.train_on_batch(x_batch, x_batch)
        reconstruction_losses.append(reconstruction_loss)
        # reconstruction_losses.append(reconstruction_loss[0])

    # print(sum(reconstruction_losses) / len(reconstruction_losses))
        # print("batch %d reconstruction_loss : %f" % (index, reconstruction_loss))
        # print("[INFO] batch %d / %d: \t\t\t reconstruction_loss: %f \t discriminator_loss: %f \t encoder_discriminator_loss: %f" % (index, int(X_train.shape[0]/batch_size), reconstruction_loss, reconstruction_loss, reconstruction_loss))

        discriminator.trainable = True
        discriminator_2_loss = discriminator_2.train_on_batch(z, y_batch)
        discriminator_2_losses.append(discriminator_2_loss)
        # print("\tbatch %d discriminator_loss : %f" % (index, discriminator_2_loss))

        discriminator.trainable = False
        encoder_discriminator_2_loss = encoder_discriminator_2.train_on_batch(x_batch, y_batch)
        encoder_discriminator_2_losses.append(encoder_discriminator_2_loss)
        # print("\t\tbatch %d encoder_discriminator_loss : %f" % (index, encoder_discriminator_2_loss))

        # discriminator.trainable = True
        # discriminator_2_loss = discriminator_2.train_on_batch(z, y_batch)
        # discriminator_2_losses.append(discriminator_2_loss)
        # # print("\tbatch %d discriminator_loss : %f" % (index, discriminator_2_loss))

        # discriminator.trainable = False
        # encoder_discriminator_2_loss = encoder_discriminator_2.train_on_batch(x_batch, y_batch)
        # encoder_discriminator_2_losses.append(encoder_discriminator_2_loss)
        # # print("\t\tbatch %d encoder_discriminator_loss : %f" % (index, encoder_discriminator_2_loss))

        # print("[INFO] batch %d / %d: \t\t\t reconstruction_loss: %f \t discriminator_loss: %f \t encoder_discriminator_loss: %f" % (index, int(X_train.shape[0]/batch_size), reconstruction_loss, discriminator_2_loss, encoder_discriminator_2_loss))
    print("[INFO] \t\t reconstruction_loss: %f \t discriminator_loss: %f \t encoder_discriminator_loss: %f" % (sum(reconstruction_losses) / len(reconstruction_losses), sum(discriminator_2_losses) / len(discriminator_2_losses), sum(encoder_discriminator_2_losses) / len(encoder_discriminator_2_losses)))

    if epoch % 10 == 9:
        encoder.save_weights("encoder_epoch_{}".format(epoch), True)
        decoder.save_weights("decoder_epoch_{}".format(epoch),True)
        discriminator.save_weights("discriminator_epoch_{}".format(epoch), True)





























