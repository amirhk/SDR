# -------------------------------------------------------------------------
# Copyright (c) 2017, Amir-Hossein Karimi, Ershad Banijamali
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the distribution

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
from sklearn import mixture

from keras.layers import Input, Dense, Lambda, Reshape, Dropout
from keras.models import Model
from keras import backend as K
from keras import objectives, utils, optimizers
import tensorflow as tf
from keras.callbacks import Callback

import os
import sys
import platform
if platform.system() == 'Windows':
  sys.path.append('..\\..\\utils')
elif platform.system() == 'Linux' or platform.system() == 'Darwin':
  sys.path.append('../../utils')

from importDatasets import importMnist
from importDatasets import importMnistFashion
from importDatasets import importOlivetti
from importDatasets import importSquareAndCross
from importDatasets import importIris
from importDatasets import importBalance
from importDatasets import importGlass
from importDatasets import importDatasetForSemisupervisedTraining

from datetime import datetime

# -----------------------------------------------------------------------------
#                                                                    Fetch Data
# -----------------------------------------------------------------------------

# fh_import_dataset = lambda : importIris()
# fh_import_dataset = lambda : importBalance()
fh_import_dataset = lambda : importGlass()

(dataset_name,
  x_train,
  x_test,
  y_train,
  y_test,
  sample_dim,
  sample_channels,
  original_dim,
  num_classes) = fh_import_dataset()

batch_size = 25
latent_dim = 2
epochs = 200
epsilon_std = 1.0
learning_rate = 0.0001
intermediate_recon_layer_dim = 10
intermediate_label_layer_dim = 10

shallow = True

# -----------------------------------------------------------------------------
#                                                                  Path-related
# -----------------------------------------------------------------------------

experiment_name = dataset_name + \
    '_____z_dim_' + str(latent_dim)

# if ~ os.path.isdir('../experiments'):
#   os.makedirs('../experiments')
experiment_dir_path = '../experiments/exp' + \
    '_____' + \
    str(datetime.now().strftime('%Y-%m-%d_____%H-%M-%S')) + \
    '_____' + \
    experiment_name
os.makedirs(experiment_dir_path)

# -----------------------------------------------------------------------------
#                                                                   Build Model
# -----------------------------------------------------------------------------

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_recon_layer_dim, activation='relu')
z_mean = Dense(latent_dim)
z_log_var = Dense(latent_dim)

if shallow:
    # _h = h(x)
    _z_mean = z_mean(x)
    _z_log_var = z_log_var(x)
else:
    _h = h(x)
    _z_mean = z_mean(_h)
    _z_log_var = z_log_var(_h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([_z_mean, _z_log_var])

# we instantiate these layers separately so as to reuse them later
if shallow:

    decoder_h_x = Dense(intermediate_recon_layer_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    # h_decoded = decoder_h_x(z)
    x_decoded_mean = decoder_mean(z)

    decoder_h_y = Dense(intermediate_label_layer_dim, activation='relu')
    y_decoded = Dense(num_classes, activation='sigmoid')
    # h_decoded_2 = decoder_h_y(z)
    _y_decoded = y_decoded(z)

else:

    decoder_h_x = Dense(intermediate_recon_layer_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h_x(z)
    x_decoded_mean = decoder_mean(h_decoded)

    decoder_h_y = Dense(intermediate_label_layer_dim, activation='relu')
    y_decoded = Dense(num_classes, activation='sigmoid')
    h_decoded_2 = decoder_h_y(z)
    _y_decoded = y_decoded(h_decoded_2)

yy = Input(batch_shape = (batch_size,num_classes))

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + _z_log_var - K.square(_z_mean) - K.exp(_z_log_var), axis=-1)
    y_loss = num_classes * objectives.categorical_crossentropy(yy, _y_decoded)
    return xent_loss + kl_loss + num_classes*y_loss

my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.1)

vae = Model(inputs = [x,yy], outputs =[x_decoded_mean,_y_decoded]) #(x,x_decoded_mean)
vae.compile(optimizer=my_adam, loss=vae_loss)



# y_test_label = np.argmax(y_test, axis=1)

# Accuracy = np.zeros((epochs,1))
# ii=0
# pickle.dump((ii),open('counter','wb'))
# text_file_name = experiment_dir_path + '/accuracy_log.txt'
# class ACCURACY(Callback):

#     def on_epoch_end(self,batch,logs = {}):
#         ii= pickle.load(open('counter', 'rb'))
#         _, b  = vaeencoder.predict(x_total_test, batch_size = batch_size)
#         Accuracy[ii, 0]

#         lll = np.argmax(b, axis =1)
#         n_error = np.count_nonzero(lll - y_test_label)
#         ACC = 1 - n_error / 10000
#         Accuracy[ii,0] = ACC
#         print('\n accuracy = ', ACC)
#         ii= ii + 1
#         pickle.dump((ii),open('counter', 'wb'))
#         with open(text_file_name, 'a') as text_file:
#           print('Epoch #{} Accuracy:{} \n'.format(ii, ACC), file=text_file)

# accuracy = ACCURACY()



# -----------------------------------------------------------------------------
#                                                                   Train Model
# -----------------------------------------------------------------------------

weights_file_name = '../saved_weights/weights_vaesdr_' + str(latent_dim) + 'd_trained_on_' + dataset_name
# if os.path.isfile(weights_file_name):
#   print('[INFO] saved weights file found; loading...')
#   model_weights = pickle.load(open(weights_file_name, 'rb'))
#   vae.set_weights(model_weights)
# else:
#   print('[INFO] NO saved weights file found; starting from scratch...')

vae.fit([x_train, y_train],[x_train,y_train],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size) #,
        # callbacks = [accuracy])

model_weights = vae.get_weights()
pickle.dump((model_weights), open(weights_file_name, 'wb'))

# -----------------------------------------------------------------------------
#                                                                      Analysis
# -----------------------------------------------------------------------------

# build a model to project inputs on the latent space
encoder = Model(x, _z_mean)

x_train_encoded = encoder.predict(x_train, batch_size=batch_size)
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)

## display a 2D plot of the digit classes in the latent space
fig = plt.figure()

y_train_label = np.argmax(y_train, axis=1)
y_test_label = np.argmax(y_test, axis=1)

# plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], linewidth = 0, c=cm.rainbow(np.linspace(0, 1, num_classes)))
plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1], linewidth = 0, c=y_train_label)
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], linewidth = 0, c=y_test_label, alpha=0.3) #facecolors='none', edgecolors=y_test)

# plt.colorbar()
# plt.show()
plt.savefig(experiment_dir_path + '/latent_space.png')

# _h_ = h(x)
# _z_mean_ = z_mean(_h_)
# _decoder_h_ =  decoder_h_x(_z_mean_)
# _decoder_mean_ = decoder_mean(_decoder_h_)
# h_decoded_2_ = decoder_h_y(_z_mean_)
# _y_decoded_ = y_decoded(h_decoded_2_)

# vaeencoder = Model(x,[_decoder_mean_,_y_decoded_])
# x_decoded, b  = vaeencoder.predict(x_test,batch_size = batch_size)

# # build a digit generator that can sample from the learned distribution
# decoder_input = Input(shape=(latent_dim,))
# _h_decoded = decoder_h_x(decoder_input)
# _x_decoded_mean = decoder_mean(_h_decoded)
# generator = Model(decoder_input, _x_decoded_mean)


#                                         # -------------------------------------
#                                         #                              Accuracy
#                                         # -------------------------------------

# lll = np.zeros((10000,1))
# for i in range(10000):
#     m = b[i,:].reshape(10,).tolist()
#     lll[i,0] = m.index(max(m))

# lll

# lll.reshape(1,10000)
# lll
# lll = lll.reshape(1,10000).astype('uint8')

# n_error = np.count_nonzero(lll - y_test)

# print(1- n_error/10000)


#                                         # -------------------------------------
#                                         #                               Fit GMM
#                                         # -------------------------------------

# # display a 2D plot of the digit classes in the latent space
# x_train_encoded = encoder.predict(x_train, batch_size=batch_size)

# n_components = num_classes
# cv_type = 'full'
# gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
# gmm.fit(x_train_encoded)




#                                         # -------------------------------------
#                                         #                                 Plots
#                                         # -------------------------------------



# def getFigureOfSamplesForInput(x_samples, sample_dim, number_of_sample_images, grid_x, grid_y):
#     figure = np.zeros((sample_dim * number_of_sample_images, sample_dim * number_of_sample_images))
#     for i, yi in enumerate(grid_x):
#         for j, xi in enumerate(grid_y):
#             digit = x_samples[i * number_of_sample_images + j, :].reshape(sample_dim, sample_dim)
#             figure[i * sample_dim: (i + 1) * sample_dim,
#                    j * sample_dim: (j + 1) * sample_dim] = digit
#     return figure


# number_of_sample_images = 15
# grid_x = norm.ppf(np.linspace(0.05, 0.95, number_of_sample_images))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, number_of_sample_images))

# plt.figure()

# ax = plt.subplot(1,3,1)
# x_samples_a = x_test
# canvas = getFigureOfSamplesForInput(x_samples_a, sample_dim, number_of_sample_images, grid_x, grid_y)
# plt.imshow(canvas, cmap='Greys_r')
# ax.set_title('Original Test Images', fontsize=8)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

# ax = plt.subplot(1,3,2)
# x_samples_b = x_decoded
# canvas = getFigureOfSamplesForInput(x_samples_b, sample_dim, number_of_sample_images, grid_x, grid_y)
# plt.imshow(canvas, cmap='Greys_r')
# ax.set_title('Reconstructed Test Images', fontsize=8)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

# ax = plt.subplot(1,3,3)
# x_samples_c = gmm.sample(10000)
# x_samples_c = np.random.permutation(x_samples_c[0]) # need to randomly permute because gmm.sample samples 1000 from class 1, then 1000 from class 2, etc.
# x_samples_c = generator.predict(x_samples_c)
# canvas = getFigureOfSamplesForInput(x_samples_c, sample_dim, number_of_sample_images, grid_x, grid_y)
# plt.imshow(canvas, cmap='Greys_r')
# ax.set_title('Generated Images', fontsize=8)
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

# # plt.show()
# plt.savefig(experiment_dir_path + '/generated_samples.png')













