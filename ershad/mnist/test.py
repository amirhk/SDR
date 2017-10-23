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
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.stats import norm
from sklearn import mixture

from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from keras import objectives , utils,optimizers
from keras.datasets import mnist
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf


import sys
import platform
if platform.system() == 'Windows':
  sys.path.append('..\\..\\utils')
elif platform.system() == 'Darwin':
  sys.path.append('../utils')

from importDatasets import importMnist
from importDatasets import importMnistFashion
from importDatasets import importOlivetti
from importDatasets import importSquareAndCross

from importDatasets import importDatasetForSemisupervisedTraining

# -----------------------------------------------------------------------------
#                                                                    Fetch Data
# -----------------------------------------------------------------------------

# fh_import_dataset = lambda : importMnist()
# fh_import_dataset = lambda : importMnistFashion()

# (dataset_name,
#   x_train,
#   x_test,
#   y_train,
#   y_test,
#   sample_dim,
#   sample_channels,
#   original_dim,
#   num_classes) = fh_import_dataset()


fh_import_dataset = lambda : importDatasetForSemisupervisedTraining('mnist',30000,30000)
(dataset_name,
  x_train,
  x_test,
  y_train,
  y_test,
  sample_dim,
  sample_channels,
  original_dim,
  num_classes,
  x_train_labeled,
  y_train_labeled,
  x_val,
  y_val,
  x_train_unlabeled,
  y_train_unlabeled) = fh_import_dataset()


x_total = np.concatenate([x_train_unlabeled,x_train_labeled],1)
x_total_test = np.concatenate([x_test,x_test],1)

num_classes = 10

batch_size = 100
latent_dim = 2
epochs = 20
intermediate_dim = 500
epsilon_std = 1.0
learning_rate = 0.005


# -----------------------------------------------------------------------------
#                                                                   Build Model
# -----------------------------------------------------------------------------

x = Input(batch_shape=(batch_size, 2*original_dim))

x_reshaped = Reshape((2,original_dim))

h = Dense(intermediate_dim, activation='relu')
z_mean = Dense(latent_dim)
z_log_var = Dense(latent_dim)

_x_reshaped = x_reshaped(x)
_h = h(_x_reshaped)
_z_mean = z_mean(_h)

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean_reshaped = Dense(original_dim, activation='sigmoid')
decoded_mean = Reshape((2*original_dim,))
h_decoded = decoder_h(_z_mean)
x_decoded_mean_reshaped = decoder_mean_reshaped(h_decoded)
x_decoded_mean = decoded_mean(x_decoded_mean_reshaped)





def vae_loss(x, x_decoded_mean):
    xent_loss = 2*original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    return xent_loss

my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.1)

vae = Model(inputs = [x], outputs =[x_decoded_mean]) #(x,x_decoded_mean)
vae.compile(optimizer=my_adam, loss=vae_loss)


# -----------------------------------------------------------------------------
#                                                                   Train Model
# -----------------------------------------------------------------------------
#
#model_weights = pickle.load(open('weights_vaesdr_' + str(latent_dim) + 'd_trained_on_' + dataset_name, 'rb'))
#vae.set_weights(model_weights)

vae.fit(x_total,x_total,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size)

model_weights = vae.get_weights()
pickle.dump((model_weights), open('weights_vaesdr_' + str(latent_dim) + 'd_trained_on_' + dataset_name, 'wb'))


# -----------------------------------------------------------------------------
#                                                                      Analysis
# -----------------------------------------------------------------------------

# build a model to project inputs on the latent space
encoder = Model(x, _z_mean)

# display a 3D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_total_test, batch_size=batch_size)
y_test_label = np.argmax(y_test,axis =1)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], x_test_encoded[:, 2], linewidth = 0, c=y_test_label)
#ax.colorbar()
#ax.show()



## display a 2D plot of the digit classes in the latent space


plt.scatter(x_test_encoded[:, 0,0], x_test_encoded[:, 0,1], linewidth = 0, c=y_test_label)
plt.colorbar()
plt.show()

#y_test_onehot = utils.to_categorical(y_test, num_classes)



_x_reshaped_ = x_reshaped(x)
_h_ = h(_x_reshaped_)
_z_mean_ = z_mean(_h_)
_decoder_h_ =  decoder_h(_z_mean_)
_decoder_mean_reshaped = decoder_mean_reshaped(_decoder_h_)
_decoder_mean_ = decoded_mean(_decoder_mean_reshaped)

vaeencoder = Model(inputs = x,outputs=  _decoder_mean_)
x_decoded = vaeencoder.predict(x_total_test,batch_size = batch_size)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)


                                        # -------------------------------------
                                        #                              Accuracy
                                        # -------------------------------------
lll_real = np.zeros((10000,1))
lll = np.zeros((10000,1))
for i in range(10000):
    m = b[i,:].reshape(10,).tolist()
    m_real = y_test[i,:].reshape(10,).tolist()
    lll[i,0] = m.index(max(m))
    lll_real[i,0] = m_real.index(max(m_real))



lll = lll.reshape(1,10000).astype('uint8')
lll_real = lll_real.reshape(1,10000).astype('uint8')
n_error = np.count_nonzero(lll - lll_real)

print(1- n_error/10000)


                                        # -------------------------------------
                                        #                               Fit GMM
                                        # -------------------------------------

# display a 2D plot of the digit classes in the latent space
x_train_encoded = encoder.predict(x_train, batch_size=batch_size)

n_components = num_classes
cv_type = 'full'
gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
gmm.fit(x_train_encoded)




                                        # -------------------------------------
                                        #                                 Plots
                                        # -------------------------------------



def getFigureOfSamplesForInput(x_samples, sample_dim, number_of_sample_images, grid_x, grid_y):
    figure = np.zeros((sample_dim * number_of_sample_images, sample_dim * number_of_sample_images))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            digit = x_samples[i * number_of_sample_images + j, :].reshape(sample_dim, sample_dim)
            figure[i * sample_dim: (i + 1) * sample_dim,
                   j * sample_dim: (j + 1) * sample_dim] = digit
    return figure


number_of_sample_images = 15
grid_x = norm.ppf(np.linspace(0.05, 0.95, number_of_sample_images))
grid_y = norm.ppf(np.linspace(0.05, 0.95, number_of_sample_images))

plt.figure()

ax = plt.subplot(1,3,1)
x_samples_a = x_test
canvas = getFigureOfSamplesForInput(x_samples_a, sample_dim, number_of_sample_images, grid_x, grid_y)
plt.imshow(canvas, cmap='Greys_r')
ax.set_title('Original Test Images', fontsize=8)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(1,3,2)
x_samples_b = x_decoded
canvas = getFigureOfSamplesForInput(x_samples_b, sample_dim, number_of_sample_images, grid_x, grid_y)
plt.imshow(canvas, cmap='Greys_r')
ax.set_title('Reconstructed Test Images', fontsize=8)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(1,3,3)
x_samples_c = gmm.sample(10000)
x_samples_c = np.random.permutation(x_samples_c[0]) # need to randomly permute because gmm.sample samples 1000 from class 1, then 1000 from class 2, etc.
x_samples_c = generator.predict(x_samples_c)
canvas = getFigureOfSamplesForInput(x_samples_c, sample_dim, number_of_sample_images, grid_x, grid_y)
plt.imshow(canvas, cmap='Greys_r')
ax.set_title('Generated Images', fontsize=8)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()
# plt.savefig('images/'+ dataset_name + '_samples.png')












