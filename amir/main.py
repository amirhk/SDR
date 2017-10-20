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

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives , utils,optimizers
from keras.datasets import mnist
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('../utils')

from importDatasets import importMnist
from importDatasets import importMnistFashion
from importDatasets import importOlivetti
from importDatasets import importSquareAndCross

# -----------------------------------------------------------------------------
#                                                                    Fetch Data
# -----------------------------------------------------------------------------

fh_import_dataset = lambda : importMnist()
# fh_import_dataset = lambda : importMnistFashion()

(dataset_name,
  x_train,
  x_test,
  y_train,
  y_test,
  sample_dim,
  sample_channels,
  original_dim,
  num_classes) = fh_import_dataset()

batch_size = 100
latent_dim = 3
epochs = 0
intermediate_dim = 500
epsilon_std = 1.0
learning_rate = 0.00001


# -----------------------------------------------------------------------------
#                                                                   Build Model
# -----------------------------------------------------------------------------

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')
z_mean = Dense(latent_dim)
z_log_var = Dense(latent_dim)

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
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

decoder_h_2 = Dense(intermediate_dim, activation='relu')
y_decoded = Dense(10, activation='sigmoid')
h_decoded_2 = decoder_h_2(z)
_y_decoded = y_decoded(h_decoded_2)

yy = Input(batch_shape = (batch_size,10))

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + _z_log_var - K.square(_z_mean) - K.exp(_z_log_var), axis=-1)
    y_loss= 10 * objectives.categorical_crossentropy(yy, _y_decoded)
    return xent_loss + kl_loss + y_loss

my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.1)

vae = Model(inputs = [x,yy], outputs =[x_decoded_mean,_y_decoded]) #(x,x_decoded_mean)
vae.compile(optimizer=my_adam, loss=vae_loss)


# -----------------------------------------------------------------------------
#                                                                   Train Model
# -----------------------------------------------------------------------------

model_weights = pickle.load(open('weights_vaesdr_' + str(latent_dim) + 'd_trained_on_' + dataset_name, 'rb'))
vae.set_weights(model_weights)

vae.fit([x_train, y_train],[x_train,y_train],
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

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], x_test_encoded[:, 2], linewidth = 0, c=y_test)
#ax.colorbar()
#ax.show()

y_test_onehot = utils.to_categorical(y_test, num_classes)


_h_ = h(x)
_z_mean_ = z_mean(_h_)
_decoder_h_ =  decoder_h(_z_mean_)
_decoder_mean_ = decoder_mean(_decoder_h_)
h_decoded_2_ = decoder_h_2(_z_mean_)
_y_decoded_ = y_decoded(h_decoded_2_)

vaeencoder = Model(x,[_decoder_mean_,_y_decoded_])
x_decoded, b  = vaeencoder.predict(x_test,batch_size = batch_size)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)


                                        # -------------------------------------
                                        #                              Accuracy
                                        # -------------------------------------

lll = np.zeros((10000,1))
for i in range(10000):
    m = b[i,:].reshape(10,).tolist()
    lll[i,0] = m.index(max(m))

lll

lll.reshape(1,10000)
lll
lll = lll.reshape(1,10000).astype('uint8')

n_error = np.count_nonzero(lll - y_test)

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
# plt.savefig('figures/'+ dataset_name + '_samples.png')













