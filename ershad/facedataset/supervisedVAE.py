# -------------------------------------------------------------------------
# Copyright (c) 2017, Amir-Hossein Karimi
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

from keras import backend as K
from keras import objectives, utils, optimizers
from keras.layers import Input, Dense, Lambda
from keras.models import Model


from scipy.stats import norm

from importDatasets import importMnist
from importDatasets import importOlivetti
from importDatasets import importSquareAndCross

#from saveUtils import saveSamples



# -----------------------------------------------------------------------------
#                                                                    Fetch Data
# -----------------------------------------------------------------------------

# fh_import_dataset = lambda : importMnist()
# fh_import_dataset = lambda : importOlivetti()
# fh_import_dataset = lambda : importOlivetti('glasses_labels')
# fh_import_dataset = lambda : importOlivetti('beard_labels')
fh_import_dataset = lambda : importOlivetti('glasses_and_beard_labels')
# fh_import_dataset = lambda : importSquareAndCross()

(dataset_name,
  x_train,
  x_test,
  y_train,
  y_test,
  sample_dim,
  sample_channels,
  original_dim,
  num_classes) = fh_import_dataset()

batch_size = 50
latent_dim = 10
# latent_dim = 2
epochs = 500
# intermediate_dim = 500
intermediate_dim = 300
epsilon_std = 1.0
learning_rate = 0.000005

# saveSamples(dataset_name, x_train, sample_dim)

# -----------------------------------------------------------------------------
#                                                                   Build Model
# -----------------------------------------------------------------------------

x = Input(batch_shape=(batch_size, original_dim))
y = Input(batch_shape = (batch_size, num_classes))
h = Dense(intermediate_dim, activation='relu')(x)
#h_1 = Dense(intermediate_dim, activation='relu')(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
#decoder_h_1 = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
#h_decoded_1 = decoder_h_1(h_decoded)
x_decoded_mean = decoder_mean(h_decoded)

decoder_h_2 = Dense(intermediate_dim, activation='relu')
y_decoded = Dense(num_classes, activation='sigmoid')
h_decoded_2 = decoder_h_2(z)
_y_decoded = y_decoded(h_decoded_2)


def vae_loss(x, x_decoded_mean):
    x_ent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    # y_loss= 10 * objectives.categorical_crossentropy(y, _y_decoded)
    y_loss= num_classes * objectives.categorical_crossentropy(y, _y_decoded)
    # return x_ent_loss + kl_loss
    # print('reconstruction error: ' + str(x_ent_loss))
    # print('classification error: ' + str(y_loss))
    return x_ent_loss + kl_loss  + y_loss
    # return y_loss
    # return 1


    

my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.1)



vae = Model(inputs = [x, y], outputs =[x_decoded_mean, _y_decoded]) #(x,x_decoded_mean)
vae.compile(optimizer=my_adam, loss=vae_loss)
# vae.compile(optimizer='Adam', loss=vae_loss)
# vae.compile(optimizer='SGD', loss=vae_loss)


# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)













# # display a 2D plot of the digit classes in the latent space
# x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
# plt.colorbar()
# # plt.show()
# plt.savefig('figures/latent_space_' + str(0) + '.png')


# # display a 2D manifold of the digits
# n = 15  # figure with 15x15 digits

# # figure = np.zeros((sample_dim * n, sample_dim * n, sample_channels))
# figure = np.zeros((sample_dim * n, sample_dim * n))
# # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# # to produce values of the latent variables z, since the prior of the latent space is Gaussian
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi]])
#         # z_sample = np.random.randn(1,latent_dim)
#         x_decoded = generator.predict(z_sample)
#         # digit = x_decoded[0].reshape(sample_dim, sample_dim, sample_channels)
#         digit = x_decoded[0].reshape(sample_dim, sample_dim)
#         figure[i * sample_dim: (i + 1) * sample_dim,
#                j * sample_dim: (j + 1) * sample_dim] = digit


# plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap='Greys_r')
# # plt.show()
# plt.savefig('figures/manifold_' + str(0) + '.png')








# batch_print_callback = LambdaCallback(
#     on_batch_begin=lambda batch,logs: print(batch))








# kk = 99
# save_interval = 200


model_weights = pickle.load(open('vaesdr', 'rb'))
vae.set_weights(model_weights)

vae.fit([x_train, y_train],[x_train, y_train],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size) # ,
            # callbacks=[batch_print_callback])
    

a,b = vae.predict([x_train,y_train],batch_size = batch_size)
imshow(a[1,:].reshape((64,64)),cmap = 'Greys_r') 

plt.figure(figsize=(6, 6))
imshow(x_train[1,:].reshape((64,64)),cmap = 'Greys_r') 

model_weights = vae.get_weights()
pickle.dump((model_weights), open('vaesdr', 'wb'))   
#
#    # display a 2D plot of the digit classes in the latent space
##    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
##    plt.figure(figsize=(6, 6))
##    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
##    plt.colorbar()
#    # plt.show()
#    plt.savefig('figures/latent_space_' + dataset_name + '_epoch_' + str((kk+1)*save_interval) + '.png')
#
#
#    # display a 2D manifold of the digits
#    n = 15  # figure with 15x15 digits
#
#    # figure = np.zeros((sample_dim * n, sample_dim * n, sample_channels))
#    figure = np.zeros((sample_dim * n, sample_dim * n))
#    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
#    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
#    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
#    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
#
#    for i, yi in enumerate(grid_x):
#        for j, xi in enumerate(grid_y):
#            z_sample = np.array([[xi, yi]])
#            # z_sample = np.random.randn(1,latent_dim)
#            x_decoded = generator.predict(z_sample)
#            # digit = x_decoded[0].reshape(sample_dim, sample_dim, sample_channels)
#            digit = x_decoded[0].reshape(sample_dim, sample_dim)
#            figure[i * sample_dim: (i + 1) * sample_dim,
#                   j * sample_dim: (j + 1) * sample_dim] = digit
#
#
#    plt.figure(figsize=(10, 10))
#    plt.imshow(figure, cmap='Greys_r')
#    # plt.show()
#    plt.savefig('figures/manifold_' + dataset_name + '_epoch_' + str((kk+1)*save_interval) + '.png')
#
#
