# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:00:13 2017

@author: sbanijam
"""


'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives, utils


# from keras.callbacks import LambdaCallback

batch_size = 100
latent_dim = 2
intermediate_dim = 500
epochs = 100 # 20
epsilon_std = 1.0





# train the VAE on MNIST digits
# from keras.datasets import mnist
# sample_dim = 28
# sample_channels = 1
# original_dim = sample_channels * sample_dim ** 2
# num_classes = 10
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # train the VAE on MNIST digits
# from keras.datasets import cifar10
# sample_dim = 32
# sample_channels = 3
# original_dim = sample_channels * (sample_dim ** 2)
# num_classes = 10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()




# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# y_train = utils.to_categorical(y_train, num_classes)






# # train a simple AE on Olivetti faces objects
import sklearn
from sklearn.datasets import fetch_olivetti_faces
sample_dim = 64
sample_channels = 1
original_dim = sample_channels * (sample_dim ** 2)
num_classes = 40
a = fetch_olivetti_faces()
random_ordering = np.random.permutation(400)
training_indices = random_ordering[:300]
testing_indices = random_ordering[300:]
x_train = a.data[training_indices]
x_test = a.data[testing_indices]
y_train = a.target[training_indices]
y_test = a.target[testing_indices]

y_train = utils.to_categorical(y_train, num_classes)





# # train the VAE on PLANE-AHMAD objects
# sample_dim = 40
# sample_channels = 1
# original_dim = sample_channels * sample_dim ** 2
# num_classes = 2
# x_train, y_train = pickle.load(open('plane_dataset_train', 'rb'))
# x_train = x_train.reshape([x_train.shape[0], -1])
# x_test, y_test = pickle.load(open('plane_dataset_test', 'rb'))
# x_test = x_test.reshape([x_test.shape[0], -1])
# y_test = [ np.where(tmp==1)[0][0] for tmp in y_test ]



tmp = 3

plt.figure(figsize=(tmp + 1, tmp + 1))
for i in range(tmp):
  for j in range(tmp):
    ax = plt.subplot(tmp, tmp, i*tmp+j+1)
    # plt.imshow(x_train[i*tmp+j].reshape(sample_dim, sample_dim, sample_channels))
    plt.imshow(x_train[i*tmp+j].reshape(sample_dim, sample_dim))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.savefig('figures/samples.png')















x = Input(batch_shape=(batch_size, original_dim))
y = Input(batch_shape = (batch_size,num_classes))
h = Dense(intermediate_dim, activation='relu')(x)
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
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
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
    return x_ent_loss + kl_loss + y_loss
    # return y_loss
    # return 1


vae = Model(inputs = [x, y], outputs =[x_decoded_mean, _y_decoded]) #(x,x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)
# vae.compile(optimizer='Adam', loss=vae_loss)
# vae.compile(optimizer='SGD', loss=vae_loss)












# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)













# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
# plt.show()
plt.savefig('figures/latent_space_' + str(0) + '.png')


# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits

# figure = np.zeros((sample_dim * n, sample_dim * n, sample_channels))
figure = np.zeros((sample_dim * n, sample_dim * n))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        # z_sample = np.random.randn(1,latent_dim)
        x_decoded = generator.predict(z_sample)
        # digit = x_decoded[0].reshape(sample_dim, sample_dim, sample_channels)
        digit = x_decoded[0].reshape(sample_dim, sample_dim)
        figure[i * sample_dim: (i + 1) * sample_dim,
               j * sample_dim: (j + 1) * sample_dim] = digit


plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
# plt.show()
plt.savefig('figures/manifold_' + str(0) + '.png')








# batch_print_callback = LambdaCallback(
#     on_batch_begin=lambda batch,logs: print(batch))



save_interval = 20
assert(epochs % save_interval == 0)

for kk in range(int(epochs / save_interval)):

# kk = 99
# save_interval = 99

    vae.fit([x_train, y_train],[x_train, y_train],
            shuffle=True,
            epochs=save_interval,
            batch_size=batch_size) # ,
            # callbacks=[batch_print_callback])

    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    # plt.show()
    plt.savefig('figures/latent_space_' + str((kk+1)*save_interval) + '.png')


    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits

    # figure = np.zeros((sample_dim * n, sample_dim * n, sample_channels))
    figure = np.zeros((sample_dim * n, sample_dim * n))
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            # z_sample = np.random.randn(1,latent_dim)
            x_decoded = generator.predict(z_sample)
            # digit = x_decoded[0].reshape(sample_dim, sample_dim, sample_channels)
            digit = x_decoded[0].reshape(sample_dim, sample_dim)
            figure[i * sample_dim: (i + 1) * sample_dim,
                   j * sample_dim: (j + 1) * sample_dim] = digit


    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    # plt.show()
    plt.savefig('figures/manifold_' + str((kk+1)*save_interval) + '.png')


