# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:00:13 2017

@author: sbanijam
"""


'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives , utils,optimizers
from keras.datasets import mnist
from mpl_toolkits.mplot3d import Axes3D
import pickle

batch_size = 100
original_dim = 784
latent_dim = 3
intermediate_dim = 500
nb_epoch = 1
epsilon_std = 1.0

learning_rate = 0.00001

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

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_classes = 10

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_train = utils.to_categorical(y_train, num_classes)

model_weights = pickle.load(open('vaesdr', 'rb'))
vae.set_weights(model_weights)







vae.fit([x_train, y_train],[x_train,y_train],
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size)


model_weights = vae.get_weights()
pickle.dump((model_weights), open('vaesdr', 'wb')) 

# build a model to project inputs on the latent space
encoder = Model(x, _z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')   
ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], x_test_encoded[:, 2],linewidth = 0,c=y_test)
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
x_decoded,b  = vaeencoder.predict(x_test,batch_size = batch_size)


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

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))





for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
#        z_sample = np.array([[xi, yi]])
#        z_sample = np.random.randn(1,latent_dim)
        
        digit = x_decoded[i*n + j,:].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()



               
               
figure = np.zeros((digit_size * n, digit_size * n))               
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
#        z_sample = np.array([[xi, yi]])
#        z_sample = np.random.randn(1,latent_dim)
        
        digit = x_test[i*n + j,:].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit               

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()


               
               
figure = np.zeros((digit_size * n, digit_size * n))               
for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
#        z_sample = np.array([[xi, yi]])
#        z_sample = np.random.randn(1,latent_dim)
        x_decoded = generator.predict(np.random.randn(1,latent_dim))
        digit = x_decoded.reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit               

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()



