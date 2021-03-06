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
from importDatasets import importDatasetForSemisupervisedTraining

from datetime import datetime



#
#def main(number_of_labeled_training_samples, number_of_unlabeled_training_samples, convex_alpha):
#  # -----------------------------------------------------------------------------
#  #                                                                    Fetch Data
#  # -----------------------------------------------------------------------------
number_of_labeled_training_samples = 100
number_of_unlabeled_training_samples = 10000
convex_alpha = .5
fh_import_dataset = lambda : importDatasetForSemisupervisedTraining('mnist',number_of_labeled_training_samples,number_of_unlabeled_training_samples)
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



synthetic_training,synthetic_training_labels = pickle.load(open('synthetic_labeled_data', 'rb'))


x_train_labeled = synthetic_training
y_train_labeled = synthetic_training_labels
x_total = np.concatenate([x_train_unlabeled,x_train_labeled],1)
x_total_test = np.concatenate([x_test,x_test],1)

batch_size = 100
latent_dim = 15
epochs = 1000
epsilon_std = 1.0
learning_rate = 0.0001
intermediate_recon_layer_dim = 500
intermediate_label_layer_dim = 500

  # -----------------------------------------------------------------------------
  #                                                                  Path-related
  # -----------------------------------------------------------------------------

experiment_name = dataset_name + \
  '_____z_dim_' + str(latent_dim) + \
  '_____num_lab_' + str(number_of_labeled_training_samples) + \
  '_____num_unlab_' + str(number_of_unlabeled_training_samples) + \
  '_____x_loss_mult_' + str(convex_alpha) + \
  '_____y_loss_mult_' + str(1 - convex_alpha)

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

  #def test_layer(args):
  #    ZZZ = args
  #    return ZZZ[:,:,:10]
  #
  #z_test_layer = Lambda(test_layer)(z)

x = Input(batch_shape=(batch_size, 2 * original_dim))

x_reshaped = Reshape((2,original_dim))

h = Dense(intermediate_recon_layer_dim, activation='relu')
z_mean = Dense(latent_dim)
z_log_var = Dense(latent_dim)

_x_reshaped = x_reshaped(x)
_h = h(_x_reshaped)
_z_mean = z_mean(_h)
_z_log_var = z_log_var(_h)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                stddev=epsilon_std)
    return z_mean[:,0,:] + K.exp(z_log_var[:,0,:] / 2) * epsilon

z_u = Lambda(sampling, output_shape=(latent_dim,))([_z_mean, _z_log_var])

def sampling_1(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                stddev=epsilon_std)
    return z_mean[:,1,:] + K.exp(z_log_var[:,1,:] / 2) * epsilon

z_l = Lambda(sampling_1, output_shape=(latent_dim,))([_z_mean, _z_log_var])

def concat_latent(args):
    z_u,z_l = args
    return tf.concat([z_u,z_l],1)

auxiliary_layer = Lambda(concat_latent)([z_u,z_l])
z = Reshape((2,latent_dim))(auxiliary_layer)

  # we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_recon_layer_dim, activation='relu')
decoder_mean_reshaped = Dense(original_dim, activation='sigmoid')
decoded_mean = Reshape((2*original_dim,))
h_decoded = decoder_h(z)
x_decoded_mean_reshaped = decoder_mean_reshaped(h_decoded)
x_decoded_mean = decoded_mean(x_decoded_mean_reshaped)

dropout_class = Dropout(0.5)
decoder_h_2 = Dense(intermediate_label_layer_dim, activation='relu')
y_decoded = Dense(num_classes, activation='sigmoid')
  #h_decoded_2 = decoder_h_2(z_l)
z_l_dropout = dropout_class(z_l)
h_decoded_2 = decoder_h_2(z_l)
_y_decoded = y_decoded(h_decoded_2)

yy = Input(batch_shape = (batch_size,num_classes))

def vae_loss(x, x_decoded_mean):
    x_ent_loss = 2*original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss_u = - 0.5 * K.sum(1 + _z_log_var[:,0,:] - K.square(_z_mean[:,0,:]) - K.exp(_z_log_var[:,0,:]), axis=-1)
    kl_loss_l = - 0.5 * K.sum(1 + _z_log_var[:,1,:] - K.square(_z_mean[:,1,:]) - K.exp(_z_log_var[:,1,:]), axis=-1)
    y_loss= num_classes * objectives.categorical_crossentropy(yy, _y_decoded)
    return x_ent_loss + y_loss + kl_loss_u + kl_loss_l

my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.1)

vae = Model(inputs = [x,yy], outputs =[x_decoded_mean,_y_decoded]) # Model(x,x_decoded_mean)
vae.compile(optimizer=my_adam, loss=vae_loss)

_x_reshaped_ = x_reshaped(x)
_h_ = h(_x_reshaped_)
_z_mean_ = z_mean(_h_)
  # z_mean_test = Lambda(test_layer)(_z_mean)
_decoder_h_ =  decoder_h(_z_mean_)
_decoder_mean_reshaped = decoder_mean_reshaped(_decoder_h_)
_decoder_mean_ = decoded_mean(_decoder_mean_reshaped)

def take_one_dim(args):
    ZZZ = args
    return ZZZ[:,1,:]

z_aux = Lambda(take_one_dim, output_shape=(latent_dim,))(_z_mean_)
h_decoded_2_ = decoder_h_2(z_aux)
_y_decoded_ = y_decoded(h_decoded_2_)

vaeencoder = Model(inputs = x,outputs=  [_decoder_mean_,_y_decoded_])

_, b  = vaeencoder.predict(x_total_test,batch_size = batch_size)

y_test_label = np.argmax(y_test,axis =1)

Accuracy = np.zeros((epochs,1))
ii=0
pickle.dump((ii),open('counter','wb'))
text_file_name = experiment_dir_path + '/accuracy_log.txt'
class ACCURACY(Callback):

    def on_epoch_end(self,batch,logs = {}):
        ii= pickle.load(open('counter', 'rb'))
        _, b  = vaeencoder.predict(x_total_test, batch_size = batch_size)
        Accuracy[ii, 0]

        lll = np.argmax(b, axis =1)
        n_error = np.count_nonzero(lll - y_test_label)
        ACC = 1 - n_error / 10000
        Accuracy[ii,0] = ACC
        print('\n accuracy = ', ACC)
        ii= ii + 1
        pickle.dump((ii),open('counter', 'wb'))
        with open(text_file_name, 'a') as text_file:
          print('Epoch #{} Accuracy:{} \n'.format(ii, ACC), file=text_file)

accuracy = ACCURACY()

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

vae.fit([x_total, y_train_labeled],[x_total,y_train_labeled],#x_total,x_total,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        callbacks = [accuracy])

model_weights = vae.get_weights()
pickle.dump((model_weights), open(weights_file_name, 'wb'))

  # -----------------------------------------------------------------------------
  #                                                                      Analysis
  # -----------------------------------------------------------------------------

  # build a model to project inputs on the latent space
encoder = Model(x, _z_mean)

  # display a 3D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_total_test, batch_size=batch_size)


  #fig = plt.figure()
  #ax = fig.add_subplot(111, projection='3d')
  #ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], x_test_encoded[:, 2], linewidth = 0, c=y_test_label)
  #ax.colorbar()
  #ax.show()



  ## display a 2D plot of the digit classes in the latent space


plt.scatter(x_test_encoded[:, 0,0], x_test_encoded[:, 0,1], linewidth = 0, c=y_test_label)
plt.colorbar()
  # plt.show()
plt.savefig(experiment_dir_path + '/latent_space.png')

  #y_test_onehot = utils.to_categorical(y_test, num_classes)



x_decoded, b  = vaeencoder.predict(x_total_test,batch_size = batch_size)

  # build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean_reshaped = decoder_mean_reshaped(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean_reshaped)


                                          # -------------------------------------
                                          #                               Fit GMM
                                          # -------------------------------------

  # display a 2D plot of the digit classes in the latent space
x_train_encoded = encoder.predict(x_total, batch_size=batch_size)

n_components = num_classes
cv_type = 'full'
gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
gmm.fit(x_train_encoded[:,0,:])



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


number_of_sample_images = num_classes
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
x_samples_b = x_decoded[:,:784]
canvas = getFigureOfSamplesForInput(x_samples_b, sample_dim, number_of_sample_images, grid_x, grid_y)
plt.imshow(canvas, cmap='Greys_r')
ax.set_title('Reconstructed Test Images', fontsize=8)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(1,3,3)
  #x_samples_c = gmm.sample(10000)
x_samples_c = gmm.sample(number_of_sample_images ** 2)
x_samples_c = x_samples_c[0]
  #x_samples_c = np.random.permutation(x_samples_c) # need to randomly permute because gmm.sample samples 1000 from class 1, then 1000 from class 2, etc.
x_samples_c = generator.predict(x_samples_c)
canvas = getFigureOfSamplesForInput(x_samples_c, sample_dim, number_of_sample_images, grid_x, grid_y)
plt.imshow(canvas, cmap='Greys_r')
ax.set_title('Generated Images', fontsize=8)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

  # plt.show()
plt.savefig(experiment_dir_path + '/generated_samples.png')

  # for i in range(100):
  #   plt.figure()
  #   ax = plt.subplot(1,1,1)
  #   single_sample = gmm.sample(1)
  #   single_sample = single_sample[0]
  #   # single_sample = np.random.permutation(single_sample) # need to randomly permute because gmm.sample samples 1000 from class 1, then 1000 from class 2, etc.
  #   single_sample = generator.predict(single_sample)
  #   plt.imshow(single_sample.reshape(sample_dim, sample_dim), cmap='Greys_r')
  #   ax.set_title('Generated Images', fontsize=8)
  #   ax.get_xaxis().set_visible(False)
  #   ax.get_yaxis().set_visible(False)
  #   plt.savefig('images/tmp/'+ dataset_name + '_sample_' + str(i+1) + '.png')



#if __name__ == "__main__":
#
#  # convex_alpha_list = [0, 0.25, 0.50, 0.75, 1]
#  # number_of_labeled_training_samples_list = [20, 50, 100, 500, 1000]
#  # number_of_unlabeled_training_samples = 10000
#
#  # number_of_labeled_training_samples_list = [20, 50, 100, 500, 1000]
#
#  # for number_of_labeled_training_samples in number_of_labeled_training_samples_list:
#
#  #   for convex_alpha in convex_alpha_list:
#
#  #     main(number_of_labeled_training_samples, number_of_unlabeled_training_samples, convex_alpha)
#
#  main(1000, 10000, 0.5)




################ build synthetic labeled points #######################

ratio = int( number_of_unlabeled_training_samples/number_of_labeled_training_samples )
  
synthetic_training = np.zeros((number_of_unlabeled_training_samples,original_dim))

synthetic_training[:number_of_labeled_training_samples,:] = x_train_labeled[:number_of_labeled_training_samples,:]
synthetic_training_labels = np.tile(y_train_labeled[:number_of_labeled_training_samples],(ratio,1))
                                  
for i in range(ratio-1):
    aux_var,_ = vae.predict([x_total[:number_of_labeled_training_samples,:],y_train_labeled[:number_of_labeled_training_samples,:]],batch_size = batch_size)
    synthetic_training[(i+1)*number_of_labeled_training_samples:(i+2)*number_of_labeled_training_samples,:] = aux_var[:,original_dim:]


random_ordering = np.random.permutation(number_of_unlabeled_training_samples)
synthetic_training = synthetic_training[random_ordering,:]
synthetic_training_labels = synthetic_training_labels[random_ordering,:]    
pickle.dump((synthetic_training,synthetic_training_labels),open('synthetic_labeled_data','wb'))
