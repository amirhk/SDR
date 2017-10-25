
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn import mixture

from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D,UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras import backend as K
from keras import objectives , utils,optimizers
from keras.datasets import mnist
from mpl_toolkits.mplot3d import Axes3D
from keras.callbacks import Callback
import tensorflow as tf

import sys
sys.path.append('../../utils')

from importDatasets import importMnist
from importDatasets import importMnistFashion
from importDatasets import importOlivetti
from importDatasets import importSquareAndCross


from datetime import datetime
# -----------------------------------------------------------------------------
#                                                                    Fetch Data
# -----------------------------------------------------------------------------

# fh_import_dataset = lambda : importMnist()
fh_import_dataset = lambda : importMnistFashion()

(dataset_name,
  x_train,
  x_test,
  y_train,
  y_test,
  sample_dim,
  sample_channels,
  original_dim,
  num_classes) = fh_import_dataset()

training_size = 55000
x_val = x_train[training_size:,:]
y_val = y_train[training_size:,:]
x_train =x_train[:training_size,:] #np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = x_test[:training_size,:] #np.reshape(x_test, (len(x_test), 28, 28, 1))
y_train = y_train[:training_size,:]
y_test = y_test[:training_size,:]



batch_size = 100
latent_dim = 15
epochs = 50
intermediate_dim = 500
epsilon_std = 1.0
learning_rate = 0.0005


# -----------------------------------------------------------------------------
#                                                                   Build Model
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


########## Network Layers ########################################################
x = Input(batch_shape=(batch_size, original_dim))
x_reshaped = Reshape((28,28,1))
h_e_1 = Conv2D(16, (3, 3), activation='relu', padding='same')
h_e_2 = MaxPooling2D((2, 2), padding='same')
h_e_3 = Conv2D(8, (3, 3), activation='relu', padding='same')
h_e_4 = MaxPooling2D((2, 2), padding='same')
h_e_5 = Conv2D(8, (3, 3), activation='relu', padding='same')
h_e_6 = MaxPooling2D((2, 2), padding='same')
h_e_7 = Flatten()

z_mean = Dense(latent_dim)
z_log_var = Dense(latent_dim)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

h_d_x_1 = Dense(4*4*8, activation = 'relu')
h_d_x_2 = Reshape((4,4,8))
h_d_x_3 = Conv2D(8, (3, 3), activation='relu', padding='same')
h_d_x_4 = UpSampling2D((2, 2))
h_d_x_5 = Conv2D(8, (3, 3), activation='relu', padding='same')
h_d_x_6 = UpSampling2D((2, 2))
h_d_x_7 = Conv2D(16, (3, 3), activation='relu')
h_d_x_8 = UpSampling2D((2, 2))
x_decoded_reshaped = Conv2D(1, (3, 3), activation='sigmoid', padding='same')
x_decoded = Flatten()

h_d_y_1 = Dense(intermediate_dim, activation='relu')
h_d_y_2 = Dense(intermediate_dim, activation='relu')
y_decoded = Dense(10, activation='sigmoid')

yy = Input(batch_shape = (batch_size,10))
##### Build model #########################################################################################
_x_reshaped = x_reshaped(x)
_h_e_1 = h_e_1(_x_reshaped)
_h_e_2 = h_e_2(_h_e_1)
_h_e_3 = h_e_3(_h_e_2)
_h_e_4 = h_e_4(_h_e_3)
_h_e_5 = h_e_5(_h_e_4)
_h_e_6 = h_e_6(_h_e_5)
_h_e_7 = h_e_7(_h_e_6)


_z_mean = z_mean(_h_e_7)
_z_log_var = z_log_var(_h_e_7)
z = Lambda(sampling, output_shape=(latent_dim,))([_z_mean, _z_log_var])

_h_d_x_1 = h_d_x_1(z)
_h_d_x_2 = h_d_x_2(_h_d_x_1)
_h_d_x_3 = h_d_x_3(_h_d_x_2)
_h_d_x_4 = h_d_x_4(_h_d_x_3)
_h_d_x_5 = h_d_x_5(_h_d_x_4)
_h_d_x_6 = h_d_x_6(_h_d_x_5)
_h_d_x_7 = h_d_x_7(_h_d_x_6)
_h_d_x_8 = h_d_x_8(_h_d_x_7)
_x_decoded_reshaped = x_decoded_reshaped(_h_d_x_8)
_x_decoded = x_decoded(_x_decoded_reshaped)


_h_d_y_1 = h_d_y_1(z)
_h_d_y_2 = h_d_y_2(_h_d_y_1)
_y_decoded = y_decoded(_h_d_y_2)

###### Define Loss ################################################################################

def vae_loss(x, _x_decoded):

    xent_loss = original_dim * objectives.binary_crossentropy(x, _x_decoded)
    kl_loss = - 0.5 * K.sum(1 + _z_log_var - K.square(_z_mean) - K.exp(_z_log_var), axis=-1)
    y_loss= 10 * objectives.categorical_crossentropy(yy, _y_decoded)
    return xent_loss + kl_loss + y_loss

my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.1)

vae = Model(inputs = [x,yy], outputs =[_x_decoded,_y_decoded]) #(x,_x_decoded)
vae.compile(optimizer=my_adam, loss=vae_loss)


# -----------------------------------------------------------------------------
#                                                                   Train Model
# -----------------------------------------------------------------------------
#### Build another model####################################################
_x_reshaped_ = x_reshaped(x)
_h_e_1_ = h_e_1(_x_reshaped_)
_h_e_2_ = h_e_2(_h_e_1_)
_h_e_3_ = h_e_3(_h_e_2_)
_h_e_4_ = h_e_4(_h_e_3_)
_h_e_5_ = h_e_5(_h_e_4_)
_h_e_6_ = h_e_6(_h_e_5_)
_h_e_7_ = h_e_7(_h_e_6_)


_z_mean_ = z_mean(_h_e_7_)


_h_d_x_1_ = h_d_x_1(_z_mean_)
_h_d_x_2_ = h_d_x_2(_h_d_x_1_)
_h_d_x_3_ = h_d_x_3(_h_d_x_2_)
_h_d_x_4_ = h_d_x_4(_h_d_x_3_)
_h_d_x_5_ = h_d_x_5(_h_d_x_4_)
_h_d_x_6_ = h_d_x_6(_h_d_x_5_)
_h_d_x_7_ = h_d_x_7(_h_d_x_6_)
_h_d_x_8_ = h_d_x_8(_h_d_x_7_)
_x_decoded_reshaped_ = x_decoded_reshaped(_h_d_x_8_)
_x_decoded_ = x_decoded(_x_decoded_reshaped_)

_h_d_y_1_ = h_d_y_1(_z_mean_)
_h_d_y_2_ = h_d_y_2(_h_d_y_1_)
_y_decoded_ = y_decoded(_h_d_y_2_)


vaeencoder = Model(x,[_x_decoded_,_y_decoded_])
#####################################################################################

_, b  = vaeencoder.predict(x_test,batch_size = batch_size)

y_test_label = np.argmax(y_test,axis =1)

Accuracy = np.zeros((epochs,1))
ii=0
pickle.dump((ii),open('counter','wb'))
text_file_name = experiment_dir_path + '/accuracy_log.txt'
class ACCURACY(Callback):

    def on_epoch_end(self,batch,logs = {}):
        ii= pickle.load(open('counter', 'rb'))
        _, b  = vaeencoder.predict(x_test, batch_size = batch_size)
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

# model_weights = pickle.load(open('weights_vaesdr_' + str(latent_dim) + 'd_trained_on_' + dataset_name, 'rb'))
# vae.set_weights(model_weights)

vae.fit([x_train, y_train],[x_train,y_train],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data =([x_val,y_val],[x_val,y_val]),
        callbacks = [accuracy])

model_weights = vae.get_weights()
pickle.dump((model_weights), open('weights_vaesdr_' + str(latent_dim) + 'd_trained_on_' + dataset_name, 'wb'))
############################################################################################################

# -----------------------------------------------------------------------------
#                                                                      Analysis
# -----------------------------------------------------------------------------

###### Builder Encoder ######################################################################
encoder = Model(x, _z_mean)

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], linewidth = 0, c=y_test_label)

#### build generator #########################################################################
generator_input = Input(shape=(latent_dim,))

_h_g_x_1_ = h_d_x_1(generator_input)
_h_g_x_2_ = h_d_x_2(_h_g_x_1_)
_h_g_x_3_ = h_d_x_3(_h_g_x_2_)
_h_g_x_4_ = h_d_x_4(_h_g_x_3_)
_h_g_x_5_ = h_d_x_5(_h_g_x_4_)
_h_g_x_6_ = h_d_x_6(_h_g_x_5_)
_h_g_x_7_ = h_d_x_7(_h_g_x_6_)
_h_g_x_8_ = h_d_x_8(_h_g_x_7_)
_x_generated_reshaped = x_decoded_reshaped(_h_g_x_8_)
_x_generated_ = x_decoded(_x_generated_reshaped)

generator = Model(generator_input,_x_generated_)
                                        # -------------------------------------
                                        #                               Fit GMM
                                        # -------------------------------------

# display a 2D plot of the digit classes in the latent space
x_train_encoded = encoder.predict(x_train, batch_size=batch_size)

n_components = num_classes
cv_type = 'full'
gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
gmm.fit(x_train_encoded)

x_decoded, b  = vaeencoder.predict(x_test,batch_size = batch_size)


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













