from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import pickle

from keras import objectives, optimizers, callbacks, utils

# this is the size of our encoded representations
encoding_dim = 100  # 32 floats -> compression of factor 24.5, assuming the input is original_dim floats




# # train a simple AE on MNIST digits
# from keras.datasets import mnist
# sample_dim = 28
# sample_channels = 1
# original_dim = sample_channels * sample_dim ** 2
# num_classes = 10
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # train a simple AE on MNIST digits
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
a = fetch_olivetti_faces()
x_train = a.data[:350]
x_test = a.data[350:]



# # train a simple AE on PLANE-AHMAD objects
# sample_dim = 40
# sample_channels = 1
# original_dim = sample_channels * sample_dim ** 2
# num_classes = 2
# x_train, y_train = pickle.load(open('plane_dataset_train', 'rb'))
# x_train = x_train.reshape([x_train.shape[0], -1])
# x_test, y_test = pickle.load(open('plane_dataset_test', 'rb'))
# x_test = x_test.reshape([x_test.shape[0], -1])
# y_test = [ np.where(tmp==1)[0][0] for tmp in y_test ]










# this is our input placeholder
input_img = Input(shape=(original_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(original_dim, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
encoder_decoder_lr = 0.001
encoder_decoder_optimizer = optimizers.Adam(lr=encoder_decoder_lr, beta_1=0.1)
autoencoder.compile(encoder_decoder_optimizer, loss='mean_squared_error')

# from keras.datasets import mnist
# (x_train, _), (x_test, _) = mnist.load_data()

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train.shape)
# print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(sample_dim, sample_dim))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(sample_dim, sample_dim))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
