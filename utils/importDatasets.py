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

import numpy as np
import sklearn
import pickle
import os
import sys

from keras import utils
from keras.datasets import mnist
from sklearn.datasets import fetch_olivetti_faces


def importMnist():
  # meta
  sample_dim = 28
  sample_channels = 1
  original_dim = sample_channels * sample_dim ** 2
  num_classes = 10

  # import
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # other processing
  x_train = x_train.astype('float32') / 255.
  x_test = x_test.astype('float32') / 255.
  x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
  x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

  y_train = utils.to_categorical(y_train, num_classes)
  y_test = utils.to_categorical(y_train, num_classes)

  return ('mnist', x_train, x_test, y_train, y_test, sample_dim, sample_channels, original_dim, num_classes)




def importMnistFashion():
  print('[INFO] importing mnist-fashion...')
  # meta
  sample_dim = 28
  sample_channels = 1
  original_dim = sample_channels * sample_dim ** 2
  num_classes = 10

  # print(os.path.join(os.path.dirname(sys.argv[0]), '../data/fashionmnist/fashion-mnist_train.csv'))
  
  dirname, _ = os.path.split(os.path.abspath(__file__))
  train_file_name = os.path.join(dirname, '..', 'data', 'fashionmnist', 'fashion-mnist_train.csv')
  test_file_name = os.path.join(dirname, '..', 'data', 'fashionmnist', 'fashion-mnist_test.csv')

  
  # train_file_name = os.path.join(os.path.dirname(sys.argv[0]), '..\\data\\fashionmnist\\fashion-mnist_train.csv')
  # test_file_name = os.path.join(os.path.dirname(sys.argv[0]), '../data/fashionmnist/fashion-mnist_test.csv')

  train_meta_data = np.genfromtxt(train_file_name, delimiter=',')
  test_meta_data = np.genfromtxt(test_file_name, delimiter=',')

  x_train = train_meta_data[:,1:]
  x_test = test_meta_data[:,1:]

  y_train = train_meta_data[:,0]
  y_test = test_meta_data[:,0]

  y_train = utils.to_categorical(y_train, num_classes)
  y_test = utils.to_categorical(y_test, num_classes)

  print('[INFO] done.')
  return ('mnist-fashion', x_train, x_test, y_train, y_test, sample_dim, sample_channels, original_dim, num_classes)




def importSquareAndCross():
  # meta
  sample_dim = 40
  sample_channels = 1
  original_dim = sample_channels * sample_dim ** 2
  num_classes = 2

  # import
  x_train, y_train = pickle.load(open('plane_dataset_train', 'rb'))
  x_test, y_test = pickle.load(open('plane_dataset_test', 'rb'))


  # other processing
  x_train = x_train.reshape([x_train.shape[0], -1])
  x_test = x_test.reshape([x_test.shape[0], -1])
  y_test = [ np.where(tmp==1)[0][0] for tmp in y_test ]

  return ('square_and_cross', x_train, x_test, y_train, y_test, sample_dim, sample_channels, original_dim, num_classes)



















def importOlivetti(label_string):
  # meta
  sample_dim = 64
  sample_channels = 1
  original_dim = sample_channels * (sample_dim ** 2)


  # import
  tmp = fetch_olivetti_faces()

  all_data = tmp.data


  if label_string == 'original_labels':
    num_classes = 40
    all_labels = tmp.target
  elif label_string == 'glasses_labels':
    num_classes = 2
    all_labels = getOlivettiGlassesLabels()
  elif label_string == 'beard_labels':
    num_classes = 2
    all_labels = getOlivettiBeardLabels()
  elif label_string == 'glasses_and_beard_labels':

    glasses_labels = getOlivettiGlassesLabels()
    beard_labels = getOlivettiBeardLabels()
    assert(len(glasses_labels) == len(beard_labels))
    all_labels = [0 for i in range(len(beard_labels))]

    # num_classes = 4
    # for i in range(len(all_labels)):
    #   if glasses_labels[i] == 0 and beard_labels[i] == 0:
    #     all_labels[i] = 0
    #   elif glasses_labels[i] == 0 and beard_labels[i] == 1:
    #     all_labels[i] = 1
    #   elif glasses_labels[i] == 1 and beard_labels[i] == 0:
    #     all_labels[i] = 2
    #   elif glasses_labels[i] == 1 and beard_labels[i] == 1:
    #     all_labels[i] = 3

    num_classes = 2
    for i in range(len(all_labels)):
      all_labels[i] = [0, 0]
      if glasses_labels[i] == 1:
        all_labels[i][0] = 1
      if beard_labels[i] == 1:
        all_labels[i][1] = 1

    all_labels = np.array(all_labels)

  # other processing
  random_ordering = np.random.permutation(400)
  # random_ordering = list(range(0,400))
  training_indices = random_ordering[:300]
  testing_indices = random_ordering[300:]


  x_train = all_data[training_indices]
  x_test = all_data[testing_indices]
  y_train = all_labels[training_indices]
  y_test = all_labels[testing_indices]


  # x_train = all_data[training_indices]
  # x_test = all_data[testing_indices]
  # y_train = all_labels[training_indices]
  # y_test = all_labels[testing_indices]

  # y_train = utils.to_categorical(y_train, num_classes)

  return ('olivetti', x_train, x_test, y_train, y_test, sample_dim, sample_channels, original_dim, num_classes)




def getOlivettiGlassesLabels():
  # glasses
  return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def getOlivettiBeardLabels():
  # facial hair
  return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])











