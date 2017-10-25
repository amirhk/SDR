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
from sklearn.datasets import fetch_olivetti_faces, load_iris

# import sys
# import platform
# if platform.system() == 'Windows':
#   sys.path.append('..\\..\\utils')
# elif platform.system() == 'Linux' or platform.system() == 'Darwin':
#   sys.path.append('../../utils')

# from generationUtils import generateAndLoadSimilarSamplesToData

def importDatasetForSemisupervisedTraining(dataset_string, number_of_labeled_training_samples, number_of_unlabeled_training_samples):
  if dataset_string == 'mnist':
    fh_import_dataset = lambda : importMnist()
  elif dataset_string == 'mnist-fashion':
    fh_import_dataset = lambda : importMnistFashion()
  elif dataset_string == 'iris':
    fh_import_dataset = lambda : importIris()

  (dataset_name,
    x_train,
    x_test,
    y_train,
    y_test,
    sample_dim,
    sample_channels,
    original_dim,
    num_classes) = fh_import_dataset()

  number_of_training_samples = x_train.shape[0]
  ratio = number_of_unlabeled_training_samples / number_of_labeled_training_samples

  try:
      assert(number_of_labeled_training_samples + number_of_unlabeled_training_samples <= number_of_training_samples)
  except AssertionError:
      print('[ERROR] The total number of samples you\'ve requested is more than the total number of samples in ' + dataset_name + '.')
      raise

  try:
      assert(np.mod(number_of_labeled_training_samples, num_classes) == 0)
  except AssertionError:
      print('[ERROR] number_of_labeled_training_samples must be divisible by num_classes so we sample equally.')
      raise

  try:
      assert(round(ratio) == ratio)
      ratio = int(ratio)
  except AssertionError:
      print('[ERROR] <# unlabeled> should be divisble by <# labeled>.')
      raise

                                        # -------------------------------------
                                        #        Separate out indices per class
                                        # -------------------------------------
  image_indices = {}
  not_hot_y_train = np.argmax(y_train, axis =1)
  for class_number in range(num_classes):
    tmp = np.where(not_hot_y_train == class_number)[0]
    image_indices[class_number] =  np.random.permutation(tmp)
    print('\t[INFO] identified ' + str(image_indices[class_number].shape[0]) + ' samples of class #' + str(class_number))


                                        # -------------------------------------
                                        #                          Placeholders
                                        # -------------------------------------

  tmp_x_train_labeled = np.zeros((number_of_labeled_training_samples, original_dim))
  tmp_y_train_labeled = np.zeros((number_of_labeled_training_samples))
  tmp_x_train_unlabeled = np.zeros((number_of_unlabeled_training_samples, original_dim))
  tmp_y_train_unlabeled = np.zeros((number_of_unlabeled_training_samples))


                                        # -------------------------------------
                                        #                   Get labeled samples
                                        # -------------------------------------

  # tmp_x_train_labeled = x_train[:number_of_labeled_training_samples,:]
  # tmp_y_train_labeled = y_train[:number_of_labeled_training_samples]

  tmp_counter = 0
  start_index_offset = 0
  number_of_labeled_training_samples_from_each_class = int(number_of_labeled_training_samples / num_classes)
  for class_number in range(num_classes):
    selected_sample_indices_from_this_class = image_indices[class_number][start_index_offset : start_index_offset + number_of_labeled_training_samples_from_each_class]

    selected_samples_from_this_class = x_train[selected_sample_indices_from_this_class,:]
    selected_labels_from_this_class = not_hot_y_train[selected_sample_indices_from_this_class]

    assert(selected_samples_from_this_class.shape[0] == number_of_labeled_training_samples_from_each_class)

    start_index = tmp_counter
    end_index = tmp_counter + number_of_labeled_training_samples_from_each_class

    tmp_x_train_labeled[start_index:end_index,:] = selected_samples_from_this_class
    tmp_y_train_labeled[start_index:end_index] = selected_labels_from_this_class

    tmp_counter += number_of_labeled_training_samples_from_each_class


  # generateAndLoadSamplesSimilarTo(X_train, y_train, 'mnist', 1)



  tmp_x_train_labeled = np.tile(tmp_x_train_labeled, (ratio,1))
  tmp_y_train_labeled = np.tile(tmp_y_train_labeled.reshape(number_of_labeled_training_samples,1), (ratio,1))


  assert(tmp_x_train_labeled.shape[0] == tmp_y_train_labeled.shape[0])
  tmp = tmp_x_train_labeled.shape[0]

  # random_ordering = np.random.permutation(tmp)
  # tmp_x_train_labeled = tmp_x_train_labeled[random_ordering,:]
  # tmp_y_train_labeled = tmp_y_train_labeled[random_ordering]

  # random_ordering = np.random.permutation(tmp)
  random_ordering_a = np.random.permutation(number_of_labeled_training_samples)
  random_ordering_b = np.random.permutation(number_of_unlabeled_training_samples - number_of_labeled_training_samples) + number_of_labeled_training_samples
  random_ordering = np.concatenate((random_ordering_a, random_ordering_b), axis=0)
  tmp_x_train_labeled = tmp_x_train_labeled[random_ordering,:]
  tmp_y_train_labeled = tmp_y_train_labeled[random_ordering]


                                        # -------------------------------------
                                        #    DEPRECATED: Get validation samples
                                        # -------------------------------------

  # tmp_x_val = x_train[1000:10000,:]
  # tmp_y_val = y_train[1000:10000]
  tmp_x_val = 'jigar'
  tmp_y_val = 'tala'

                                        # -------------------------------------
                                        #                 Get unlabeled samples
                                        # -------------------------------------

  # tmp_x_train_unlabeled = x_train[-number_of_unlabeled_training_samples:,:]
  # tmp_y_train_unlabeled = y_train[-number_of_unlabeled_training_samples:]

  tmp_counter = 0
  start_index_offset = number_of_labeled_training_samples_from_each_class
  number_of_unlabeled_training_samples_from_each_class = int(number_of_unlabeled_training_samples / num_classes)
  for class_number in range(num_classes):
    selected_sample_indices_from_this_class = image_indices[class_number][start_index_offset : start_index_offset + number_of_unlabeled_training_samples_from_each_class]

    selected_samples_from_this_class = x_train[selected_sample_indices_from_this_class,:]
    selected_labels_from_this_class = not_hot_y_train[selected_sample_indices_from_this_class]

    assert(selected_samples_from_this_class.shape[0] == number_of_unlabeled_training_samples_from_each_class)

    start_index = tmp_counter
    end_index = tmp_counter + number_of_unlabeled_training_samples_from_each_class

    tmp_x_train_unlabeled[start_index:end_index,:] = selected_samples_from_this_class
    tmp_y_train_unlabeled[start_index:end_index] = selected_labels_from_this_class

    tmp_counter += number_of_unlabeled_training_samples_from_each_class


  assert(tmp_x_train_unlabeled.shape[0] == tmp_y_train_unlabeled.shape[0])
  tmp = tmp_x_train_unlabeled.shape[0]

  random_ordering = np.random.permutation(tmp)
  tmp_x_train_unlabeled = tmp_x_train_unlabeled[random_ordering,:]
  tmp_y_train_unlabeled = tmp_y_train_unlabeled[random_ordering]

                                        # -------------------------------------
                                        #                                Return
                                        # -------------------------------------

  tmp_y_train_labeled = utils.to_categorical(tmp_y_train_labeled, num_classes)
  tmp_y_train_unlabeled = utils.to_categorical(tmp_y_train_unlabeled, num_classes)

  return (dataset_string, x_train, x_test, y_train, y_test, sample_dim, sample_channels, original_dim, num_classes, tmp_x_train_labeled, tmp_y_train_labeled, tmp_x_val, tmp_y_val, tmp_x_train_unlabeled, tmp_y_train_unlabeled)




def importMnist():
  print('[INFO] importing mnist-digits...')
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
  y_test = utils.to_categorical(y_test, num_classes)

  print('[INFO] done.')

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

  # other processing
  x_train = x_train.astype('float32') / 255.
  x_test = x_test.astype('float32') / 255.
  x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
  x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

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


















def importGlass():
  dirname, _ = os.path.split(os.path.abspath(__file__))
  data_file_name = os.path.join(dirname, '..', 'data', 'Glass.txt')
  data = np.loadtxt(data_file_name)

  x = data[:,:10]
  y = data[:,10] - 1

  random_ordering = np.random.permutation(x.shape[0])
  x = x[random_ordering,:]
  y = y[random_ordering]

  train_split = 0.70
  split_index = int(np.round(x.shape[0] * train_split))
  split_index = int(25 * round(float(split_index)/25))# rounded to the nearest 25

  x_train = x[:split_index, :]
  x_test = x[split_index:200, :]
  y_train = y[:split_index]
  y_test = y[split_index:200]

  sample_dim = x_train.shape[1]
  sample_channels = -1
  original_dim = sample_dim
  # num_classes = len(np.unique(y))
  num_classes = int(np.max(y)) + 1

  y_train = utils.to_categorical(y_train, num_classes)
  y_test = utils.to_categorical(y_test, num_classes)

  return ('glass', x_train, x_test, y_train, y_test, sample_dim, sample_channels, original_dim, num_classes)




def importBalance():
  dirname, _ = os.path.split(os.path.abspath(__file__))
  data_file_name = os.path.join(dirname, '..', 'data', 'Balance.txt')
  data = np.loadtxt(data_file_name)

  x = data[:,:4]
  y = data[:,4] - 1

  random_ordering = np.random.permutation(x.shape[0])
  x = x[random_ordering,:]
  y = y[random_ordering]

  train_split = 0.70
  split_index = int(np.round(x.shape[0] * train_split))
  split_index = int(25 * round(float(split_index)/25))# rounded to the nearest 25

  x_train = x[:split_index, :]
  x_test = x[split_index:, :]
  y_train = y[:split_index]
  y_test = y[split_index:]

  sample_dim = x_train.shape[1]
  sample_channels = -1
  original_dim = sample_dim
  num_classes = len(np.unique(y))

  y_train = utils.to_categorical(y_train, num_classes)
  y_test = utils.to_categorical(y_test, num_classes)

  return ('balance', x_train, x_test, y_train, y_test, sample_dim, sample_channels, original_dim, num_classes)




def importIris():
  dirname, _ = os.path.split(os.path.abspath(__file__))
  data_file_name = os.path.join(dirname, '..', 'data', 'Iris.txt')
  data = np.loadtxt(data_file_name)

  x = data[:,:4]
  y = data[:,4] - 1

  random_ordering = np.random.permutation(x.shape[0])
  x = x[random_ordering,:]
  y = y[random_ordering]

  train_split = 0.70
  split_index = int(np.round(x.shape[0] * train_split))
  split_index = int(25 * round(float(split_index)/25))# rounded to the nearest 25

  x_train = x[:split_index, :]
  x_test = x[split_index:, :]
  y_train = y[:split_index]
  y_test = y[split_index:]

  sample_dim = x_train.shape[1]
  sample_channels = -1
  original_dim = sample_dim
  num_classes = len(np.unique(y))

  y_train = utils.to_categorical(y_train, num_classes)
  y_test = utils.to_categorical(y_test, num_classes)

  # iris = load_iris()
  # X = iris.data
  # y = iris.target

  # random_ordering = np.random.permutation(x.shape[0])
  # X = X[random_ordering,:]
  # y = y[random_ordering]

  # sample_dim = X.shape[1]
  # sample_channels = -1
  # original_dim = sample_dim
  # num_classes = len(np.unique(y))

  # x_train = X[:125,:]
  # x_test = X[125:,:]

  # y_train = y[:125]
  # y_test = y[125:]

  # y_train = utils.to_categorical(y_train, num_classes)
  # y_test = utils.to_categorical(y_test, num_classes)

  return ('iris', x_train, x_test, y_train, y_test, sample_dim, sample_channels, original_dim, num_classes)








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











