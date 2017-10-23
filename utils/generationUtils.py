import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
from PIL import Image
import numpy as np
import glob


def generateSamplesSimilarTo(x_data, y_data, dataset_name, number_of_generated_samples):
  # Save augmented images to file
  K.set_image_dim_ordering('th')

  # Reshape to be [samples][channels][width][height]
  if dataset_name == 'mnist' or dataset_name == 'mnist-fashion':
    data = data.reshape(data.shape[0], 1, 28, 28)
  elif dataset_name == 'cifar':
    data = data.reshape(data.shape[0], 3, 32, 32)

  # Convert from int to float
  data = data.astype('float32')

  # Define data preparation
  datagen = ImageDataGenerator()

  # Fit parameters from data
  datagen.fit(data)

  # configure batch size and retrieve one batch of images
  # os.makedirs('images')
  datagen.flow(
    x_data,
    y_data,
    batch_size = number_of_generated_samples,
    save_to_dir = '../shared_images/generated',
    save_prefix = 'aug',
    save_format = 'png'):
  # for X_batch, y_batch in datagen.flow(data, y_train, batch_size = 9, save_to_dir='images', save_prefix='aug', save_format='png'):
  #   # create a grid of 3x3 images
  #   for i in range(0, 9):
  #     pyplot.subplot(330 + 1 + i)
  #     pyplot.imshow(X_batch[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
  #   # show the plot
  #   pyplot.show()
  #   break


def generateAndLoadSimilarSamplesToData(x_data, y_data, dataset_name, number_of_generated_samples):
  generateSamplesSimilarTo(x_data, y_data, dataset_name, number_of_generated_samples)

  image_list = []
  dir_path = 'tmp_images'
  file_format = 'png'
  for filename in glob.glob(dir_path + '/*.' + file_format):
      im=Image.open(filename)
      image_list.append(im)

  assert(len(image_list) == number_of_generated_samples)

  # just to get the dimensions right to create placeholder variables
  one_generated_image = np.asarray(image_list[0])
  sample_dim = np.prod(one_generated_image.shape)

  generated_images_tensor = np.zeros([number_of_generated_samples, sample_dim])

  for i in range(number_of_generated_samples):
    generated_image = np.asarray(image_list[i])
    generated_images_tensor[i,:,:,:] = generated_image.reshape(1, sample_dim)

  return generated_images_tensor





