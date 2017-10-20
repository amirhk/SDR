import matplotlib.pyplot as plt


def saveSamples(dataset_name, x_train, sample_dim):
  tmp = 4

  plt.figure(figsize=(tmp + 1, tmp + 1))
  for i in range(tmp):
    for j in range(tmp):
      ax = plt.subplot(tmp, tmp, i*tmp+j+1)
      # plt.imshow(x_train[i*tmp+j].reshape(sample_dim, sample_dim, sample_channels))
      plt.imshow(x_train[i*tmp+j].reshape(sample_dim, sample_dim))
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)


  plt.savefig('figures/'+ dataset_name + '_samples.png')

