"""
Date: 03/05/2018
Author: Guillaume Mougeot
"""

import sys
import numpy as np
import pickle


def load_CIFAR10_batch(filename):
  '''load data from single CIFAR-10 file'''

  with open(filename, 'rb') as f:
    if sys.version_info[0] < 3:
      dict = pickle.load(f)
    else:
      dict = pickle.load(f, encoding='latin1')
    x = dict['data']
    y = dict['labels']
    x = x.astype(float)
    y = np.array(y)
  return x, y


def load_data():
  '''load all CIFAR-10 data and merge training batches'''

  xs = []
  ys = []
  for i in range(1, 6):
    filename = 'cifar-10-python/data_batch_' + str(i)
    X, Y = load_CIFAR10_batch(filename)
    xs.append(X)
    ys.append(Y)

  x_train = np.concatenate(xs)
  y_train = np.concatenate(ys)
  del xs, ys

  x_test, y_test = load_CIFAR10_batch('cifar-10-python/test_batch')

  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck']

  # Normalize Data
  #mean_image = np.mean(x_train, axis=0)
  #x_train -= mean_image
  #x_test -= mean_image

  data_dict = {
    'images_train': x_train,
    'labels_train': y_train,
    'images_test': x_test,
    'labels_test': y_test,
    'classes': classes
  }
  return data_dict


# Load the data
data_sets = load_data()

# Define batches
batch_len = 3000
indices = np.random.choice(data_sets['images_train'].shape[0], batch_len)
batch_xs = data_sets['images_train'][indices]
batch_ys = data_sets['labels_train'][indices]
print(len(batch_xs[0]))
