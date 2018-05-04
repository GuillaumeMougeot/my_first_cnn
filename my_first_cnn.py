"""
Date: 03/05/2018
Author: Guillaume Mougeot
"""

import sys
import numpy as np
import pickle
from PIL import Image

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
# batch_len = 3000
# indices = np.random.choice(data_sets['images_train'].shape[0], batch_len)
# batch_xs = data_sets['images_train'][indices]
# batch_ys = data_sets['labels_train'][indices]
# print(len(batch_xs[0]))


# ConvNet structure
# 3 stages:
#  4x17x17 Conv stage
#  16x9x9  Conv stage
#  64x5x5  Conv stage
#  Normalize+Summation stage

# Random initialization
c1 = [np.random.random((3,3)) for i in range(4)]
# Normalization
for i in range(4):
    c1[i] = c1[i] / sum(sum(c1[i]))

c2 = [np.random.random((9,9)) for i in range(16)]
# Normalization
for i in range(16):
    c2[i] = c2[i] / sum(sum(c2[i]))

c3 = [np.random.random((5,5)) for i in range(64)]
# Normalization
for i in range(64):
    c3[i] = c3[i] / sum(sum(c3[i]))

def convolution(image, conv):
    # Image is suppose to be an array
    size_image = image.shape
    size_conv = conv.shape
    size_output = (size_image[0]-size_conv[0]+1, size_image[1]-size_conv[1]+1)
    output = np.zeros(size_output)

    for i in range(0, size_output[0]):
        for j in range(0, size_output[1]):
            for k in range(size_conv[0]):
                for l in range(size_conv[1]):
                    output[i][j] += image[i + k][j + l] * conv[k][l]
    # np.zeros is unfortunatly an array of float64 so we have to convert it
    # into an uint8 array 
    output = output.astype(np.uint8)
    return output

def convertImage(l,w,h):
    data = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            for k in range(3):
                data[i][j][k] = l[i*h+j+k*h*w]
    return data, Image.fromarray(data, 'RGB')

index = 801
matrix, img = convertImage(data_sets['images_train'][index],32,32)
print(matrix[:,:,0].shape)
print(c1[0].shape)
conv = Image.fromarray(c1[0], 'L')
conv.save('conv_index.png')

img_conv = Image.fromarray(convolution(matrix[:,:,0], c1[0]), 'L');
img_conv.save('conv.png')
print(data_sets['classes'][data_sets['labels_train'][index]])
img.save('img.png')
