import re
import os
import numpy as np
from glob import glob
from net.label import Label


def create_continuous_noise(num_continuous, style_size, size):
  continuous = np.random.uniform(-1.0, 1.0, size=(size, num_continuous))
  style = np.random.standard_normal(size=(size, style_size))
  return np.hstack([continuous, style])


def create_categorical_noise(categorical_cardinality, size):
  noise = []
  for cardinality in categorical_cardinality:
    noise.append(
      np.random.randint(0, cardinality, size=size)
    )
  return noise


def encode_infogan_noise(categorical_cardinality, categorical_samples, continuous_samples):
  noise = []
  for cardinality, sample in zip(categorical_cardinality, categorical_samples):
    noise.append(make_one_hot(sample, size=cardinality))
  noise.append(continuous_samples)
  return np.hstack(noise)


def create_infogan_noise_sample(categorical_cardinality, num_continuous, style_size):
  def sample(batch_size):
    return encode_infogan_noise(
      categorical_cardinality,
      create_categorical_noise(categorical_cardinality, size=batch_size),
      create_continuous_noise(num_continuous, style_size, size=batch_size)
    )

  return sample


def create_gan_noise_sample(style_size):
  def sample(batch_size):
    return np.random.standard_normal(size=(batch_size, style_size))

  return sample


def make_one_hot(indices, size):
  as_one_hot = np.zeros((indices.shape[0], size))
  as_one_hot[np.arange(0, indices.shape[0]), indices] = 1.0
  return as_one_hot


def get_random_one_hot(batch_size, class_num):
  array = np.zeros((batch_size, class_num))
  for i in range(batch_size):
    j = np.random.randint(0, class_num, size=1)
    array[i][j] = 1.0
  return array


def get_regular_one_hot(batch_size, class_num):
  array = np.zeros((batch_size, class_num))
  j = 0
  for i in range(batch_size):
    array[i][j] = 1.0
    if (i + 1) % 3 == 0 and (j + 1) < class_num:
      j += 2
  return array


def get_one_hot(batch_size, class_num, class_index):
  array = np.zeros((batch_size, class_num))
  j = class_index
  for i in range(batch_size):
    array[i][j] = 1.0
  return array


def get_one_hot_from_data_files_name(batch_size, class_num, data_files):
  value_list = []
  for data_file in data_files:
    name = re.findall(r'.*\/(.+?)[0-9]*\.jpg', data_file)[0]
    name = re.sub(re.compile(r'[a-zA-Z0-9]{0,9}_'), "", name)
    value_list.append(Label[name].value)
  return get_one_hot_from_value_list(batch_size, class_num, value_list)


def get_one_hot_from_value_list(batch_size, class_num, value_list):
  array = np.zeros((batch_size, class_num))
  for i in range(batch_size):
    j = value_list[i]
    array[i][j] = 1.0
  return array


if __name__ == '__main__':
  # a = re.sub(re.compile(r'[a-zA-Z0-9]{0,9}_'), "", 'rot90_harbor')
  # data_files = glob(os.path.join("../dataset/uc_train_256_data", "*.jpg"))
  # get_one_hot_from_data_files_name(1, 1, data_files)
  # a = np.ones(shape=(64, 21)).astype(np.float32) * 15
  # a = np.random.randint(0, 1, size=(64, 21))
  # a = np.zeros((64, 21))
  # a = get_random_one_hot(3, 3)
  # a = get_regular_one_hot(64, 21)
  # a = get_one_hot(64, 21, 2)
  # print(a)
  pass
