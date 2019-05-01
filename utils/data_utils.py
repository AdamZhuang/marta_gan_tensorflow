import math
import numpy as np
import utils.image_utils as image_utils


def get_shuffled_batch_file_names(image_label_dict, batch_size):
  # total batch
  total_batch = math.ceil(len(image_label_dict) / batch_size)
  # all image_names
  image_names = list(image_label_dict)
  np.random.shuffle(image_names)
  # divide all into n batch
  data = []
  for cur_batch in range(total_batch):
    data.append(image_names[cur_batch * batch_size:(cur_batch + 1) * batch_size])

  return data


def get_one_batch_data(one_batch_file_names, image_label_dict, class_num):
  # data
  images = image_utils.read_image(one_batch_file_names)
  batch_data = []
  for image in images:
    batch_data.append(image)
  batch_data = np.array(batch_data)

  # label
  labels = []
  for image_name in one_batch_file_names:
    labels.append(image_label_dict[image_name])
  batch_labels = np.eye(class_num)[labels]

  return batch_data, batch_labels


def get_one_batch_data_without_labels(one_batch_file_names):
  # data
  return image_utils.read_image(one_batch_file_names)

def get_batch_one_hot(code, batch_size, class_num):
  code = np.array([code] * batch_size)
  return np.eye(class_num)[code]

