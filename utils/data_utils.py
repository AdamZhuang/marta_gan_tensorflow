import math
import numpy as np
import utils.image_utils as image_utils
from net.label import LabelUtils


def get_shuffled_batch_file_names(image_label_dict, batch_size):
  # total batch
  total_batch = math.ceil(len(image_label_dict) / batch_size)
  # all image_names
  image_names = list(image_label_dict)
  np.random.shuffle(image_names)
  # divide all into n batch
  n_batch_image_name_1 = []
  n_batch_image_name_2 = []
  for cur_batch in range(total_batch):
    one_batch_image_names_1 = image_names[cur_batch * batch_size:(cur_batch + 1) * batch_size]
    n_batch_image_name_1.append(one_batch_image_names_1)

    one_batch_image_names_2 = []
    for image_name_1 in one_batch_image_names_1:
      image_label_1 = LabelUtils.get_label_num(image_name_1)
      while True:
        random_int = np.random.random_integers(len(image_names) - 1)
        image_name_2 = image_names[random_int]
        image_label_2 = LabelUtils.get_label_num(image_name_2)
        if image_label_1 != image_label_2:
          one_batch_image_names_2.append(image_name_2)
          break
    n_batch_image_name_2.append(one_batch_image_names_2)

  return n_batch_image_name_1, n_batch_image_name_2


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


def get_batch_one_hot(code, class_num):
  return np.eye(class_num)[code]


def save_image(images, horizontal_num, vertical_num, filename):
  # 初始化空白图片
  image_height, image_width = images.shape[1], images.shape[2]
  img = np.zeros((image_height * vertical_num, image_width * horizontal_num, 3))

  # 讲图片的数据填充进去
  for index, image in enumerate(images):
    horizontal_index = index % horizontal_num
    vertical_index = horizontal_index % vertical_num
    img[vertical_index * image_height:vertical_index * image_height + image_height,
    horizontal_index * image_width:horizontal_index * image_width + image_width, :] = image

  image_utils.save_image(filename, img)


# if __name__ == '__main__':
# from net.label import LabelUtils
# import os
#
#
# def get_file_names(path):
#   for _, _, file_names in os.walk(path):
#     return path, file_names
#
#   TRAIN_DATE_SET_PATH = "../dataset/uc_train_256"
#   # 训练集
#   train_image_label_dict = {}
#   path, file_names = get_file_names(TRAIN_DATE_SET_PATH)
#   for file_name in file_names:
#     file_name = path + "/" + file_name
#     train_image_label_dict[file_name] = LabelUtils.get_label_num(file_name)
#
#   a, b = get_shuffled_batch_file_names(train_image_label_dict, 64)
#
#   print(len(a[0]))
#   print(a[0])
#   print('----------------')
#   print(len(b[0]))
#   print(b[0])
