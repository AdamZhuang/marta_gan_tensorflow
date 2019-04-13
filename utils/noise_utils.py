import numpy as np


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


def get_continuous_code(batch_size):
  return np.random.rand(batch_size, 1)


def get_random_one_hot(batch_size, class_num):
  array = np.zeros((batch_size, class_num))
  for i in range(batch_size):
    j = np.random.randint(0, class_num, size=1)
    array[i][j] = 1.0
  return array


def get_one_hot_by_index(batch_size, class_num, class_index):
  array = np.zeros((batch_size, class_num))
  j = class_index
  for i in range(batch_size):
    array[i][j] = 1.0
  return array

def get_regular_one_hot(batch_size, class_num):
  array = np.zeros((batch_size, class_num))
  j = 0
  num = 0
  for i in range(batch_size):
    array[i][j] = 1.0
    num += 1
    if num == 6:
      j += 1
      j = j % class_num
      num = 0
  return array


if __name__ == '__main__':
  a = get_regular_one_hot(64, 21)
  print(a)
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
