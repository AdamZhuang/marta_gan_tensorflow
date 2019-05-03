# from net.label import LabelUtils
# from net.marta_gan import MartaGan
# from utils.file_utils import get_file_names
#
# TRAIN_DATE_SET_PATH = "./dataset/uc_train_256"
# SAMPLE_PATH = "./sample"
# CHECKPOINT_PATH = "./checkpoint"
#
# if __name__ == '__main__':
#   # 训练集
#   train_image_label_dict = {}
#   path, file_names = get_file_names(TRAIN_DATE_SET_PATH)
#   for file_name in file_names:
#     file_name = path + "/" + file_name
#     train_image_label_dict[file_name] = LabelUtils.get_label_num(file_name)
#
#
#   marta_gan = MartaGan(dataset_path="./dataset/uc_train_256", batch_size=64, learning_rate=0.0002)
#
#   marta_gan.train(image_label_dict=train_image_label_dict,
#                   epoch=10000,
#                   load_epoch=0,
#                   batch_size=64)


import os
from net.label import LabelUtils

if __name__ == '__main__':
  for i in range(21):
    try:
      path = './dataset/' + LabelUtils.get_label_name(i)
      os.mkdir(path)
    except Exception:
      pass
