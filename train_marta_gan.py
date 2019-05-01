from net.label import LabelUtils
from net.marta_gan import MartaGan
from utils.file_utils import get_file_names

TRAIN_DATE_SET_PATH = "dataset/uc_train_256"

if __name__ == '__main__':
  # 训练集
  train_image_label_dict = {}
  path, file_names = get_file_names(TRAIN_DATE_SET_PATH)
  for file_name in file_names:
    file_name = path + "/" + file_name
    train_image_label_dict[file_name] = LabelUtils.get_label_num(file_name)


  marta_gan = MartaGan(learning_rate=0.0005)
  marta_gan.train(train_image_label_dict, epoch=10000, load_epoch=0)
