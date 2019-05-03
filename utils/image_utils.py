import cv2
import numpy as np

from utils.utils import Utils


def read_image(image_path_list):
  images = []
  for image_path in image_path_list:
    # image = cv2.imread(image_path)
    image = Utils.get_image(image_path, 256, resize_w=256)
    images.append(image)
    if image is None:
      print(image_path)

  return np.array(images)


def save_image(filename, image):
  cv2.imwrite(filename, image)
