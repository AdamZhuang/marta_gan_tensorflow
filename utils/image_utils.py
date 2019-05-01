import cv2
import numpy as np

def read_image(image_path_list):
  images = []
  for image_path in image_path_list:
    image = cv2.imread(image_path)
    images.append(image)

  return np.array(images)


def save_image(image):
  pass
