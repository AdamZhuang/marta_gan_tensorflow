from enum import Enum


class Label(Enum):
  sparseresidential = 0
  mediumresidential = 1
  denseresidential = 2
  buildings = 3
  tenniscourt = 4
  golfcourse = 5
  baseballdiamond = 6
  intersection = 7
  overpass = 8
  freeway = 9
  runway= 10
  river = 11
  harbor = 12
  airplane = 13
  storagetanks = 14
  mobilehomepark = 15
  parkinglot = 16
  forest = 17
  chaparral = 18
  beach = 19
  agricultural = 20


class LabelUtils:
  @staticmethod
  def get_label_num(name):
    for label in Label:
      label_name = str(label).replace("Label.", "")
      if name.find(label_name) >= 0:
        return label.value
    return -1

  @staticmethod
  def get_label_name(num):
    label_name = Label(num)
    return str(label_name).replace("Label.", "")

  @staticmethod
  def get_total_label_num():
    return len(Label)