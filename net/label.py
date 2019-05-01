from enum import Enum


class Label(Enum):
  mobilehomepark = 0
  intersection = 1
  runway = 2
  river = 3
  beach = 4
  tenniscourt = 5
  forest = 6
  buildings = 7
  denseresidential = 8
  chaparral = 9
  golfcourse = 10
  mediumresidential = 11
  airplane = 12
  agricultural = 13
  baseballdiamond = 14
  freeway = 15
  harbor = 16
  overpass = 17
  sparseresidential = 18
  storagetanks = 19
  parkinglot = 20


class LabelUtils:
  @staticmethod
  def get_label_num(name):
    for lable in Label:
      label_name = str(lable).replace("Label.", "")
      if name.find(label_name) >= 0:
        return lable.value
    return -1

  @staticmethod
  def get_label_name(num):
    label_name = Label(num)
    return str(label_name).replace("Label.", "")

  @staticmethod
  def get_total_label_num():
    return len(Label)
