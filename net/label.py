from enum import Enum


class Label(Enum):
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
  mobilehomepark = 21




# if __name__ == '__main__':
#   date_files = glob(os.path.join("../dataset/uc_test_256", "*.jpg"))
#   name_list = []
#   for data_file in date_files:
#     temp = data_file.replace("../dataset/uc_test_256/", "")
#     temp = re.findall(r'(.+?)[0-9]*\.jpg', temp)[0]
#     name_list.append(temp)
#
#   name_set = set(name_list)
#   with open('classifications.txt', 'w') as f:
#     count = 1
#     for name in name_set:
#       if count == len(name_set) - 1:
#         print(len(name_set))
#         f.write(name + " = " + str(count))
#       else:
#         f.write(name + " = " + str(count) + "\n")
#
#       count += 1
