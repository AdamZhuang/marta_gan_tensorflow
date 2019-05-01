import os
import os.path


def get_file_names(path):
  for _, _, file_names in os.walk(path):
    return path, file_names
