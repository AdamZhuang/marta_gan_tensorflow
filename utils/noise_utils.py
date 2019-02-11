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


def encode_infogan_noise(categorical_cardinality, categorical_samples, continuous_samples):
  noise = []
  for cardinality, sample in zip(categorical_cardinality, categorical_samples):
    noise.append(make_one_hot(sample, size=cardinality))
  noise.append(continuous_samples)
  return np.hstack(noise)


def create_infogan_noise_sample(categorical_cardinality, num_continuous, style_size):
  def sample(batch_size):
    return encode_infogan_noise(
      categorical_cardinality,
      create_categorical_noise(categorical_cardinality, size=batch_size),
      create_continuous_noise(num_continuous, style_size, size=batch_size)
    )

  return sample


def create_gan_noise_sample(style_size):
  def sample(batch_size):
    return np.random.standard_normal(size=(batch_size, style_size))

  return sample


def make_one_hot(indices, size):
  as_one_hot = np.zeros((indices.shape[0], size))
  as_one_hot[np.arange(0, indices.shape[0]), indices] = 1.0
  return as_one_hot


if __name__ == '__main__':
  a = np.random.randint(0, 21, size=64)
  print(a)
