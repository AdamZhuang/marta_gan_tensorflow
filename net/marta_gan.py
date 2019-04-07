import os
import time
import tensorflow as tf
import tensorlayer as tl
import logging
from glob import glob
from random import shuffle
from net.basic_network import MartaGanBasicNetWork
from utils.image_utils import Utils
from utils.noise_utils import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MartaGan:
  def __init__(self,
               dataset_path="./dataset/uc_test_256",
               checkpoint_path="./checkpoint",
               sample_path="./sample",
               image_size=256,
               image_dim=3,
               input_noise_dim=50,
               learning_rate=0.0001,
               beta1=0.5,
               batch_size=64):
    self.dataset_path = dataset_path
    self.checkpoint_path = checkpoint_path
    self.sample_path = sample_path
    self.image_size = image_size
    self.image_dim = image_dim
    self.input_noise_dim = input_noise_dim
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.batch_size = batch_size

    self.build_model()

  def build_model(self):
    logging.info("building model")

    #####################
    # for fake images
    #####################
    # latent code for generate
    self.input_latent_code_1 = tf.placeholder(tf.float32, [self.batch_size, 21], name='input_latent_code_1')
    self.input_latent_code_2 = tf.placeholder(tf.float32, [self.batch_size, 1], name='input_latent_code_2')
    self.input_latent_code_3 = tf.placeholder(tf.float32, [self.batch_size, 1], name='input_latent_code_3')
    # random noise for generator
    self.input_noise = tf.placeholder(tf.float32, [self.batch_size, self.input_noise_dim - 23], name='input_noise')
    # input z
    self.input_z = tf.concat(
      [self.input_latent_code_1, self.input_latent_code_2, self.input_latent_code_3, self.input_noise], axis=1)
    # the fake dataset of generating
    self.net_g = MartaGanBasicNetWork.generator(input_data=self.input_z, image_size=self.image_size, is_train=True,
                                                reuse=False)
    # feature of fake images
    self.net_d_output_fake, self.logits_fake, self.style_features_fake, self.latent_code_layer_1_fake, self.latent_code_layer_2_fake, self.latent_code_layer_3_fake \
      = MartaGanBasicNetWork.feature_extract_layer(input_data=self.net_g.outputs,
                                                   is_train=True,
                                                   reuse=False)
    #####################
    # for real images
    #####################
    # real image placeholder (the dataset will feed from dataset)
    self.real_images = tf.placeholder(tf.float32,
                                      [self.batch_size, self.image_size, self.image_size, self.image_dim],
                                      name='real_images')
    # feature of real images
    self.net_d_output_real, self.logits_real, self.style_features_real, _, _, _ \
      = MartaGanBasicNetWork.feature_extract_layer(input_data=self.real_images,
                                                   is_train=True,
                                                   reuse=True)
    #####################
    # sample
    #####################
    self.sample_image = self.sampler(self.input_z)

    #####################
    # loss and optimizer
    #####################
    self.g_loss, self.g_optimizer = self.generator()
    self.d_loss, self.d_optimizer = self.discriminator()

  def load_param(self, sess, net_g, net_d, load_epoch):
    try:
      g_vars_name = "./checkpoint/g_vars_{}.npz".format(str(load_epoch))
      d_vars_name = "./checkpoint/d_vars_{}.npz".format(str(load_epoch))
      net_g_loaded_params = tl.files.load_npz(name=g_vars_name)
      net_d_loaded_params = tl.files.load_npz(name=d_vars_name)
      tl.files.assign_params(sess, net_g_loaded_params, net_g)
      tl.files.assign_params(sess, net_d_loaded_params, net_d)
      logging.info("[*] Loading checkpoints SUCCESS!")
    except Exception as e:
      logging.error("[*] Loading checkpoints Failed!\n" + str(e))

  def train(self, epoch=1000, load_epoch=0):
    with tf.Session() as sess:
      # init network variables
      sess.run(tf.global_variables_initializer())
      # get dataset files path
      data_files = glob(os.path.join(self.dataset_path, "*.jpg"))
      # load param
      self.load_param(sess, self.net_g, self.net_d_output_fake, load_epoch)
      # start training
      for cur_epoch in range(load_epoch + 1, epoch):
        # every epoch shuffle data_files
        shuffle(data_files)
        # get total batch
        total_batch = int(len(data_files) / self.batch_size)
        for cur_batch in range(total_batch):
          # get one batch of real images
          batch_files = data_files[cur_batch * self.batch_size:(cur_batch + 1) * self.batch_size]
          batch_images = [Utils.get_image(batch_file, self.image_size, resize_w=self.image_size, is_grayscale=0) for
                          batch_file in batch_files]
          batch_real_images = np.array(batch_images).astype(np.float32)
          # random init a latent code for fake image generate
          input_latent_code_1 = get_random_one_hot(self.batch_size, 21)
          input_latent_code_2 = get_continuous_code(self.batch_size)
          input_latent_code_3 = get_continuous_code(self.batch_size)
          # every batch get a random input noise
          input_noise = np.random.uniform(low=-1, high=1, size=(self.batch_size, self.input_noise_dim - 21 - 1 - 1)).astype(
            np.float32)

          start_time = time.time()
          # train d
          d_loss, _ = sess.run([self.d_loss, self.d_optimizer],
                               feed_dict={self.input_latent_code_1: input_latent_code_1,
                                          self.input_latent_code_2: input_latent_code_2,
                                          self.input_latent_code_3: input_latent_code_3,
                                          self.input_noise: input_noise,
                                          self.real_images: batch_real_images})
          # train g
          for _ in range(2):
            g_loss, _ = sess.run([self.g_loss, self.g_optimizer],
                                 feed_dict={self.input_noise: input_noise,
                                            self.input_latent_code_1: input_latent_code_1,
                                            self.input_latent_code_2: input_latent_code_2,
                                            self.input_latent_code_3: input_latent_code_3,
                                            self.real_images: batch_real_images})
          end_time = time.time()

          logging.info("epoch:[%4d/%4d], batch:[%4d/%4d], d_loss: %.8f, g_loss: %.8f, time: %4f",
                       cur_epoch, epoch, cur_batch + 1, total_batch, d_loss, g_loss, end_time - start_time)

        if cur_epoch % 2 == 0:
          # save images
          for sample_image_num in range(3):
            input_latent_code_1 = get_one_hot_by_index(self.batch_size, 21, sample_image_num)
            input_latent_code_2 = get_continuous_code(self.batch_size)
            input_latent_code_3 = get_continuous_code(self.batch_size)
            input_noise = np.random.uniform(low=-1, high=1, size=(self.batch_size, self.input_noise_dim - 21 - 1 - 1)).astype(
              np.float32)
            images = sess.run(self.sample_image, feed_dict={self.input_noise: input_noise,
                                                            self.input_latent_code_1: input_latent_code_1,
                                                            self.input_latent_code_1: input_latent_code_2,
                                                            self.input_latent_code_1: input_latent_code_3,
                                                            self.real_images: batch_real_images})
            # save images
            side = 1
            while side * side < self.batch_size:
              side += 1
            Utils.save_images(images,
                              [side, side],
                              os.path.join(self.sample_path,
                                           "epoch{}-sample{}.png".format(str(cur_epoch), str(sample_image_num))))
          logging.info("sample image saved!")

        if cur_epoch % 10 == 0:
          # save net param
          g_vars = self.net_g.all_params
          d_vars = self.net_feature_extract_fake.all_params
          g_vars_name = os.path.join(self.checkpoint_path, "g_vars_{}.npz".format(str(cur_epoch)))
          d_vars_name = os.path.join(self.checkpoint_path, "d_vars_{}.npz".format(str(cur_epoch)))
          tl.files.save_npz(g_vars, name=g_vars_name, sess=sess)
          tl.files.save_npz(d_vars, name=d_vars_name, sess=sess)
          logging.info("net param saved!")

  def generator(self):
    """
    the generator should gen image that is smiler with real image

    the loss contains 2 part
    g_loss1: high-dim feature match loss
    g_loss2: high-dim feature perceptual loss

    """
    # loss of generator
    g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_fake,
                                                                     labels=tf.ones_like(self.logits_fake)))
    g_loss2 = 0
    for layer_name in self.style_features_fake:
      # param
      _, height, width, dim = self.style_features_fake[layer_name].get_shape()
      N = height.value * width.value
      M = dim.value
      w = 1 / len(self.style_features_fake)
      #
      gram_fake = self.gram_matrix(self.style_features_fake[layer_name])
      gram_real = self.gram_matrix(self.style_features_real[layer_name])
      g_loss2 += w * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((gram_fake - gram_real), 2))
    # g_loss_q_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.latent_code_layer_1_fake,
    #                                                                  labels=self.input_latent_code_1))
    # g_loss_q_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.latent_code_layer_2_fake,
    #                                                                  labels=self.input_latent_code_1))
    # g_loss_q_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.latent_code_layer_3_fake,
    #                                                                     labels=self.input_latent_code_1))
    g_loss_q_1 = tf.reduce_mean(tf.multiply(self.latent_code_layer_1_fake, self.input_latent_code_1), axis=1)
    g_loss_q_2 = tf.reduce_mean(tf.multiply(self.latent_code_layer_2_fake, self.input_latent_code_2), axis=1)
    g_loss_q_3 = tf.reduce_mean(tf.multiply(self.latent_code_layer_3_fake, self.input_latent_code_3), axis=1)
    logging.info("g_loss1:" + str(g_loss1))
    logging.info("g_loss2:" + str(g_loss2))
    logging.info("g_loss3:" + str((g_loss_q_1 + g_loss_q_2 + g_loss_q_3)))
    g_loss = 1 * g_loss1 + 1 * g_loss2 - 1 * (g_loss_q_1 + g_loss_q_2 + g_loss_q_3)

    # optimizer of generator
    g_vars = self.net_g.all_params
    # self.net_g.print_params(False)
    g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(g_loss, var_list=g_vars)

    return g_loss, g_optimizer

  def discriminator(self):
    """
    the discriminator should marked real image as 1 and mark fake image as 0

    """
    # loss of discriminator
    d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_real,
                                              labels=tf.ones_like(self.logits_real)))  # real == 1
    d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_fake,
                                              labels=tf.zeros_like(self.logits_fake)))  # fake == 0
    d_loss = d_loss_real + d_loss_fake

    # optimizer of discriminator
    d_vars = self.net_d_output_real.all_params
    d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(d_loss, var_list=d_vars)

    return d_loss, d_optimizer

  def sampler(self, input_data):
    """
    set is_train=False, reuse=True, fix param ten generate sampler

    """
    net_g = MartaGanBasicNetWork.generator(input_data=input_data, image_size=self.image_size, is_train=False,
                                           reuse=True)
    return net_g.outputs

  def generate_image(self, load_epoch):
    with tf.Session() as sess:
      self.load_param(sess, self.net_g, self.net_feature_extract_fake, load_epoch)
      for sample_image_num in range(3):
        input_latent_code = get_one_hot_by_index(self.batch_size, 21, sample_image_num)
        input_noise = np.random.uniform(low=-1, high=1, size=(self.batch_size, self.input_noise_dim - 21)).astype(
          np.float32)
        images = sess.run(self.sample_image, feed_dict={self.input_noise: input_noise,
                                                        self.input_latent_code_1: input_latent_code
                                                        })
        # save images
        side = 1
        while side * side < self.batch_size:
          side += 1
        Utils.save_images(images,
                          [side, side],
                          os.path.join(self.sample_path,
                                       "gen/gen-epoch{}-sample{}.png".format(str(load_epoch), str(sample_image_num))))
      logging.info("sample image saved!")

  def gram_matrix(self, tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)
    return gram
