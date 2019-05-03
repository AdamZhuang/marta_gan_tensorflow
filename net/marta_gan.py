import math
import os
import time
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import logging
from glob import glob
from random import shuffle
from net.basic_network import MartaGanBasicNetWork
from utils import data_utils
from utils.utils import Utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MartaGan:
  def __init__(self,
               dataset_path="./dataset/uc_test_256",
               checkpoint_path="./checkpoint",
               sample_path="./sample",
               class_num=21,
               image_size=256,
               image_dim=3,
               input_z_dim=100,
               input_c_dim=21,
               learning_rate=0.001,
               beta1=0.5,
               batch_size=64):
    self.dataset_path = dataset_path
    self.checkpoint_path = checkpoint_path
    self.sample_path = sample_path
    self.class_num = class_num
    self.image_size = image_size
    self.image_dim = image_dim
    self.input_c_dim = input_c_dim
    self.input_z_dim = input_z_dim
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.batch_size = batch_size

    self.build_model()

  def build_model(self):
    logging.info("building model")

    #####################
    # for fake images
    #####################
    # random noise for generator
    self.input_c = tf.placeholder(tf.float32, [self.batch_size, self.input_c_dim], name='input_c')
    self.input_z = tf.placeholder(tf.float32, [self.batch_size, self.input_z_dim], name='input_z')
    # the fake dataset of generating
    self.output_img = MartaGanBasicNetWork.generator(input_c=self.input_c,
                                                     input_z=self.input_z,
                                                     image_size=self.image_size,
                                                     is_train=True,
                                                     reuse=False)
    # feature of fake images
    self.output_scalar_fake, self.feature_fake, self.style_features_fake = MartaGanBasicNetWork.discriminator(
      input_c=self.input_c,
      input_img=self.output_img.outputs,
      is_train=True,
      reuse=False)

    #####################
    # for real images
    #####################
    # real image placeholder (the dataset will feed from dataset)
    self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.image_dim],
                                      name='real_images')
    # feature of real images
    self.output_scalar_real, self.feature_real, self.style_features_real = MartaGanBasicNetWork.discriminator(
      input_c=self.input_c,
      input_img=self.real_images,
      is_train=True,
      reuse=True)

    #####################
    # for real images but not pair
    #####################
    # real image placeholder (the dataset will feed from dataset)
    self.real_images_but_not_pair = tf.placeholder(tf.float32,
                                                   [self.batch_size, self.image_size, self.image_size, self.image_dim],
                                                   name='real_images')
    # feature of real images
    self.output_scalar_real_but_not_pair, _, _ = MartaGanBasicNetWork.discriminator(
      input_c=self.input_c,
      input_img=self.real_images_but_not_pair,
      is_train=True,
      reuse=True)

    #####################
    # sample
    #####################
    self.sample_image = self.sampler()

    #####################
    # loss and optimizer
    #####################
    self.g_loss, self.g_optimizer = self.generator(self.output_scalar_fake,
                                                   self.feature_real, self.feature_fake,
                                                   self.style_features_real, self.style_features_fake)
    self.d_loss, self.d_optimizer = self.discriminator(self.output_scalar_real,
                                                       self.output_scalar_fake,
                                                       self.output_scalar_real_but_not_pair)

  def _load_param(self, sess, net_g, net_d, load_epoch):
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

  def train(self, image_label_dict, batch_size=64, epoch=1000, load_epoch=0):
    with tf.Session() as sess:
      # init network variables
      sess.run(tf.global_variables_initializer())
      # load param
      self._load_param(sess, self.output_img, self.output_scalar_fake, load_epoch)
      for cur_epoch in range(load_epoch + 1, epoch):
        total_batch = math.floor(len(image_label_dict) / batch_size)

        batched_file_names_1, batched_file_names_2 = data_utils.get_shuffled_batch_file_names(image_label_dict,
                                                                                              batch_size)
        for cur_batch in range(total_batch):
          # real image and its labels
          cur_batch_file_names_1 = batched_file_names_1[cur_batch]
          cur_batch_file_names_2 = batched_file_names_2[cur_batch]
          batch_images_real, batch_labels_real = data_utils.get_one_batch_data(cur_batch_file_names_1, image_label_dict,
                                                                               self.class_num)

          batch_images_real_but_not_pair = data_utils.get_one_batch_data_without_labels(cur_batch_file_names_2)

          input_z = np.random.uniform(low=-1, high=1, size=(len(batch_labels_real), self.input_z_dim)).astype(
            np.float32)

          start_time = time.time()
          # train d
          d_loss, _ = sess.run([self.d_loss, self.d_optimizer],
                               feed_dict={
                                 self.input_c: batch_labels_real,
                                 self.input_z: input_z,
                                 self.real_images: batch_images_real,
                                 self.real_images_but_not_pair: batch_images_real_but_not_pair
                               })

          # train g
          g_loss = 0
          for _ in range(2):
            g_loss, _ = sess.run([self.g_loss, self.g_optimizer],
                                 feed_dict={
                                   self.input_c: batch_labels_real,
                                   self.input_z: input_z,
                                   self.real_images: batch_images_real,
                                   self.real_images_but_not_pair: batch_images_real_but_not_pair
                                 })

          end_time = time.time()

          logging.info("epoch:[%4d/%4d], batch:[%4d/%4d], d_loss: %.8f, g_loss: %.8f, time: %4f",
                       cur_epoch, epoch, cur_batch + 1, total_batch, d_loss, g_loss, end_time - start_time)
        # # init network variables
        # sess.run(tf.global_variables_initializer())
        # # get dataset files path
        # data_files = glob(os.path.join(self.dataset_path, "*.jpg"))
        # # load param
        # # self.load_parma(sess, self.net_g, self.net_feature_extract_fake, load_epoch)
        # # static_input_noise
        # static_input_noise = np.random.uniform(low=-1, high=1, size=(self.batch_size, self.input_z_dim)).astype(
        #   np.float32)
        # for cur_epoch in range(load_epoch + 1, epoch):
        #   # every epoch shuffle data_files
        #   shuffle(data_files)
        #   # get total batch
        #   total_batch = int(len(data_files) / self.batch_size)
        #   for cur_batch in range(total_batch):
        #
        #     input_c = np.random.uniform(low=-1, high=1, size=(self.batch_size, self.input_c_dim)).astype(
        #       np.float32)
        #     input_z = np.random.uniform(low=-1, high=1, size=(self.batch_size, self.input_z_dim)).astype(
        #       np.float32)
        #     # get one batch of real images
        #     batch_files = data_files[cur_batch * self.batch_size:(cur_batch + 1) * self.batch_size]
        #     batch_images = [Utils.get_image(batch_file, self.image_size, resize_w=self.image_size,
        #                                     is_grayscale=0) for batch_file in batch_files]
        #     batch_real_images = np.array(batch_images).astype(np.float32)
        #
        #     start_time = time.time()

        if cur_epoch % 1 == 0:
          # save images
          input_c = np.random.uniform(low=-1, high=1, size=(self.batch_size, self.input_c_dim)).astype(
            np.float32)
          input_z = np.random.uniform(low=-1, high=1, size=(self.batch_size, self.input_z_dim)).astype(
            np.float32)
          for sample_image_num in range(1):
            images = sess.run(self.sample_image,
                              feed_dict={
                                self.input_c: input_c,
                                self.input_z: input_z
                              })
            # save images
            side = 1
            while side * side < self.batch_size:
              side += 1
            Utils.save_images(images, [side, side],
                              os.path.join(self.sample_path,
                                           "epoch{}-sample{}.png".format(str(cur_epoch), str(sample_image_num))))
          logging.info("sample image saved!")

        if cur_epoch % 10 == 0:
          # save net param
          g_vars = self.output_img.all_params
          d_vars = self.output_scalar_fake.all_params
          g_vars_name = os.path.join(self.checkpoint_path, "g_vars_{}.npz".format(str(cur_epoch)))
          d_vars_name = os.path.join(self.checkpoint_path, "d_vars_{}.npz".format(str(cur_epoch)))
          tl.files.save_npz(g_vars, name=g_vars_name, sess=sess)
          tl.files.save_npz(d_vars, name=d_vars_name, sess=sess)
          logging.info("net param saved!")

  def generator(self, output_scalar_fake, feature_real, feature_fake, style_features_real, style_features_fake):
    """
    the generator should gen image that is smiler with real image

    the loss contains 2 part
    g_loss1: high-dim feature match loss
    g_loss2: high-dim feature perceptual loss

    """
    # loss of generator
    g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_scalar_fake.outputs,
                                                                     labels=tf.ones_like(output_scalar_fake.outputs)))
    g_loss2 = tf.reduce_mean(tf.nn.l2_loss(feature_real - feature_fake)) / (self.image_size * self.image_size)
    g_loss3 = 0
    for layer_name in style_features_fake:
      # param
      _, height, width, dim = style_features_fake[layer_name].get_shape()
      N = height.value * width.value
      M = dim.value
      w = 1 / len(style_features_fake)
      #
      gram_fake = self.gram_matrix(style_features_fake[layer_name])
      gram_real = self.gram_matrix(style_features_real[layer_name])
      g_loss3 += w * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((gram_fake - gram_real), 2))

    g_loss = g_loss1 + 0.1 * g_loss2 + 10 * g_loss3

    # optimizer of generator
    g_vars = self.output_img.all_params
    # self.net_g.print_params(False)
    g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(g_loss, var_list=g_vars)

    return g_loss, g_optimizer

  def discriminator(self, output_scalar_real, output_scalar_fake, output_scalar_real_but_not_pair=None):
    """
    the discriminator should marked real image as 1 and mark fake image as 0

    """
    # loss of discriminator
    d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=output_scalar_real.outputs,
                                              labels=tf.ones_like(output_scalar_real.outputs)))  # real == 1
    d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=output_scalar_fake.outputs,
                                              labels=tf.zeros_like(output_scalar_fake.outputs)))  # fake == 0
    d_loss_real_but_not_pair = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=output_scalar_real_but_not_pair.outputs,
                                              labels=0.5 * tf.ones_like(
                                                output_scalar_real_but_not_pair.outputs)))  # fake == 0
    d_loss = d_loss_real + d_loss_fake + d_loss_real_but_not_pair
    # d_loss = d_loss_real + d_loss_real_but_not_pair
    # d_loss = d_loss_real + d_loss_fake

    # optimizer of discriminator
    d_vars = output_scalar_real.all_params
    d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(d_loss, var_list=d_vars)

    return d_loss, d_optimizer

  def sampler(self):
    """
    set is_train=False, reuse=True, fix param ten generate sampler

    """
    net_g = MartaGanBasicNetWork.generator(input_c=self.input_c,
                                           input_z=self.input_z,
                                           image_size=self.image_size,
                                           is_train=False,
                                           reuse=True)
    return net_g.outputs

  def generate_image(self, num):
    pass

  def gram_matrix(self, tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)
    return gram
