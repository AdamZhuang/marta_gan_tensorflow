import logging
import math
import os
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from net.basic_network import MartaGanBasicNetWork
from utils import data_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MartaGan:
  def __init__(self,
               checkpoint_path="./checkpoint",
               sample_path="./sample",
               class_num=21,
               input_noise_dim=29,
               image_size=256,
               image_dim=3,
               learning_rate=0.001,
               beta1=0.5):
    self.checkpoint_path = checkpoint_path
    self.sample_path = sample_path
    self.class_num = class_num
    self.image_size = image_size
    self.image_dim = image_dim
    self.input_noise_dim = input_noise_dim
    self.learning_rate = learning_rate
    self.beta1 = beta1

    self.build_model()

  def build_model(self):
    logging.info("building model")

    #####################
    # for fake images
    #####################
    # random noise for generator
    self.input_code = tf.placeholder(tf.float32, [None, self.class_num], name='input_code')
    self.input_noise = tf.placeholder(tf.float32, [None, self.input_noise_dim], name='input_noise')
    # the fake dataset of generating
    self.net_g = MartaGanBasicNetWork.generator(input_c=self.input_code, input_z=self.input_noise,
                                                image_size=self.image_size, is_train=True, reuse=False)
    # feature of fake images
    self.output_confidence_fake, self.output_pair_confidence_fake, self.feature_fake, self.style_features_fake = MartaGanBasicNetWork.discriminator(
      input_c=self.input_code,
      input_x=self.net_g.outputs,
      class_num=self.class_num,
      is_train=True,
      reuse=False)

    #####################
    # for real images
    #####################
    # real image placeholder (the dataset will feed from dataset)
    self.real_images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_dim],
                                      name='real_images')
    # feature of real images
    self.output_confidence_real, self.output_pair_confidence_real, self.feature_real, self.style_features_real = MartaGanBasicNetWork.discriminator(
      input_c=self.input_code,
      input_x=self.real_images,
      class_num=self.class_num,
      is_train=True,
      reuse=True)

    self.sample_image = self.sampler(self.input_code, self.input_noise);

    #####################
    # loss and optimizer
    #####################
    self.g_loss, self.g_optimizer = self._generator(self.output_confidence_fake.outputs,
                                                    self.feature_real,
                                                    self.feature_fake,
                                                    self.style_features_fake,
                                                    self.style_features_real)
    self.d_loss, self.d_optimizer = self._discriminator(self.output_pair_confidence_real,
                                                        self.output_pair_confidence_fake,
                                                        self.output_confidence_real,
                                                        self.output_confidence_fake)

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

  def train(self, image_label_dict, batch_size=64, epoch=1000, load_epoch=0):
    with tf.Session() as sess:
      # init network variables
      sess.run(tf.global_variables_initializer())
      # load param
      self.load_param(sess, self.net_g, self.output_confidence_fake, load_epoch)
      for cur_epoch in range(load_epoch + 1, epoch):
        total_batch = math.ceil(len(image_label_dict) / batch_size)
        batched_file_names = data_utils.get_shuffled_batch_file_names(image_label_dict, batch_size)
        for cur_batch in range(total_batch):
          # real image and its labels
          cur_batch_file_names = batched_file_names[cur_batch]
          batch_images_real, batch_labels_real = data_utils.get_one_batch_data(cur_batch_file_names, image_label_dict,
                                                                               self.class_num)
          # fake image input noise
          input_noise = np.random.uniform(low=-1, high=1,
                                          size=(len(cur_batch_file_names), self.input_noise_dim)).astype(
            np.float32)

          start_time = time.time()
          # train d
          d_loss, _ = sess.run([self.d_loss, self.d_optimizer],
                               feed_dict={self.input_code: batch_labels_real,
                                          self.input_noise: input_noise,
                                          self.real_images: batch_images_real})
          # train g
          for _ in range(2):
            g_loss, _ = sess.run([self.g_loss, self.g_optimizer],
                                 feed_dict={self.input_code: batch_labels_real,
                                            self.input_noise: input_noise,
                                            self.real_images: batch_images_real})

          logging.info("epoch:[%4d/%4d], batch:[%4d/%4d], d_loss: %.8f, g_loss: %.8f, time: %4f",
                       cur_epoch, epoch, cur_batch + 1, total_batch, d_loss, g_loss, time.time() - start_time)

        if cur_epoch % 1 == 0:
          # static_input_noise

          # save images
          for sample_image_num in range(1):
            code = np.arange(self.class_num)
            batch_one_hot = data_utils.get_batch_one_hot(code, self.class_num)

            input_noise = np.random.uniform(low=-1, high=1, size=(len(code), self.input_noise_dim)).astype(
              np.float32)

            images = sess.run(self.sample_image, feed_dict={self.input_code: batch_one_hot,
                                                            self.input_noise: input_noise})
            # save images
            side = 1
            while side * side < batch_size:
              side += 1

            data_utils.save_image(images,
                                  side,
                                  side,
                                  os.path.join(self.sample_path,
                                                       "epoch{}-sample{}.png".format(str(cur_epoch), str(sample_image_num))))
          logging.info("sample image saved!")

        if cur_epoch % 10 == 0:
          # save net param
          g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
          d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
          g_vars_name = os.path.join(self.checkpoint_path, "g_vars_{}.npz".format(str(cur_epoch)))
          d_vars_name = os.path.join(self.checkpoint_path, "d_vars_{}.npz".format(str(cur_epoch)))
          tl.files.save_npz(g_vars, name=g_vars_name, sess=sess)
          tl.files.save_npz(d_vars, name=d_vars_name, sess=sess)
          logging.info("net param saved!")

  def _generator(self, output_confidence, feature_real, feature_fake, style_features_fake, style_features_real):
    """
    the generator should gen image that is smiler with real image

    the loss contains 2 part
    g_loss1: high-dim feature match loss
    g_loss2: high-dim feature perceptual loss

    """
    # loss of generator
    g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_confidence,
                                                                     labels=tf.ones_like(output_confidence)))
    g_loss2 = tf.reduce_mean(tf.nn.l2_loss(feature_real - feature_fake)) / (self.image_size * self.image_size)
    g_loss3 = 0
    for layer_name in style_features_fake:
      # param
      _, height, width, dim = style_features_fake[layer_name].get_shape()
      N = height.value * width.value
      M = dim.value
      w = 1 / len(style_features_fake)
      #
      gram_fake = self._gram_matrix(style_features_fake[layer_name])
      gram_real = self._gram_matrix(style_features_real[layer_name])
      g_loss3 += w * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((gram_fake - gram_real), 2))

    g_loss = g_loss1 + 0.1 * g_loss2 + 10 * g_loss3

    # optimizer of generator
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
    # self.net_g.print_params(False)
    g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(g_loss, var_list=g_vars)

    return g_loss, g_optimizer

  def _discriminator(self, output_pair_confidence_real, output_pair_confidence_fake,
                     output_confidence_real, output_confidence_fake):
    """
    the discriminator should marked real image as 1 and mark fake image as 0

    """
    # loss of discriminator
    d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=output_confidence_real.outputs,
                                              labels=tf.ones_like(output_confidence_real.outputs)))  # real == 1
    d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=output_confidence_fake.outputs,
                                              labels=tf.zeros_like(output_confidence_fake.outputs)))  # fake == 0
    d_loss1 = d_loss_real + d_loss_fake

    # incorrect pair should add loss
    d_loss_pair_fake = tf.reduce_mean(tf.sigmoid(output_pair_confidence_fake.outputs))
    d_loss_pair_real = -tf.reduce_mean(tf.sigmoid(output_pair_confidence_real.outputs))
    d_loss2 = d_loss_pair_fake + d_loss_pair_real

    d_loss = d_loss2

    # optimizer of discriminator
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

    d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(d_loss, var_list=d_vars)

    return d_loss, d_optimizer

  def _gram_matrix(self, tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    gram = tf.matmul(tf.transpose(matrix), matrix)
    return gram

  def generate_image(self, num):
    pass

  def sampler(self, input_c, input_noise):
    """
    set is_train=False, reuse=True, fix param ten generate sampler

    """
    net_g = MartaGanBasicNetWork.generator(input_c=input_c, input_z=input_noise, image_size=self.image_size,
                                           is_train=False,
                                           reuse=True)
    return net_g.outputs
