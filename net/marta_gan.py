import os
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import logging
from glob import glob
from random import shuffle
from net.basic_network import MartaGanBasicNetWork
from utils.utils import Utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MartaGan:
  def __init__(self,
               dataset_path="./dataset/uc_test_256",
               checkpoint_path="./checkpoint",
               sample_path="./sample",
               image_size=256,
               image_dim=3,
               input_noise_dim=100,
               learning_rate=0.001,
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
    # random noise for generator
    self.input_noise = tf.placeholder(tf.float32, [self.batch_size, self.input_noise_dim], name='noise')
    # the fake dataset of generating
    self.net_g = MartaGanBasicNetWork.generator(input_data=self.input_noise, image_size=self.image_size, is_train=True,
                                                reuse=False)
    # feature of fake images
    self.net_feature_extract_fake, self.logits_fake, self.feature_fake = MartaGanBasicNetWork.feature_extract_layer(
      input_data=self.net_g.outputs,
      is_train=True,
      reuse=False)

    #####################
    # for real images
    #####################
    # real image placeholder (the dataset will feed from dataset)
    self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.image_dim],
                                      name='real_images')
    # feature of real images
    self.net_feature_extract_real, self.logits_real, self.feature_real = MartaGanBasicNetWork.feature_extract_layer(
      input_data=self.real_images,
      is_train=True,
      reuse=True)

    #####################
    # sample
    #####################
    self.sample_image = self.sampler(self.input_noise)

    #####################
    # loss and optimizer
    #####################
    self.g_loss, self.g_optimizer = self.generator(self.net_g, self.logits_fake, self.feature_real, self.feature_fake)
    self.d_loss, self.d_optimizer = self.discriminator(self.net_feature_extract_fake, self.logits_real,
                                                       self.logits_fake)

  def load_parma(self, sess, net_g, net_d, load_epoch):
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
      self.load_parma(sess, self.net_g, self.net_feature_extract_fake, load_epoch)
      for cur_epoch in range(load_epoch + 1, epoch):
        # every epoch shuffle data_files
        shuffle(data_files)
        # get total batch
        total_batch = int(len(data_files) / self.batch_size)
        # every batch get a random input noise
        input_noise = np.random.uniform(low=-1, high=1, size=(self.batch_size, self.input_noise_dim)).astype(
          np.float32)
        for cur_batch in range(total_batch):
          # get one batch of real images
          batch_files = data_files[cur_batch * self.batch_size:(cur_batch + 1) * self.batch_size]
          batch_images = [Utils.get_image(batch_file, self.image_size, resize_w=self.image_size,
                                          is_grayscale=0) for batch_file in batch_files]
          batch_real_images = np.array(batch_images).astype(np.float32)

          # train d
          d_loss, _ = sess.run([self.d_loss, self.d_optimizer],
                               feed_dict={self.input_noise: input_noise, self.real_images: batch_real_images})

          # train g
          for _ in range(2):
            g_loss, _ = sess.run([self.g_loss, self.g_optimizer],
                                 feed_dict={self.input_noise: input_noise, self.real_images: batch_real_images})
          logging.info("epoch:[%4d/%4d], batch:[%4d/%4d], d_loss: %.8f, g_loss: %.8f",
                       cur_epoch, epoch, cur_batch + 1, total_batch, d_loss, g_loss)

        if cur_epoch % 10 == 0:
          # save images
          for sample_image_num in range(5):
            # a new input noise
            input_noise = np.random.uniform(low=-1, high=1, size=(self.batch_size, self.input_noise_dim)).astype(
              np.float32)
            images = sess.run(self.sample_image, feed_dict={self.input_noise: input_noise,
                                                            self.real_images: batch_real_images})
            # save images
            side = 1
            while side * side < self.batch_size:
              side += 1
            Utils.save_images(images, [side, side],
                              os.path.join(self.sample_path,
                                           "epoch{}-sample{}.png".format(str(cur_epoch), str(sample_image_num))))
          logging.info("sample image saved!")

          # save net param
          g_vars = self.net_g.all_params
          d_vars = self.net_feature_extract_fake.all_params
          g_vars_name = os.path.join(self.checkpoint_path, "g_vars_{}.npz".format(str(cur_epoch)))
          d_vars_name = os.path.join(self.checkpoint_path, "d_vars_{}.npz".format(str(cur_epoch)))
          tl.files.save_npz(g_vars, name=g_vars_name, sess=sess)
          tl.files.save_npz(d_vars, name=d_vars_name, sess=sess)

  def generator(self, net_g, logits_fake, feature_real, feature_fake):
    """
    the generator should gen image that is smiler with real image

    the loss contains 2 part
    g_loss1: high-dim feature match loss
    g_loss2: high-dim feature perceptual loss

    """
    # loss of generator
    g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,
                                                                     labels=tf.ones_like(logits_fake)))
    g_loss2 = tf.reduce_mean(tf.nn.l2_loss(feature_real - feature_fake)) / (self.image_size * self.image_size)
    g_loss = g_loss1 + g_loss2

    # optimizer of generator
    g_vars = net_g.all_params
    # self.net_g.print_params(False)
    g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(g_loss, var_list=g_vars)

    return g_loss, g_optimizer

  def discriminator(self, net_feature_extract_fake, logits_real, logits_fake):
    """
    the discriminator should marked real image as 1 and mark fake image as 0

    """
    # loss of discriminator
    d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real)))  # real == 1
    d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake)))  # fake == 0
    d_loss = d_loss_real + d_loss_fake

    # optimizer of discriminator
    d_vars = net_feature_extract_fake.all_params
    d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(d_loss, var_list=d_vars)

    return d_loss, d_optimizer

  def sampler(self, input_data):
    """
    set is_train=False, reuse=True, fix param ten generate sampler

    """
    net_g = MartaGanBasicNetWork.generator(input_data=input_data, image_size=self.image_size, is_train=False,
                                           reuse=True)
    return net_g.outputs

  def generate_image(self, num):
    pass
