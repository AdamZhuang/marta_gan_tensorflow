import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


class MartaGanBasicNetWork:
  def __init__(self):
    pass

  @staticmethod
  def generator(input_data, image_size, is_train=True, reuse=False):
    # filter_size
    k = 4
    # n_filter
    n_filter = 16

    # init weight
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    # build network
    with tf.variable_scope("generator", reuse=reuse):
      net_in = InputLayer(input_data, name='g/in')

      net_h0 = DenseLayer(prev_layer=net_in, n_units=n_filter * 32 * int(image_size / 64) * int(image_size / 64),
                          W_init=w_init,
                          act=tf.identity, name='g/h0/lin')
      net_h0 = ReshapeLayer(prev_layer=net_h0, shape=[-1, int(image_size / 64), int(image_size / 64), n_filter * 32],
                            name='g/h0/reshape')
      net_h0 = BatchNormLayer(prev_layer=net_h0, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                              name='g/h0/batch_norm')

      net_h1 = DeConv2d(prev_layer=net_h0, n_filter=n_filter * 16, filter_size=(k, k), strides=(2, 2), padding='SAME',
                        W_init=w_init, name='g/h1/deconv2d')
      net_h1 = BatchNormLayer(prev_layer=net_h1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                              name='g/h1/batch_norm')

      net_h2 = DeConv2d(prev_layer=net_h1, n_filter=n_filter * 8, filter_size=(k, k), strides=(2, 2), padding='SAME',
                        W_init=w_init, name='g/h2/deconv2d')
      net_h2 = BatchNormLayer(prev_layer=net_h2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                              name='g/h2/batch_norm')

      net_h3 = DeConv2d(prev_layer=net_h2, n_filter=n_filter * 4, filter_size=(k, k), strides=(2, 2), padding='SAME',
                        W_init=w_init, name='g/h3/deconv2d')
      net_h3 = BatchNormLayer(prev_layer=net_h3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                              name='g/h3/batch_norm')

      net_h4 = DeConv2d(prev_layer=net_h3, n_filter=n_filter * 2, filter_size=(k, k), strides=(2, 2), padding='SAME',
                        W_init=w_init, name='g/h4/deconv2d')
      net_h4 = BatchNormLayer(prev_layer=net_h4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                              name='g/h4/batch_norm')

      net_h5 = DeConv2d(prev_layer=net_h4, n_filter=n_filter * 1, filter_size=(k, k), strides=(2, 2), padding='SAME',
                        W_init=w_init, name='g/h5/deconv2d')
      net_h5 = BatchNormLayer(prev_layer=net_h5, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init,
                              name='g/h5/batch_norm')

      net_h6 = DeConv2d(prev_layer=net_h5, n_filter=3, filter_size=(k, k), strides=(2, 2), padding='SAME',
                        W_init=w_init, name='g/h6/deconv2d')

      net_h6.outputs = tf.nn.tanh(net_h6.outputs)

    return net_h6

  @staticmethod
  def feature_extract_layer(input_data, is_train=True, reuse=False):
    # filter size
    k = 5
    # n_filter
    n_filter = 16

    # init weight
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    # build network
    with tf.variable_scope("discriminator", reuse=reuse):
      net_in = InputLayer(input_data, name='d/in')

      net_h0 = Conv2d(prev_layer=net_in, n_filter=n_filter * 1, filter_size=(k, k), strides=(2, 2),
                      act=lambda x: tf.nn.leaky_relu(x, 0.2), padding='SAME', W_init=w_init, name='d/h0/conv2d')

      net_h1 = Conv2d(prev_layer=net_h0, n_filter=n_filter * 2, filter_size=(k, k), strides=(2, 2), padding='SAME',
                      W_init=w_init, name='d/h1/conv2d')
      net_h1 = BatchNormLayer(prev_layer=net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                              is_train=is_train, gamma_init=gamma_init, name='d/h1/batch_norm')

      net_h2 = Conv2d(prev_layer=net_h1, n_filter=n_filter * 4, filter_size=(k, k), strides=(2, 2), padding='SAME',
                      W_init=w_init, name='d/h2/conv2d')
      net_h2 = BatchNormLayer(prev_layer=net_h2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                              gamma_init=gamma_init, name='d/h2/batch_norm')

      net_h3 = Conv2d(prev_layer=net_h2, n_filter=n_filter * 8, filter_size=(k, k), strides=(2, 2), padding='SAME',
                      W_init=w_init, name='d/h3/conv2d')
      net_h3 = BatchNormLayer(prev_layer=net_h3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                              gamma_init=gamma_init, name='d/h3/batch_norm')

      net_h4 = Conv2d(prev_layer=net_h3, n_filter=n_filter * 16, filter_size=(k, k), strides=(2, 2), padding='SAME',
                      W_init=w_init, name='d/h4/conv2d')
      net_h4 = BatchNormLayer(prev_layer=net_h4, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                              gamma_init=gamma_init, name='d/h4/batch_norm')

      net_h5 = Conv2d(prev_layer=net_h4, n_filter=n_filter * 32, filter_size=(k, k), strides=(2, 2), act=None,
                      padding='SAME', W_init=w_init, name='d/h5/conv2d')
      net_h5 = BatchNormLayer(prev_layer=net_h5, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                              gamma_init=gamma_init, name='d/h5/batch_norm')

      global_max1 = MaxPool2d(prev_layer=net_h3, filter_size=(4, 4), strides=None, padding='SAME', name='maxpool1')
      global_max1 = FlattenLayer(prev_layer=global_max1, name='d/h3/flatten')
      global_max2 = MaxPool2d(prev_layer=net_h4, filter_size=(2, 2), strides=None, padding='SAME', name='maxpool2')
      global_max2 = FlattenLayer(prev_layer=global_max2, name='d/h4/flatten')
      global_max3 = FlattenLayer(prev_layer=net_h5, name='d/h5/flatten')

      # multi-feature
      feature = ConcatLayer([global_max1, global_max2, global_max3], name='d/concat_layer1')
      # dense layer
      net_h6 = DenseLayer(prev_layer=feature, n_units=1, act=tf.identity, W_init=w_init, name='d/h6/lin_sigmoid')
      # origin output (tensor)
      logits = net_h6.outputs
      # output net (tensorlayer)
      net_h6.outputs = tf.nn.sigmoid(net_h6.outputs)
      # style features
      style_features = {
        "net_h1": net_h1.outputs,
        "net_h2": net_h2.outputs,
        "net_h3": net_h3.outputs,
        "net_h4": net_h4.outputs,
        "net_h5": net_h5.outputs,
      }
      # latent code layer
      latent_code_layer = DenseLayer(prev_layer=feature, n_units=21, act=tf.identity, W_init=w_init,
                                     name='d/latent_code_layer')
      latent_code_layer_output = tf.nn.softmax(latent_code_layer.outputs)

    return net_h6, logits, feature.outputs, style_features, latent_code_layer_output
