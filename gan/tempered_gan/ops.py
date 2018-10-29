# -*- coding:utf-8-*-
import tensorflow as tf


def batch_normalizer(x, epsilon=1e-5, momentum=0.9, train=True, name='batch_norm', reuse=False):
    """
    :param x: input feature map
    :param epsilon:
    :param momentum:
    :param train: train or not?
    :param name:
    :param reuse: reuse or not?
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                            scale=True, is_training=train)


def instance_normalizer(x, name='instance_norm', reuse=False):
    """
    :param x:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        batch, height, width, channel = [i for i in x.shape]
        var_shape = [channel]
        # return axes 's mean and variance
        mu, sigma_sq = tf.nn.moments(x, [1, 2], keep_dims=True)
        # shift is beta, scale is alpha in in_norm form
        shift = tf.get_variable('shift', shape=var_shape, initializer=tf.zeros_initializer())
        scale = tf.get_variable('scale', shape=var_shape, initializer=tf.ones_initializer())
        epsilon = 1e-3
        normalized = (x-mu)/(sigma_sq + epsilon)**0.5
        return scale * normalized + shift


def full_connect(x, output_num, stddev=0.02, bias=0.0, name='full_connect', reuse=False):
    """
    :param x: the input feature map
    :param output_num: the output feature map size
    :param stddev:
    :param bias:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        shape = x.shape.as_list()
        w = tf.get_variable('w', [shape[1], output_num], tf.float32, tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_num], tf.float32, tf.constant_initializer(bias))
        return tf.matmul(x, w) + b


def conv2d(x, output_num, stride=2, filter_size=5, stddev=0.02, padding='SAME', name='conv2d', reuse=False):
    """
    :param x:
    :param output_num:
    :param stride:
    :param filter_size:
    :param stddev:
    :param padding:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, in_channels, output_channels]
        shape = x.shape.as_list()
        filter_shape = [filter_size, filter_size, shape[-1], output_num]
        strides_shape = [1, stride, stride, 1]
        w = tf.get_variable('w', filter_shape, tf.float32, tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_num], tf.float32, tf.constant_initializer(0.0))
        return tf.nn.bias_add(tf.nn.conv2d(x, w, strides=strides_shape, padding=padding), b)


def deconv2d(x, output_size, stride=2, filter_size=5, stddev=0.02, padding='SAME', name='deconv2d', reuse=False):
    """
    :param x:
    :param output_size:
    :param stride:
    :param filter_size:
    :param stddev:
    :param padding:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        shape = x.shape.as_list()
        filter_shape = [filter_size, filter_size, output_size[-1], shape[-1]]
        strides_shape = [1, stride, stride, 1]
        w = tf.get_variable('w', filter_shape, tf.float32, tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size[-1]], tf.float32, tf.constant_initializer(0.0))
        return tf.nn.bias_add(tf.nn.conv2d_transpose(x, filter=w, output_shape=output_size,
                                                     strides=strides_shape, padding=padding), b)


def res_block(x, name='res_block', reuse=False):
    """
    keep size, 1 * 1 + 3 * 3 + 1 * 1
    :param x:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        batch_, h_, w_, output_num = x.shape.as_list()
        x1 = conv2d(x, output_num=output_num / 2, stride=1, filter_size=1, padding='VALID', name='conv1')  # 1 * 1 conv
        x1 = lrelu(instance_normalizer(x1, name='bn1'))
        x2 = conv2d(x1, output_num=output_num / 2, stride=1, filter_size=3, padding='SAME', name='conv2')  # 3 * 3 conv
        x2 = lrelu(instance_normalizer(x2, name='bn2'))
        x3 = conv2d(x2, output_num=output_num, stride=1, filter_size=1, padding='VALID', name='conv3')  # 1 * 1 conv
        x3 = instance_normalizer(x3, name='bn3')
        return lrelu(x3 + x)


def res_block3_3(x, name='res_block3_3', reuse=False):
    """
    keep size, 3 * 3 + 3 * 3
    :param x:
    :param name:
    :param reuse:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        batch_, h_, w_, output_num = x.shape.as_list()
        x1 = conv2d(x, output_num=output_num, stride=1, filter_size=3, padding='SAME', name='conv1')  # 3 * 3 conv
        x1 = lrelu(instance_normalizer(x1, name='bn1'))
        x2 = conv2d(x1, output_num=output_num, stride=1, filter_size=3, padding='SAME', name='conv2')  # 3 * 3 conv
        x2 = instance_normalizer(x2, name='bn3')
        return lrelu(x2 + x)


def lrelu(x, leak=0.2, name='lrelu'):
    """
    :param x:
    :param leak:
    :param name:
    :return:
    """
    return tf.maximum(x, leak * x, name=name)


def resize_nn(x, resize_h, resize_w):
    """
    :param x:
    :param resize_h: the result image height
    :param resize_w: the result image width
    :return: resized image
    """
    return tf.image.resize_nearest_neighbor(x, size=(int(resize_h), int(resize_w)))