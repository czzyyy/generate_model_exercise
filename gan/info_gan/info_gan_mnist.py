# -*- coding: utf-8 -*-
# https://github.com/hwalsuklee/tensorflow-generative-model-collections

import tensorflow as tf
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import scipy.misc

from tensorflow.examples.tutorials.mnist import input_data


class InfoGan(object):
    def __init__(self, learning_rate, noise_size, input_size,
                 training_epochs, batch_size, display_step, c_class_num=None, c_code_dim=None, lamda=1, size=32,
                 path=None):
        self.train_data = None
        self.learning_rate = learning_rate
        self.noise_size = noise_size
        self.input_size = input_size  # [h , w , c]
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.display_step = display_step
        self.chunk_size = None
        self.c_class_dim = c_class_num
        self.c_code_dim = c_code_dim
        self.lamda = lamda
        self.size = size
        self.batch_index = 0
        self.data_path = path

    @staticmethod
    def conv_out_size_same(size, stride):
        return int(math.ceil(float(size) / float(stride)))

    @staticmethod
    def full_connect(x, output_size, stddev=0.02, bias=0.0, name='full_connect'):
        with tf.variable_scope(name):
            shape = x.shape.as_list()
            w = tf.get_variable('w', [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [output_size],  tf.float32, tf.constant_initializer(bias))
            return tf.matmul(x, w) + b

    @staticmethod
    def batch_normalizer(x, epsilon=1e-5, momentum=0.9, train=True, name='batch_norm', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                                scale=True, is_training=train)

    @staticmethod
    def conv2d(x, output_size, stddev=0.02, name='conv2d'):
        with tf.variable_scope(name):
            # filter : [height, width, in_channels, output_channels]
            shape = x.shape.as_list()
            filter_shape = [5, 5, shape[-1], output_size]
            strides_shape = [1, 2, 2, 1]
            w = tf.get_variable('w', filter_shape, tf.float32, tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [output_size], tf.float32, tf.constant_initializer(0.0))
            return tf.nn.bias_add(tf.nn.conv2d(x, w, strides=strides_shape, padding='SAME'), b)

    @staticmethod
    def deconv2d(x, output_size, stddev=0.02, name='deconv2d'):
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            shape = x.shape.as_list()
            filter_shape = [5, 5, output_size[-1], shape[-1]]
            strides_shape = [1, 2, 2, 1]
            w = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable('b', [output_size[-1]], tf.float32, tf.constant_initializer(0.0))
            return tf.nn.bias_add(tf.nn.conv2d_transpose(x, filter=w, output_shape=output_size,
                                                         strides=strides_shape, padding='SAME'), b)

    @staticmethod
    def resize_img(img, resize_h, resize_w):
        return scipy.misc.imresize(img, [resize_h, resize_w])

    @staticmethod
    def lrelu(x, leak=0.2):
        return tf.maximum(x, leak * x)

    def load_mnist(self):
        mnist = input_data.read_data_sets("/home/ziyangcheng/datasets/mnist/", one_hot=True)
        self.train_data = mnist.train
        self.chunk_size = int(math.ceil(float(len(mnist.train.images)) / float(self.batch_size)))
        print('Chunk_size(mnist):', self.chunk_size)
        print('Full dataset tensor(mnist):', self.train_data.images.shape)
        # batch_xs, batch_ys = self.train_data.next_batch(self.batch_size)

    def load_img(self, folder):
        image_files = os.listdir(folder)
        dataset = np.ndarray(
            shape=[len(image_files), ] + self.input_size,
            dtype=np.float32
        )
        num_images = 0
        for image in image_files:
            image_file = os.path.join(folder, image)
            im = scipy.misc.imread(image_file).astype(np.float)
            if im.shape != self.input_size:
                im = self.resize_img(im, self.input_size[0], self.input_size[1])
            image_data = (np.array(im).astype(float) / (255.0 / 2.0) - 1)
            dataset[num_images, :, :, :] = np.reshape(image_data, newshape=self.input_size)
            num_images = num_images + 1
            if num_images == 51200:
                break
        self.train_data = dataset[0:num_images, :, :, :]
        self.chunk_size = int(math.ceil(float(num_images) / float(self.batch_size)))
        print('Chunk_size:', self.chunk_size)
        print('Full dataset tensor:', self.train_data.shape)
        return dataset, self.chunk_size

    def get_batches(self):
        batch = self.train_data[self.batch_index:self.batch_index + self.batch_size, :, :, :]
        self.batch_index = (self.batch_index + self.batch_size) % ((self.chunk_size - 1) * self.batch_size)
        return batch

    # generate (model 1)
    def build_generator(self, noise, train=True, ismnist=False, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            s_h, s_w, _ = self.input_size
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)

            z = self.full_connect(noise, self.size * 8 * s_h16 * s_w16, name='g_full')
            # reshape
            h0 = tf.reshape(z, [-1, s_h16, s_w16, self.size * 8])
            h0 = self.batch_normalizer(h0, train=train, name='g_bn0', reuse=reuse)
            h0 = tf.nn.relu(h0, name='g_l0')

            h1 = self.deconv2d(h0, output_size=[self.batch_size, s_h8, s_w8, self.size * 4], name='g_h1')
            h1 = self.batch_normalizer(h1, train=train, name='g_bn1', reuse=reuse)
            h1 = tf.nn.relu(h1, name='g_l1')

            h2 = self.deconv2d(h1, output_size=[self.batch_size, s_h4, s_w4, self.size * 2], name='g_h2')
            h2 = self.batch_normalizer(h2, train=train, name='g_bn2', reuse=reuse)
            h2 = tf.nn.relu(h2, name='g_l2')

            h3 = self.deconv2d(h2, output_size=[self.batch_size, s_h2, s_w2, self.size * 1], name='g_h3')
            h3 = self.batch_normalizer(h3, train=train, name='g_bn3', reuse=reuse)
            h3 = tf.nn.relu(h3, name='g_l3')

            h4 = self.deconv2d(h3, output_size=[self.batch_size, ] + self.input_size, name='g_h4')
            if ismnist:
                x_generate = tf.nn.sigmoid(h4, name='g_s4')
            else:
                x_generate = tf.nn.tanh(h4, name='g_t4')

            return x_generate

    # discriminator (model 2)
    def build_discriminator(self, imgs, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            s_h, s_w, _ = self.input_size
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)
            h0 = self.conv2d(imgs, self.size, name='d_h0')
            h0 = self.lrelu(h0)

            h1 = self.conv2d(h0, self.size * 2, name='d_h1')
            h1 = self.batch_normalizer(h1, name='d_bn1', reuse=reuse)
            h1 = self.lrelu(h1)

            h2 = self.conv2d(h1, self.size * 4, name='d_h2')
            h2 = self.batch_normalizer(h2, name='d_bn2', reuse=reuse)
            h2 = self.lrelu(h2)

            h3 = self.conv2d(h2, self.size * 8, name='d_h3')
            h3 = self.batch_normalizer(h3, name='d_bn3', reuse=reuse)
            h3 = self.lrelu(h3)

            # share h4
            h4 = tf.reshape(h3, [self.batch_size, s_h16 * s_w16 * self.size * 8])

            # categorical one-hot label
            c_class = tf.nn.softmax(self.full_connect(h4, self.c_class_dim, name='c_class'))
            # continuous var stddev & mean
            c_code_stddev = self.full_connect(h4, self.c_code_dim, name='c_code_stddev')
            c_code_mean = self.full_connect(h4, self.c_code_dim, name='c_code_mean')
            # real or fake for GAN
            y_data = self.full_connect(h4, 1, name='d_full')

            return y_data, c_class, c_code_stddev, c_code_mean

    def generate_samples(self, num):
        noise_imgs = tf.placeholder(tf.float32, [None, self.noise_size], name='noise_imgs')
        c_class_input = tf.placeholder(tf.float32, [None, self.c_class_dim], name='c_class_input')
        c_code_input = tf.placeholder(tf.float32, [None, self.c_code_dim], name='c_code_input')
        total_noise = tf.concat([noise_imgs, c_class_input, c_code_input], axis=1)

        sample_imgs = self.build_generator(total_noise, ismnist=True, train=False, reuse=tf.AUTO_REUSE)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, '/home/ziyangcheng/python_save_file/info_gan/save_model/1/info_gan.ckpt')
            sample_noise = np.random.uniform(-1.0, 1.0, size=(num, self.noise_size)).astype(np.float32)
            sample_class = np.zeros(shape=(num, self.c_class_dim)).astype(np.float32)
            sample_class[:, 0] = 1.0
            sample_code = np.zeros(shape=(num, self.c_code_dim)).astype(np.float32)
            n_batch = num // self.batch_size
            for j in range(n_batch):
                samples = sess.run(sample_imgs,
                                   feed_dict={
                                       noise_imgs: sample_noise[(j * self.batch_size):((j + 1) * self.batch_size)],
                                       c_class_input: sample_class[(j * self.batch_size):((j + 1) * self.batch_size)],
                                       c_code_input: sample_code[(j * self.batch_size):((j + 1) * self.batch_size)]})
                for i in range(len(samples)):
                    print('index', j * self.batch_size + i)
                    scipy.misc.imsave(
                        '/home/ziyangcheng/python_save_file/info_gan/output/generate/1/' + str(
                            j * self.batch_size + i) + 'generate.png', samples[i][:, :, 0])
        print('generate done!')

    def format_generate(self):
        noise_imgs = tf.placeholder(tf.float32, [None, self.noise_size], name='noise_imgs')
        c_class_input = tf.placeholder(tf.float32, [None, self.c_class_dim], name='c_class_input')
        c_code_input = tf.placeholder(tf.float32, [None, self.c_code_dim], name='c_code_input')
        total_noise = tf.concat([noise_imgs, c_class_input, c_code_input], axis=1)

        sample_imgs = self.build_generator(total_noise, ismnist=True, train=False, reuse=tf.AUTO_REUSE)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, '/home/ziyangcheng/python_save_file/info_gan/save_model/1/info_gan.ckpt')
            sample_code = np.zeros(shape=(self.batch_size, self.c_code_dim)).astype(np.float32)
            f, a = plt.subplots(10, 10, figsize=(10, 10))
            for j in range(10):
                sample_noise = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.noise_size)).astype(np.float32)
                sample_class = np.zeros(shape=(self.batch_size, self.c_class_dim)).astype(np.float32)
                sample_class[:, j] = 1.0
                samples = sess.run(sample_imgs,
                                   feed_dict={noise_imgs: sample_noise, c_class_input: sample_class,
                                              c_code_input: sample_code})
                for i in range(10):
                    a[j][i].imshow(np.reshape(samples[i][:, :, 0], (28, 28)), cmap="gray")
            f.savefig('/home/ziyangcheng/python_save_file/info_gan/output/generate/1/format_one.png')

            f1, a1 = plt.subplots(10, 10, figsize=(10, 10))
            for j in range(10):
                sample_class = np.zeros(shape=(self.batch_size, self.c_class_dim)).astype(np.float32)
                sample_class[:, j] = 1.0

                for i in range(10):
                    sample_code = np.zeros(shape=(self.batch_size, self.c_code_dim)).astype(np.float32)
                    sample_code[:, 0] = -1 + 0.2 * i
                    sample_noise = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.noise_size)).astype(
                        np.float32)
                    samples = sess.run(sample_imgs,
                                       feed_dict={noise_imgs: sample_noise, c_class_input: sample_class,
                                                  c_code_input: sample_code})
                    a1[j][i].imshow(np.reshape(samples[0][:, :, 0], (28, 28)), cmap="gray")
            f1.savefig('/home/ziyangcheng/python_save_file/info_gan/output/generate/1/format_two.png')

            f2, a2 = plt.subplots(10, 10, figsize=(10, 10))
            for j in range(10):
                sample_class = np.zeros(shape=(self.batch_size, self.c_class_dim)).astype(np.float32)
                sample_class[:, j] = 1.0

                for i in range(10):
                    sample_code = np.zeros(shape=(self.batch_size, self.c_code_dim)).astype(np.float32)
                    sample_code[:, 1] = -1 + 0.2 * i
                    sample_noise = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.noise_size)).astype(
                        np.float32)
                    samples = sess.run(sample_imgs,
                                       feed_dict={noise_imgs: sample_noise, c_class_input: sample_class,
                                                  c_code_input: sample_code})
                    a2[j][i].imshow(np.reshape(samples[0][:, :, 0], (28, 28)), cmap="gray")
            f2.savefig('/home/ziyangcheng/python_save_file/info_gan/output/generate/1/format_three.png')

            f3, a3 = plt.subplots(10, 10, figsize=(10, 10))
            sample_class = np.zeros(shape=(self.batch_size, self.c_class_dim)).astype(np.float32)
            sample_class[:, 8] = 1.0
            for j in range(10):
                sample_code = np.zeros(shape=(self.batch_size, self.c_code_dim)).astype(np.float32)
                sample_code[:, 0] = -1 + 0.2 * j

                for i in range(10):
                    sample_code[:, 1] = -1 + 0.2 * i
                    sample_noise = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.noise_size)).astype(
                        np.float32)
                    samples = sess.run(sample_imgs,
                                       feed_dict={noise_imgs: sample_noise, c_class_input: sample_class,
                                                  c_code_input: sample_code})
                    a3[j][i].imshow(np.reshape(samples[0][:, :, 0], (28, 28)), cmap="gray")
            f3.savefig('/home/ziyangcheng/python_save_file/info_gan/output/generate/1/format_four.png')

            print('generate done!')

    def train(self):
        if self.data_path is None:
            self.load_mnist()
        else:
            self.load_img(self.data_path)
        with tf.name_scope('inputs'):
            real_imgs = tf.placeholder(tf.float32, [None, ] + self.input_size, name='real_imgs')
            noise_imgs = tf.placeholder(tf.float32, [None, self.noise_size], name='noise_imgs')
            c_class_input = tf.placeholder(tf.float32, [None, self.c_class_dim], name='c_class_input')
            c_code_input = tf.placeholder(tf.float32, [None, self.c_code_dim], name='c_code_input')

        total_noise = tf.concat([noise_imgs, c_class_input, c_code_input], axis=1)
        fake_imgs = self.build_generator(total_noise, ismnist=True)

        real_logits, _, _, _ = self.build_discriminator(real_imgs)
        fake_logits, fake_c_class, fake_c_stddev, fake_c_mean = self.build_discriminator(fake_imgs, reuse=True)

        with tf.name_scope('loss'):
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_logits, labels=tf.ones_like(fake_logits)))
            d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_logits, labels=tf.zeros_like(fake_logits)))
            d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=real_logits, labels=tf.ones_like(real_logits)))
            d_loss = d_fake_loss + d_real_loss

            # c_loss
            c_class_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_c_class, labels=c_class_input))

            # sample
            samples = tf.random_uniform([self.batch_size, self.c_code_dim], -1, 1, dtype=tf.float32)
            c_code_fake = tf.add(tf.multiply(samples, fake_c_stddev), fake_c_mean, name='c_code_fake')
            c_code_loss = tf.reduce_mean(tf.multiply((c_code_input - c_code_fake), (c_code_input - c_code_fake))) / 2

            c_loss = self.lamda * (c_code_loss + c_class_loss)

            tf.summary.scalar('g_loss', g_loss)
            tf.summary.scalar('c_loss', c_loss)
            tf.summary.scalar('d_fake_loss', d_fake_loss)
            tf.summary.scalar('d_real_loss', d_real_loss)
            tf.summary.scalar('d_loss', d_loss)
        with tf.name_scope('optimizer'):
            train_vars = tf.trainable_variables()
            gen_vars = [var for var in train_vars if var.name.startswith('generator')]
            dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
            # c
            c_vars = [var for var in train_vars if var.name.startswith('discriminator') or
                      var.name.startswith('generator')]

            # global_step = tf.Variable(0, trainable=False)
            # rate = tf.train.exponential_decay(self.learning_rate, global_step, 256, 0.85, staircase=True)
            # rate = tf.maximum(rate, 0.00002)
            d_trainer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(d_loss, var_list=dis_vars)
            g_trainer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(g_loss, var_list=gen_vars)
            c_trainer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(c_loss, var_list=c_vars)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            # merge summary
            merged = tf.summary.merge_all()
            # choose dir
            writer = tf.summary.FileWriter('/home/ziyangcheng/python_save_file/info_gan/tf_board/', sess.graph)
            sess.run(tf.global_variables_initializer())
            for e in range(self.training_epochs):
                for batch_i in range(self.chunk_size):
                    # mnist
                    if self.data_path is None:
                        batch_data, batch_ys = self.train_data.next_batch(self.batch_size)
                        batch_data = np.reshape(batch_data, newshape=[-1, ] + self.input_size)
                    else:
                        # not mnist
                        batch_data = self.get_batches()

                    # noise
                    noise = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.noise_size)).astype(np.float32)

                    # c inputs
                    c_class = np.random.multinomial(1, self.c_class_dim * [float(1.0 / self.c_class_dim)],
                                                    size=[self.batch_size])
                    c_code = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.c_code_dim)).astype(np.float32)

                    # Run optimizers
                    sess.run(d_trainer, feed_dict={real_imgs: batch_data, noise_imgs: noise, c_class_input: c_class,
                                                   c_code_input: c_code})
                    sess.run(g_trainer, feed_dict={noise_imgs: noise, c_class_input: c_class, c_code_input: c_code})
                    check_imgs, _ = sess.run([fake_imgs, c_trainer],
                                             feed_dict={noise_imgs: noise, c_class_input: c_class,
                                                        c_code_input: c_code})

                    if (self.chunk_size * e + batch_i) % self.display_step == 0:
                        train_loss_d = sess.run(d_loss, feed_dict={real_imgs: batch_data, noise_imgs: noise,
                                                                   c_class_input: c_class, c_code_input: c_code})
                        fake_loss_d = sess.run(d_fake_loss, feed_dict={noise_imgs: noise, c_class_input: c_class,
                                                                       c_code_input: c_code})
                        real_loss_d = sess.run(d_real_loss, feed_dict={real_imgs: batch_data})
                        # generator loss
                        train_loss_g = sess.run(g_loss, feed_dict={noise_imgs: noise, c_class_input: c_class,
                                                                   c_code_input: c_code})
                        # c loss
                        train_loss_c = sess.run(c_loss, feed_dict={noise_imgs: noise, c_class_input: c_class,
                                                                   c_code_input: c_code})

                        merge_result = sess.run(merged, feed_dict={real_imgs: batch_data, noise_imgs: noise,
                                                                   c_class_input: c_class, c_code_input: c_code})

                        writer.add_summary(merge_result, self.chunk_size * e + batch_i)

                        print(
                            "step {}/of epoch {}/{}...".format(self.chunk_size * e + batch_i, e, self.training_epochs),
                            "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(
                                train_loss_d, real_loss_d, fake_loss_d), "Generator Loss: {:.4f}".format(train_loss_g),
                            "C Loss: {:.4f}".format(train_loss_c))

                        # show pic
                        scipy.misc.imsave('/home/ziyangcheng/python_save_file/info_gan/output/train/' + str(
                            self.chunk_size * e + batch_i) +
                                          '-' + str(0) + '.png', check_imgs[0][:, :, 0])

            print('train done')
            # save sess
            saver.save(sess, '/home/ziyangcheng/python_save_file/info_gan/save_model/1/info_gan.ckpt')


if __name__ == '__main__':
    data_folder = '/home/ziyangcheng/datasets/faces'
    infogan = InfoGan(learning_rate=0.01, noise_size=200, input_size=[28, 28, 1], training_epochs=30, batch_size=64,
                      display_step=512, c_class_num=10, c_code_dim=2, lamda=1, size=24)
    infogan.train()
    # infogan.format_generate()
    # infogan.generate_samples(100)
