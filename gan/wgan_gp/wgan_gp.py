# -*- coding: utf-8 -*-
# https://github.com/hwalsuklee/tensorflow-generative-model-collections

import tensorflow as tf
import numpy as np
import math
import os
import scipy.misc

learning_rate = 0.0001
noise_size = 1024
input_size = [96, 96, 3]
training_epochs = 40
batch_size = 32
display_step = 512
size = 32
lamda = 10


def load_img(folder):
    image_files = os.listdir(folder)
    dataset = np.ndarray(
        shape=[len(image_files), ] + input_size,
        dtype=np.float32
    )
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        im = scipy.misc.imread(image_file).astype(np.float)
        if im.shape != input_size:
            im = resize_img(im, input_size[0], input_size[1])
        image_data = (np.array(im).astype(float) / (255.0 / 2.0) - 1)
        dataset[num_images, :, :, :] = np.reshape(image_data, newshape=input_size)
        num_images = num_images + 1
        # print(num_images)
        if num_images == 51200:
            break
    dataset = dataset[0:num_images, :, :, :]
    chunk_size = int(math.ceil(float(num_images) / float(batch_size)))
    print('Chunk_size:', chunk_size)
    print('Full dataset tensor:', dataset.shape)
    return dataset, chunk_size


def conv_out_size_same(s, stride):
    return int(math.ceil(float(s) / float(stride)))


def batch_normalizer(x, epsilon=1e-5, momentum=0.9, train=True, name='batch_norm', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon,
                                            scale=True, is_training=train)


def instance_normalizer(x, name='instance_norm', reuse=False):
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
    with tf.variable_scope(name, reuse=reuse):
        shape = x.shape.as_list()
        w = tf.get_variable('w', [shape[1], output_num], tf.float32, tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_num], tf.float32, tf.constant_initializer(bias))
        return tf.matmul(x, w) + b


def conv2d(x, output_num, stride=2, filter_size=5, stddev=0.02, padding='SAME', name='conv2d', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, in_channels, output_channels]
        shape = x.shape.as_list()
        filter_shape = [filter_size, filter_size, shape[-1], output_num]
        strides_shape = [1, stride, stride, 1]
        w = tf.get_variable('w', filter_shape, tf.float32, tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_num], tf.float32, tf.constant_initializer(0.0))
        return tf.nn.bias_add(tf.nn.conv2d(x, w, strides=strides_shape, padding=padding), b)


def deconv2d(x, output_size, stride=2, filter_size=5, stddev=0.02, padding='SAME', name='deconv2d', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, output_channels, in_channels]
        shape = x.shape.as_list()
        filter_shape = [filter_size, filter_size, output_size[-1], shape[-1]]
        strides_shape = [1, stride, stride, 1]
        w = tf.get_variable('w', filter_shape, tf.float32, tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size[-1]], tf.float32, tf.constant_initializer(0.0))
        return tf.nn.bias_add(tf.nn.conv2d_transpose(x, filter=w, output_shape=output_size,
                                                     strides=strides_shape, padding=padding), b)


# keep size
def res_block(x, train=True, name='res_block', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_, h_, w_, output_num = x.shape.as_list()
        x1 = conv2d(x, output_num=output_num, stride=1, filter_size=3, padding='SAME', name='conv1')  # 3 * 3 conv
        x1 = lrelu(batch_normalizer(x1, train=train, name='bn1'))
        x2 = conv2d(x1, output_num=output_num, stride=1, filter_size=3, padding='SAME', name='conv2')  # 3 * 3 conv
        x2 = batch_normalizer(x2, train=train, name='bn2')
        return lrelu(x2 + x)


def resize_img(img, resize_h, resize_w):
    return scipy.misc.imresize(img, [resize_h, resize_w])


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak * x, name=name)


def get_batches(data, batch_index):
    batch = data[batch_index:batch_index + batch_size, :, :, :]
    return batch


# layer height, width
s_h, s_w, _ = input_size
s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)


# generate (model 1)
def build_generator(noise, train=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # AttributeError: 'tuple' object has no attribute 'as_list
        z = full_connect(noise, output_num=size * 8 * s_h16 * s_w16, name='g_full', reuse=reuse)
        # reshape
        h0 = tf.reshape(z, [-1, s_h16, s_w16, size * 8])
        h0 = batch_normalizer(h0, train=train, name='g_bn0', reuse=reuse)
        h0 = lrelu(h0, name='g_l0')

        h1 = deconv2d(h0, output_size=[batch_size, s_h8, s_w8, size * 4], name='g_h1', reuse=reuse)
        h1 = batch_normalizer(h1, train=train, name='g_bn1', reuse=reuse)
        h1 = lrelu(h1, name='g_l1')

        h1 = res_block(h1, train=train, name='res_block0', reuse=reuse)

        h2 = deconv2d(h1, output_size=[batch_size, s_h4, s_w4, size * 2], name='g_h2', reuse=reuse)
        h2 = batch_normalizer(h2, train=train, name='g_bn2', reuse=reuse)
        h2 = lrelu(h2, name='g_l2')

        h2 = res_block(h2, train=train, name='res_block1', reuse=reuse)

        h3 = deconv2d(h2, output_size=[batch_size, s_h2, s_w2, size * 1], name='g_h3', reuse=reuse)
        h3 = batch_normalizer(h3, train=train, name='g_bn3', reuse=reuse)
        h3 = lrelu(h3, name='g_l3')

        h3 = res_block(h3, train=train, name='res_block2', reuse=reuse)

        h4 = deconv2d(h3, output_size=[batch_size, ] + input_size, name='g_h4', reuse=reuse)
        x_generate = tf.nn.tanh(h4, name='g_l4')
        return x_generate


# discriminator (model 2)
def build_discriminator(imgs, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        h0 = conv2d(imgs, output_num=size, name='d_h0', reuse=reuse)
        h0 = lrelu(h0)

        h1 = conv2d(h0, output_num=size * 2, name='d_h1', reuse=reuse)
        h1 = instance_normalizer(h1, name='d_in1', reuse=reuse)
        h1 = lrelu(h1)

        h2 = conv2d(h1, output_num=size * 4, name='d_h2', reuse=reuse)
        h2 = instance_normalizer(h2, name='d_in2', reuse=reuse)
        h2 = lrelu(h2)

        h3 = conv2d(h2, output_num=size * 8, name='d_h3', reuse=reuse)
        h3 = instance_normalizer(h3, name='d_in3', reuse=reuse)
        h3 = lrelu(h3)

        h4 = tf.reshape(h3, [batch_size, s_h16 * s_w16 * size * 8])

        y_data = full_connect(h4, output_num=1, name='d_full', reuse=reuse)
        return y_data


def generate_samples(num):
    noise_imgs = tf.placeholder(tf.float32, [None, noise_size], name='noise_imgs')
    sample_imgs = build_generator(noise_imgs, train=False, reuse=tf.AUTO_REUSE)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/ziyangcheng/python_save_file/wgan_gp/save_model/2/wgan_gp.ckpt')
        sample_noise = np.random.uniform(-1.0, 1.0, size=(num, noise_size))
        n_batch = num // batch_size
        for j in range(n_batch):
            samples = sess.run(sample_imgs,
                               feed_dict={noise_imgs: sample_noise[(j * batch_size):((j + 1) * batch_size)]})
            for i in range(len(samples)):
                print('index', j * batch_size + i)
                scipy.misc.imsave(
                    '/home/ziyangcheng/python_save_file/wgan_gp/output/generate/2/' + str(j * batch_size + i) + 'generate.png', samples[i])
    print('generate done!')


def start_train(dataset, chunk_size):
    with tf.name_scope('inputs'):
        real_imgs = tf.placeholder(tf.float32, [None, ] + input_size, name='real_images')
        noise_imgs = tf.placeholder(tf.float32, [None, noise_size], name='noise_images')

    fake_imgs = build_generator(noise_imgs, train=True, reuse=False)

    real_logits = build_discriminator(real_imgs, reuse=False)
    fake_logits = build_discriminator(fake_imgs, reuse=True)

    with tf.name_scope('loss'):
        # 1
        g_loss = - tf.reduce_mean(fake_logits)
        # loss
        # 0
        d_fake_loss = tf.reduce_mean(fake_logits)
        # loss
        # 1
        d_real_loss = - tf.reduce_mean(real_logits)
        # total d_loss
        d_loss = tf.add(d_fake_loss, d_real_loss)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('d_fake_loss', d_fake_loss)
        tf.summary.scalar('d_real_loss', d_real_loss)
        tf.summary.scalar('d_loss', d_loss)
    with tf.name_scope('optimizer'):
        # gradient penalty
        rand = tf.random_uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        inter_img = real_imgs * rand + (1 - rand) * fake_imgs
        gradiants = tf.gradients(build_discriminator(inter_img, reuse=True), [inter_img])[0]
        g_norm = tf.sqrt(tf.reduce_sum(tf.square(gradiants), axis=1))
        gradiants_penalty = lamda * tf.reduce_mean(tf.multiply((g_norm - 1), (g_norm - 1)))
        d_loss = d_loss + gradiants_penalty

        # optimizer
        train_vars = tf.trainable_variables()
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]

        d_trainer = tf.train.RMSPropOptimizer(learning_rate).minimize(d_loss, var_list=dis_vars)
        g_trainer = tf.train.RMSPropOptimizer(learning_rate).minimize(g_loss, var_list=gen_vars)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        # merge summary
        merged = tf.summary.merge_all()
        # choose dir
        writer = tf.summary.FileWriter('/home/ziyangcheng/python_save_file/wgan_gp/tf_board', sess.graph)
        batch_index = 0  # init index
        sess.run(tf.global_variables_initializer())
        for e in range(training_epochs):
            for batch_i in range(chunk_size):
                batch_data = get_batches(dataset, batch_index)
                batch_index = (batch_index + batch_size) % ((chunk_size - 1) * batch_size)

                # noise
                noise = np.random.uniform(-1.0, 1.0, size=(batch_size, noise_size)).astype(np.float32)

                # Run optimizers
                sess.run(d_trainer, feed_dict={real_imgs: batch_data, noise_imgs: noise})
                sess.run(d_trainer, feed_dict={real_imgs: batch_data, noise_imgs: noise})

                sess.run(g_trainer, feed_dict={noise_imgs: noise})
                check_imgs, _ = sess.run([fake_imgs, g_trainer], feed_dict={noise_imgs: noise})

                if (chunk_size * e + batch_i) % display_step == 0:
                    train_loss_d = sess.run(d_loss, feed_dict={real_imgs: batch_data, noise_imgs: noise})
                    fake_loss_d = sess.run(d_fake_loss, feed_dict={noise_imgs: noise})
                    real_loss_d = sess.run(d_real_loss, feed_dict={real_imgs: batch_data})
                    # generator loss
                    train_loss_g = sess.run(g_loss, feed_dict={noise_imgs: noise})

                    merge_result = sess.run(merged, feed_dict={real_imgs: batch_data, noise_imgs: noise})
                    # merge_result = sess.run(merged, feed_dict={X: batch_xs})
                    writer.add_summary(merge_result, chunk_size * e + batch_i)

                    print("step {}/of epoch {}/{}...".format(chunk_size * e + batch_i, e,training_epochs),
                          "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(
                              train_loss_d,real_loss_d,fake_loss_d), "Generator Loss: {:.4f}".format(train_loss_g))

                    # save pic
                    scipy.misc.imsave('/home/ziyangcheng/python_save_file/wgan_gp/output/train/' +
                                      str(chunk_size * e + batch_i) + '-' + str(0) + '.png', check_imgs[0])

        print('train done')
        # save sess
        saver.save(sess, '/home/ziyangcheng/python_save_file/wgan_gp/save_model/2/wgan_gp.ckpt')


if __name__ == '__main__':
    data_folder = '/home/ziyangcheng/datasets/faces'
    dataimgs, chunk_s = load_img(data_folder)
    start_train(dataimgs, chunk_s)
    # generate_samples(100)
