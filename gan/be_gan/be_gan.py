# -*- coding: utf-8 -*-
# https://github.com/czzyyy/BEGAN-tensorflow-1
# https://github.com/hwalsuklee/tensorflow-generative-model-collections

import tensorflow as tf
import numpy as np
import math
import os
import scipy.misc
import out_image_data_load as image_load
learning_rate = 0.0002
noise_size = 512
input_size = [96, 96, 3]
training_epochs = 30
batch_size = 32
display_step = 1024
delay_step = 1024*10
size = 32
gamma = 0.4
lamda_k = 0.001

# faces data & lsun data


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


def get_batches(data, batch_index):
    batch = data[batch_index:batch_index + batch_size, :, :, :]
    return batch


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
        w = tf.get_variable('w', [shape[1], output_num], tf.float32, tf.random_normal_initializer(stddev=stddev))
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
        w = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_size[-1]], tf.float32, tf.constant_initializer(0.0))
        return tf.nn.bias_add(tf.nn.conv2d_transpose(x, filter=w, output_shape=output_size,
                                                     strides=strides_shape, padding=padding), b)


def resize_nn(x, resize_h, resize_w):
    return tf.image.resize_nearest_neighbor(x, size=(int(resize_h), int(resize_w)))


# keep size
def res_block(x, train=True, name='res_block', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_, h_, w_, output_num = x.shape.as_list()
        x1 = conv2d(x, output_num=output_num / 2, stride=1, filter_size=1, padding='VALID', name='conv1')  # 1 * 1 conv
        x1 = tf.nn.elu(batch_normalizer(x1, train=train, name='bn1'))
        x2 = conv2d(x1, output_num=output_num / 2, stride=1, filter_size=3, padding='SAME', name='conv2')  # 3 * 3 conv
        x2 = tf.nn.elu(batch_normalizer(x2, train=train, name='bn2'))
        x3 = conv2d(x2, output_num=output_num / 2, stride=1, filter_size=3, padding='SAME', name='conv3')  # 3 * 3 conv
        x3 = tf.nn.elu(batch_normalizer(x3, train=train, name='bn3'))
        x4 = conv2d(x3, output_num=output_num, stride=1, filter_size=1, padding='VALID', name='conv4')  # 1 * 1 conv
        x4 = batch_normalizer(x4, train=train, name='bn4')
        return tf.nn.elu(x4 + x)


# keep size
def res_block_instance_norm(x, name='res_block_instance_norm', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_, h_, w_, output_num = x.shape.as_list()
        x1 = conv2d(x, output_num=output_num, stride=1, filter_size=3, padding='SAME', name='conv1')  # 3 * 3 conv
        x1 = instance_normalizer(x1, name='norm1')
        x1 = tf.nn.elu(x1)
        x2 = conv2d(x1, output_num=output_num, stride=1, filter_size=3, padding='SAME', name='conv2')  # 3 * 3 conv
        x2 = instance_normalizer(x2, name='norm2')
        x2 = tf.nn.elu(x2)
        return tf.nn.elu(x2 + x)


# keep size
def res_block_no_norm(x, name='res_block_no_norm', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        batch_, h_, w_, output_num = x.shape.as_list()
        x1 = conv2d(x, output_num=output_num, stride=1, filter_size=3, padding='SAME', name='conv1')  # 3 * 3 conv
        x1 = tf.nn.elu(x1)
        x2 = conv2d(x1, output_num=output_num, stride=1, filter_size=3, padding='SAME', name='conv2')  # 3 * 3 conv
        x2 = tf.nn.elu(x2)
        return tf.nn.elu(x2 + x)


def resize_img(img, resize_h, resize_w):
    return scipy.misc.imresize(img, [resize_h, resize_w])


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak * x, name=name)


# layer height, width
s_h, s_w, _ = input_size
s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
# s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)


# generate (model 1)
def build_generator(noise, train=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        #AttributeError: 'tuple' object has no attribute 'as_list
        cur_size = int(1.75 * size)
        z = full_connect(noise, output_num=cur_size * s_h8 * s_w8, name='g_full', reuse=reuse)
        # reshape
        h0 = tf.reshape(z, [-1, s_h8, s_w8, cur_size])
        h0 = tf.nn.elu(h0, name='g_l0')
        h1 = res_block_no_norm(h0, name='res_block0', reuse=reuse)

        h2 = resize_nn(h1, s_h4, s_w4)
        h2 = res_block_no_norm(h2, name='res_block1', reuse=reuse)

        h3 = resize_nn(h2, s_h2, s_w2)
        h3 = res_block_no_norm(h3, name='res_block2', reuse=reuse)

        h4 = resize_nn(h3, input_size[0], input_size[1])
        h4 = res_block_no_norm(h4, name='res_block3', reuse=reuse)

        h5 = conv2d(h4, output_num=3, stride=1, filter_size=3, name='g_h5', reuse=reuse)
        x_generate = tf.nn.elu(h5, name='g_l5')
        return x_generate


# discriminator as a auto-encoder(model 2)
def build_discriminator(imgs, train=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # encoder
        h0 = conv2d(imgs, output_num=size, stride=1, filter_size=3, padding='SAME', name='en_h0', reuse=reuse)
        # h0 = batch_normalizer(h0, train=train, name='en_bn0')
        # h0 = lrelu(h0)
        h0 = tf.nn.elu(h0, name='en_l0')

        h1 = conv2d(h0, output_num=size, name='en_h1', reuse=reuse)
        # h1 = batch_normalizer(h1, train=train, name='en_bn1')
        # h1 = lrelu(h1)
        h1 = tf.nn.elu(h1, name='en_l1')

        h2 = conv2d(h1, output_num=size * 2, name='en_h2', reuse=reuse)
        # h2 = batch_normalizer(h2, train=train, name='en_bn2')
        # h2 = lrelu(h2)
        h2 = tf.nn.elu(h2, name='en_l2')

        h3 = conv2d(h2, output_num=size * 4, name='en_h3', reuse=reuse)
        # h3 = batch_normalizer(h3, train=train, name='en_bn3')
        # h3 = lrelu(h3)
        h3 = tf.nn.elu(h3, name='en_l3')

        h4 = tf.reshape(h3, [batch_size, s_h8 * s_h8 * size * 4])

        hidden_x = full_connect(h4, output_num=noise_size, name='en_full', reuse=reuse)

        # decoder
        cur_size = int(1.75 * size)
        z = full_connect(hidden_x, output_num=cur_size * s_h8 * s_w8, name='de_full', reuse=reuse)
        # reshape
        h0 = tf.reshape(z, [-1, s_h8, s_w8, cur_size])
        h0 = tf.nn.elu(h0, name='de_l0')
        h1 = res_block_no_norm(h0, name='de_res_block0', reuse=reuse)

        h2 = resize_nn(h1, s_h4, s_w4)
        h2 = res_block_no_norm(h2, name='de_res_block1', reuse=reuse)

        h3 = resize_nn(h2, s_h2, s_w2)
        h3 = res_block_no_norm(h3, name='de_res_block2', reuse=reuse)

        h4 = resize_nn(h3, input_size[0], input_size[1])
        h4 = res_block_no_norm(h4, name='de_res_block3', reuse=reuse)

        h5 = conv2d(h4, output_num=3, stride=1, filter_size=3, name='de_h5', reuse=reuse)
        x_generate = tf.nn.elu(h5, name='de_l5')
        return x_generate


def generate_samples(num):
    noise_imgs = tf.placeholder(tf.float32, [None, noise_size], name='noise_imgs')
    sample_imgs = build_generator(noise_imgs, train=False, reuse=tf.AUTO_REUSE)  # tf.AUTO_REUSE
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/ziyangcheng/python_save_file/be_gan/save_model/4/be_gan.ckpt')
        sample_noise = np.random.uniform(-1.0, 1.0, size=(num, noise_size))
        n_batch = num // batch_size
        for j in range(n_batch):
            samples = sess.run(sample_imgs,
                               feed_dict={noise_imgs: sample_noise[(j * batch_size):((j + 1) * batch_size)]})
            for i in range(len(samples)):
                print('index', j * batch_size + i)
                scipy.misc.imsave(
                    '/home/ziyangcheng/python_save_file/be_gan/output/generate/4/' + str(
                        j * batch_size + i) + 'generate.png', samples[i])
    print('generate done!')


def l2_loss(pred, data):
    loss_val = 2 * tf.nn.l2_loss(pred - data) / batch_size
    return loss_val


def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def start_train(dataset, chunk_size):
    with tf.name_scope('inputs'):
        real_imgs = tf.placeholder(tf.float32, [None, ] + input_size, name='real_images')
        noise_imgs = tf.placeholder(tf.float32, [None, noise_size], name='noise_images')

        kt = tf.placeholder(tf.float32, name='kt')
        lr = tf.placeholder(tf.float32, name='lr')

    fake_imgs = build_generator(noise_imgs, train=True, reuse=False)

    real_decoder = build_discriminator(real_imgs, reuse=False)
    fake_decoder = build_discriminator(fake_imgs, reuse=True)

    with tf.name_scope('loss'):
        d_fake_loss = l1_loss(fake_imgs, fake_decoder)
        d_real_loss = l1_loss(real_imgs, real_decoder)
        g_loss = d_fake_loss
        # total d_loss
        d_loss = d_real_loss - kt * d_fake_loss
        # m_global measure
        m_global = d_real_loss + tf.abs(gamma * d_real_loss - d_fake_loss)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('d_fake_loss', d_fake_loss)
        tf.summary.scalar('d_real_loss', d_real_loss)
        tf.summary.scalar('d_loss', d_loss)
        tf.summary.scalar('m_global', m_global)
        tf.summary.scalar('kt', kt)
        tf.summary.scalar('lr', lr)
    with tf.name_scope('optimizer'):
        train_vars = tf.trainable_variables()
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        d_trainer = tf.train.AdamOptimizer(lr, beta1=0.0).minimize(d_loss, var_list=dis_vars)
        g_trainer = tf.train.AdamOptimizer(lr * 2, beta1=0.0).minimize(g_loss, var_list=gen_vars)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # merge summary
        merged = tf.summary.merge_all()
        # choose dir
        writer = tf.summary.FileWriter('/home/ziyangcheng/python_save_file/be_gan/tf_board', sess.graph)
        # init
        batch_index = 0  # index
        cur_kt = np.float32(0.)  # kt
        cur_lr = np.float32(learning_rate)  # lr
        sess.run(tf.global_variables_initializer())
        for e in range(training_epochs):
            for batch_i in range(chunk_size):
                # batch_data = get_batches(dataset, batch_index)
                batch_data = image_load.load_image_batch(dataset, input_size, batch_size, batch_index)
                batch_index = (batch_index + batch_size) % ((chunk_size - 1) * batch_size)

                # noise
                noise = np.random.uniform(-1.0, 1.0, size=(batch_size, noise_size)).astype(np.float32)

                # Run optimizers
                cur_d_real_loss, cur_d_fake_loss, check_imgs, _ = sess.run(
                    [d_real_loss, d_fake_loss, fake_imgs, g_trainer],
                    feed_dict={real_imgs: batch_data, noise_imgs: noise, lr: cur_lr})
                sess.run(d_trainer, feed_dict={real_imgs: batch_data, noise_imgs: noise, kt: cur_kt, lr: cur_lr})

                # update kt, m_global
                cur_kt = np.maximum(np.minimum(1., cur_kt + lamda_k * (gamma * cur_d_real_loss - cur_d_fake_loss)), 0.)

                if (chunk_size * e + batch_i) % delay_step == 0:
                    # update learning rate
                    cur_lr = np.maximum(0.00001, cur_lr * 0.80)
                if (chunk_size * e + batch_i) % display_step == 0:
                    # print
                    train_loss_d = sess.run(d_loss, feed_dict={real_imgs: batch_data, noise_imgs: noise, kt: cur_kt})
                    fake_loss_d = sess.run(d_fake_loss, feed_dict={noise_imgs: noise})
                    real_loss_d = sess.run(d_real_loss, feed_dict={real_imgs: batch_data})
                    # generator loss
                    train_loss_g = sess.run(g_loss, feed_dict={noise_imgs: noise})
                    # m_global
                    cur_m_global = sess.run(m_global, feed_dict={real_imgs: batch_data, noise_imgs: noise})

                    merge_result = sess.run(merged, feed_dict={real_imgs: batch_data, noise_imgs: noise, kt: cur_kt,
                                                               lr: cur_lr})
                    # merge_result = sess.run(merged, feed_dict={X: batch_xs})
                    writer.add_summary(merge_result, chunk_size * e + batch_i)

                    print("step {}/of epoch {}/{}...".format(chunk_size * e + batch_i, e,training_epochs),
                          "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(
                              train_loss_d,real_loss_d,fake_loss_d), "Generator Loss: {:.4f}".format(train_loss_g))
                    print("m_global: {:.4f}".format(cur_m_global))
                    # save pic
                    scipy.misc.imsave('/home/ziyangcheng/python_save_file/be_gan/output/train/' +
                                      str(chunk_size * e + batch_i) + '-' + str(0) + '.png', check_imgs[0])

        print('train done')
        # save sess
        saver.save(sess, '/home/ziyangcheng/python_save_file/be_gan/save_model/4/be_gan.ckpt')


if __name__ == '__main__':
    data_folder = '/home/ziyangcheng/datasets/faces'
    lsun_path = '/home/ziyangcheng/datasets/lsun/church_outdoor_train_whole/'
    celebA_folder = '/home/ziyangcheng/datasets/celebA/img_align_celeba'
    celebA_path_cut96 = '/home/ziyangcheng/datasets/celebA/celebA_cut96/'
    dataimgs, chunk_s = image_load.list_out_image_files(celebA_path_cut96, batch_size)
    # dataimgs, chunk_s = load_img(data_folder)
    # start_train(dataimgs, chunk_s)
    generate_samples(100)
