# -*- coding: utf-8 -*-
# https://github.com/czzyyy/SNGAN
import tensorflow as tf
import numpy as np
import math
import os
import scipy.misc
import out_image_data_load as image_load

learning_rate = 0.0001
noise_size = 128
input_size = [96, 96, 3]
training_epochs = 6
batch_size = 16
display_step = 1024
delay_step = 1024 * 20
size = 32


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


# https://github.com/taki0112/Spectral_Normalization-Tensorflow
def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w_mat = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", shape=[1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w_mat))
        v_hat = tf.nn.l2_normalize(v_)
        u_ = tf.matmul(v_hat, w_mat)
        u_hat = tf.nn.l2_normalize(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w_mat), tf.transpose(u_hat))
    w_norm = w_mat / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm


def full_connect(x, output_num, sn=True, stddev=0.02, bias=0.0, name='full_connect', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        shape = x.shape.as_list()
        w = tf.get_variable('w', [shape[1], output_num], tf.float32, tf.truncated_normal_initializer(stddev=stddev))
        if sn:
            w = spectral_norm(w)
        b = tf.get_variable('b', [output_num], tf.float32, tf.constant_initializer(bias))
        return tf.matmul(x, w) + b


def conv2d(x, output_num, sn=True, stride=2, filter_size=5, stddev=0.02, padding='SAME', name='conv2d', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        # filter : [height, width, in_channels, output_channels]
        shape = x.shape.as_list()
        filter_shape = [filter_size, filter_size, shape[-1], output_num]
        strides_shape = [1, stride, stride, 1]
        w = tf.get_variable('w', filter_shape, tf.float32, tf.truncated_normal_initializer(stddev=stddev))
        if sn:
            w = spectral_norm(w)
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
        x1 = conv2d(x, sn=True, output_num=output_num / 2, stride=1, filter_size=1, padding='VALID', name='conv1')  # 1 * 1 conv
        x1 = tf.nn.relu(batch_normalizer(x1, train=train, name='bn1'))
        x2 = conv2d(x1, sn=True, output_num=output_num / 2, stride=1, filter_size=3, padding='SAME', name='conv2')  # 3 * 3 conv
        x2 = tf.nn.relu(batch_normalizer(x2, train=train, name='bn2'))
        x3 = conv2d(x2, sn=True, output_num=output_num, stride=1, filter_size=1, padding='VALID', name='conv3')
        x3 = batch_normalizer(x3, train=train, name='bn3')
        return tf.nn.relu(x3 + x)


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
        z = full_connect(noise, sn=True, output_num=size * 8 * s_h16 * s_w16, name='g_full', reuse=reuse)
        # reshape
        h0 = tf.reshape(z, [-1, s_h16, s_w16, size * 8])
        h0 = batch_normalizer(h0, train=train, name='g_bn0', reuse=reuse)
        h0 = lrelu(h0, name='g_l0')

        h1 = deconv2d(h0, output_size=[batch_size, s_h8, s_w8, size * 4], name='g_h1', reuse=reuse)
        h1 = batch_normalizer(h1, train=train, name='g_bn1', reuse=reuse)
        h1 = lrelu(h1, name='g_l1')

        h2 = deconv2d(h1, output_size=[batch_size, s_h4, s_w4, size * 2], name='g_h2', reuse=reuse)
        h2 = batch_normalizer(h2, train=train, name='g_bn2', reuse=reuse)
        h2 = lrelu(h2, name='g_l2')

        h3 = deconv2d(h2, output_size=[batch_size, s_h2, s_w2, size * 1], name='g_h3', reuse=reuse)
        h3 = batch_normalizer(h3, train=train, name='g_bn3', reuse=reuse)
        h3 = lrelu(h3, name='g_l3')

        h4 = deconv2d(h3, output_size=[batch_size, ] + input_size, name='g_h4', reuse=reuse)
        x_generate = tf.nn.tanh(h4, name='g_l4')
        return x_generate


# discriminator (model 2)
def build_discriminator(imgs, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        h0 = conv2d(imgs, sn=True, output_num=size, name='d_h0', reuse=reuse)
        h0 = lrelu(h0)

        h1 = conv2d(h0, sn=True, output_num=size * 2, name='d_h1', reuse=reuse)
        h1 = batch_normalizer(h1, name='d_bn1', reuse=reuse)
        h1 = lrelu(h1)

        h2 = conv2d(h1, sn=True, output_num=size * 4, name='d_h2', reuse=reuse)
        h2 = batch_normalizer(h2, name='d_bn2', reuse=reuse)
        h2 = lrelu(h2)

        h3 = conv2d(h2, sn=True, output_num=size * 8, name='d_h3', reuse=reuse)
        h3 = batch_normalizer(h3, name='d_bn3', reuse=reuse)
        h3 = lrelu(h3)

        h4 = tf.reshape(h3, [batch_size, s_h16 * s_w16 * size * 8])

        h4 = full_connect(h4, sn=True, output_num=1, name='d_full', reuse=reuse)
        return h4


def generate_samples(num):
    noise_imgs = tf.placeholder(tf.float32, [None, noise_size], name='noise_imgs')
    sample_imgs = build_generator(noise_imgs, train=False, reuse=tf.AUTO_REUSE)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '/home/ziyangcheng/python_save_file/sn_gan/save_model/1/sn_gan.ckpt')
        sample_noise = np.random.uniform(-1.0, 1.0, size=(num, noise_size)).astype(np.float32)
        n_batch = num // batch_size
        for j in range(n_batch):
            samples = sess.run(sample_imgs,
                               feed_dict={noise_imgs: sample_noise[(j * batch_size):((j + 1) * batch_size)]})
            for i in range(len(samples)):
                print('index', j * batch_size + i)
                scipy.misc.imsave(
                    '/home/ziyangcheng/python_save_file/sn_gan/output/generate/1/' + str(
                        j * batch_size + i) + 'generate.png', samples[i])
    print('generate done!')


def mse_loss(pred, data):
    loss_val = tf.reduce_mean(tf.multiply((pred - data), (pred - data))) / 2
    return loss_val


def start_train(dataset, chunk_size):
    with tf.name_scope('inputs'):
        real_imgs = tf.placeholder(tf.float32, [None, ] + input_size, name='real_images')
        noise_imgs = tf.placeholder(tf.float32, [None, noise_size], name='noise_images')

    fake_imgs = build_generator(noise_imgs, train=True, reuse=False)

    real_logits = build_discriminator(real_imgs, reuse=False)
    fake_logits = build_discriminator(fake_imgs, reuse=True)

    with tf.name_scope('loss'):
        # g_loss = mse_loss(fake_logits, tf.ones_like(fake_logits))
        # d_fake_loss = mse_loss(fake_logits, tf.zeros_like(fake_logits))
        # d_real_loss = mse_loss(real_logits, tf.ones_like(real_logits))
        # d_loss = tf.add(d_fake_loss, d_real_loss)
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits, labels=tf.ones_like(fake_logits)))
        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits, labels=tf.zeros_like(fake_logits)))
        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_logits, labels=tf.ones_like(real_logits)))
        d_loss = tf.add(d_fake_loss, d_real_loss)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('d_fake_loss', d_fake_loss)
        tf.summary.scalar('d_real_loss', d_real_loss)
        tf.summary.scalar('d_loss', d_loss)
    with tf.name_scope('optimizer'):
        train_vars = tf.trainable_variables()
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        global_step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(learning_rate, global_step, delay_step, 0.50, staircase=True)
        rate = tf.maximum(rate, 0.00001)
        d_trainer = tf.train.AdamOptimizer(rate, beta1=0.0).minimize(d_loss, var_list=dis_vars, global_step=global_step)
        g_trainer = tf.train.AdamOptimizer(rate, beta1=0.0).minimize(g_loss, var_list=gen_vars, global_step=global_step)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # merge summary
        merged = tf.summary.merge_all()
        # choose dir
        writer = tf.summary.FileWriter('/home/ziyangcheng/python_save_file/sn_gan/tf_board', sess.graph)
        batch_index = 0  # init index
        sess.run(tf.global_variables_initializer())
        for e in range(training_epochs):
            for batch_i in range(chunk_size):
                # batch_data = get_batches(dataset, batch_index)
                batch_data = image_load.load_image_batch(dataset, input_size, batch_size, batch_index)
                batch_index = (batch_index + batch_size) % ((chunk_size - 1) * batch_size)

                # noise
                noise = np.random.uniform(-1.0, 1.0, size=(batch_size, noise_size)).astype(np.float32)

                # Run optimizers
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

                    writer.add_summary(merge_result, chunk_size * e + batch_i)

                    print("step {}/of epoch {}/{}...".format(chunk_size * e + batch_i, e,training_epochs),
                          "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(
                              train_loss_d,real_loss_d,fake_loss_d), "Generator Loss: {:.4f}".format(train_loss_g))

                    # save pic
                    scipy.misc.imsave('/home/ziyangcheng/python_save_file/sn_gan/output/train/' +
                                      str(chunk_size * e + batch_i) + '-' + str(0) + '.png', check_imgs[0])

        print('train done')
        # save sess
        saver.save(sess, '/home/ziyangcheng/python_save_file/sn_gan/save_model/1/sn_gan.ckpt')


if __name__ == '__main__':
    data_folder = '/home/ziyangcheng/datasets/faces'
    lsun_path = '/home/ziyangcheng/datasets/lsun/church_outdoor_train_whole/'
    celebA_cut96 = '/home/ziyangcheng/datasets/celebA/celebA_cut96'
    # dataimgs, chunk_s = load_img(data_folder)
    dataimgs, chunk_s = image_load.list_out_image_files(celebA_cut96, batch_size)
    start_train(dataimgs, chunk_s)
    # generate_samples(100)