# -*- coding:utf-8-*-
import tensorflow as tf
import numpy as np
import math
import scipy.misc
import GANs.tempered_gan.ops as ops
import GANs.tempered_gan.utils as utl
import out_image_data_load as image_load

# modified from dc_gan
# using lsun dataset


class TemperedGAN(object):
    def __init__(self, da_f, in_s, s, lr, no_s, tr_e, ba_s, di_s):
        self.data_folder = da_f
        self.input_size = in_s
        self.size = s
        self.learning_rate = lr
        self.noise_size = no_s
        self.training_epochs = tr_e
        self.batch_size = ba_s
        self.display_step = di_s
        self.k = 24000
        self.data = None
        self.chunk_size = None

    def _load_dataset(self):
        self.data, self.chunk_size = utl.load_img(self.data_folder, self.input_size, self.batch_size)

    def _get_batches(self, batch_index):
        batch = self.data[batch_index:batch_index + self.batch_size, :, :, :]
        return batch

    @staticmethod
    def _mse_loss(pred, data):
        loss_val = tf.reduce_mean(tf.multiply((pred - data), (pred - data))) / 2
        return loss_val

    def generate_net(self, noise, train=True, reuse=False):
        """
        :param noise: source noise z
        :param train:
        :param reuse:
        :return:
        """
        # layer height, width
        s_h, s_w, _ = self.input_size
        s_h2, s_w2 = utl.get_out_size(s_h, 2), utl.get_out_size(s_w, 2)
        s_h4, s_w4 = utl.get_out_size(s_h2, 2), utl.get_out_size(s_w2, 2)
        s_h8, s_w8 = utl.get_out_size(s_h4, 2), utl.get_out_size(s_w4, 2)
        s_h16, s_w16 = utl.get_out_size(s_h8, 2), utl.get_out_size(s_w8, 2)
        with tf.variable_scope('generator', reuse=reuse):
            # AttributeError: 'tuple' object has no attribute 'as_list
            z = ops.full_connect(noise, output_num=self.size * 8 * s_h16 * s_w16, name='g_full', reuse=reuse)
            # reshape [batch_size, h, w, c]
            h0 = tf.reshape(z, [-1, s_h16, s_w16, self.size * 8])
            h0 = ops.batch_normalizer(h0, train=train, name='g_bn0', reuse=reuse)
            h0 = ops.lrelu(h0, name='g_l0')

            h1 = ops.deconv2d(h0, output_size=[self.batch_size, s_h8, s_w8, self.size * 4], name='g_h1', reuse=reuse)
            h1 = ops.batch_normalizer(h1, train=train, name='g_bn1', reuse=reuse)
            h1 = ops.lrelu(h1, name='g_l1')

            h2 = ops.deconv2d(h1, output_size=[self.batch_size, s_h4, s_w4, self.size * 2], name='g_h2', reuse=reuse)
            h2 = ops.batch_normalizer(h2, train=train, name='g_bn2', reuse=reuse)
            h2 = ops.lrelu(h2, name='g_l2')

            h3 = ops.deconv2d(h2, output_size=[self.batch_size, s_h2, s_w2, self.size * 1], name='g_h3', reuse=reuse)
            h3 = ops.batch_normalizer(h3, train=train, name='g_bn3', reuse=reuse)
            h3 = ops.lrelu(h3, name='g_l3')

            h4 = ops.deconv2d(h3, output_size=[self.batch_size, ] + self.input_size, name='g_h4', reuse=reuse)
            x_generate = tf.nn.tanh(h4, name='g_t4')
            return x_generate

    def discriminator_net(self, lx, reuse=False):
        """
        :param lx: the images from lens
        :param reuse:
        :return:
        """
        # layer height, width
        s_h, s_w, _ = self.input_size
        s_h2, s_w2 = utl.get_out_size(s_h, 2), utl.get_out_size(s_w, 2)
        s_h4, s_w4 = utl.get_out_size(s_h2, 2), utl.get_out_size(s_w2, 2)
        s_h8, s_w8 = utl.get_out_size(s_h4, 2), utl.get_out_size(s_w4, 2)
        s_h16, s_w16 = utl.get_out_size(s_h8, 2), utl.get_out_size(s_w8, 2)
        with tf.variable_scope('discriminator', reuse=reuse):
            h0 = ops.conv2d(lx, output_num=self.size, name='d_h0', reuse=reuse)
            h0 = ops.lrelu(h0, name='d_l0')

            h1 = ops.conv2d(h0, output_num=self.size * 2, name='d_h1', reuse=reuse)
            h1 = ops.batch_normalizer(h1, name='d_bn1', reuse=reuse)
            h1 = ops.lrelu(h1, name='d_l1')

            h2 = ops.conv2d(h1, output_num=self.size * 4, name='d_h2', reuse=reuse)
            h2 = ops.batch_normalizer(h2, name='d_bn2', reuse=reuse)
            h2 = ops.lrelu(h2, name='d_l2')

            h3 = ops.conv2d(h2, output_num=self.size * 8, name='d_h3', reuse=reuse)
            h3 = ops.batch_normalizer(h3, name='d_bn3', reuse=reuse)
            h3 = ops.lrelu(h3, name='d_l3')

            h4 = tf.reshape(h3, [self.batch_size, s_h16 * s_w16 * self.size * 8])

            h4 = ops.full_connect(h4, output_num=1, name='d_full', reuse=reuse)
            return h4

    def lens_net(self, x, reuse=False):
        """
        :param x: input real data x
        :param reuse:
        :return:lens x: lx
        """
        with tf.variable_scope('lens', reuse=reuse):
            h0 = ops.conv2d(x, output_num=self.size, stride=1, filter_size=3, name='l_h0')
            h0 = ops.lrelu(h0, name='l_l0')

            h1 = ops.res_block3_3(h0, name='l_res_1', reuse=reuse)
            h2 = ops.res_block3_3(h1, name='l_res_2', reuse=reuse)

            h3 = ops.conv2d(h2, output_num=3, stride=1, filter_size=3, name='l_h4')
            h3 = ops.lrelu(h3, leak=0.4, name='l_l4')
            h3 = h3 + x
            return h3

    def start_train(self):
        # self._load_dataset()
        files, self.chunk_size = image_load.list_out_image_files(self.data_folder, self.batch_size)
        with tf.name_scope('inputs'):
            real_imgs = tf.placeholder(tf.float32, [None, ] + self.input_size, name='real_images')
            noise_imgs = tf.placeholder(tf.float32, [None, self.noise_size], name='noise_images')
            lamda = tf.placeholder(tf.float32, name='lamda')

        fake_imgs = self.generate_net(noise_imgs, train=True, reuse=False)
        lens_imgs = self.lens_net(real_imgs, reuse=False)

        lens_logits = self.discriminator_net(lens_imgs, reuse=False)
        fake_logits = self.discriminator_net(fake_imgs, reuse=True)

        with tf.name_scope('loss'):
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_logits, labels=tf.ones_like(fake_logits)))

            l_loss_a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=lens_logits, labels=tf.zeros_like(lens_logits)))
            l_loss_r = self._mse_loss(real_imgs, lens_imgs)
            l_loss = lamda * l_loss_a + l_loss_r

            d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_logits, labels=tf.zeros_like(fake_logits)))
            d_lens_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=lens_logits, labels=tf.ones_like(lens_logits)))
            d_loss = d_fake_loss + d_lens_loss

            tf.summary.scalar('g_loss', g_loss)
            tf.summary.scalar('l_loss', l_loss)
            tf.summary.scalar('l_lamda', lamda)
            tf.summary.scalar('d_fake_loss', d_fake_loss)
            tf.summary.scalar('d_real_loss', d_lens_loss)
            tf.summary.scalar('d_loss', d_loss)
        with tf.name_scope('optimizer'):
            train_vars = tf.trainable_variables()
            gen_vars = [var for var in train_vars if var.name.startswith('generator')]
            dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
            lens_vars = [var for var in train_vars if var.name.startswith('lens')]
            # global_step = tf.Variable(0, trainable=False)
            # rate = tf.train.exponential_decay(self.learning_rate, global_step, 1024 * 4, 0.80, staircase=True)
            # rate = tf.maximum(rate, 0.00001)
            # d_trainer = tf.train.AdamOptimizer(rate, beta1=0.5).minimize(d_loss, var_list=dis_vars,
            #                                                              global_step=global_step)
            # g_trainer = tf.train.AdamOptimizer(rate, beta1=0.5).minimize(g_loss, var_list=gen_vars,
            #                                                              global_step=global_step)
            # g_trainer = tf.train.AdamOptimizer(rate, beta1=0.5).minimize(g_loss, var_list=gen_vars,
            #                                                              global_step=global_step)
            d_trainer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.0).minimize(d_loss, var_list=dis_vars)
            g_trainer = tf.train.AdamOptimizer(self.learning_rate * 2, beta1=0.0).minimize(g_loss, var_list=gen_vars)
            l_trainer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.0).minimize(l_loss, var_list=lens_vars)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            # merge summary
            merged = tf.summary.merge_all()
            # choose dir
            writer = tf.summary.FileWriter('/home/ziyangcheng/python_save_file/temp_gan/tf_board', sess.graph)
            batch_index = 0  # init index
            cur_lamda = 1
            sess.run(tf.global_variables_initializer())
            for e in range(self.training_epochs):
                for batch_i in range(self.chunk_size):
                    # batch_data = self._get_batches(batch_index)
                    batch_data = image_load.load_image_batch(files, self.input_size, self.batch_size, batch_index)
                    batch_index = (batch_index + self.batch_size) % ((self.chunk_size - 1) * self.batch_size)

                    # noise
                    noise = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.noise_size)).astype(np.float32)

                    if (self.chunk_size * e + batch_i) <= self.k:
                        cur_lamda = 1 - np.sin(((self.chunk_size * e + batch_i) * math.pi) / (2 * self.k))
                    else:
                        cur_lamda = 0

                    # Run optimizers
                    sess.run(d_trainer, feed_dict={real_imgs: batch_data, noise_imgs: noise})
                    sess.run(g_trainer, feed_dict={noise_imgs: noise})
                    check_imgs, whatever, _ = sess.run([fake_imgs, lens_imgs, l_trainer],
                                                       feed_dict={noise_imgs: noise, real_imgs: batch_data,
                                                                  lamda: cur_lamda})

                    if (self.chunk_size * e + batch_i) % self.display_step == 0:
                        train_loss_d = sess.run(d_loss, feed_dict={real_imgs: batch_data, noise_imgs: noise})
                        fake_loss_d = sess.run(d_fake_loss, feed_dict={noise_imgs: noise})
                        lens_loss_d = sess.run(d_lens_loss, feed_dict={real_imgs: batch_data})
                        # generator loss
                        train_loss_g = sess.run(g_loss, feed_dict={noise_imgs: noise})
                        # lens loss
                        train_loss_l = sess.run(l_loss, feed_dict={real_imgs: batch_data, lamda: cur_lamda})

                        merge_result = sess.run(merged,
                                                feed_dict={real_imgs: batch_data, noise_imgs: noise, lamda: cur_lamda})
                        writer.add_summary(merge_result, self.chunk_size * e + batch_i)

                        print(
                            "step {}/of epoch {}/{}...".format(self.chunk_size * e + batch_i, e, self.training_epochs),
                            "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(
                                train_loss_d, lens_loss_d, fake_loss_d),
                            "Generator Loss: {:.4f}".format(train_loss_g),
                            "Lens Loss: {:.4f}".format(train_loss_l), "cur_lamda: {:.4f}".format(cur_lamda))

                        # save pic
                        scipy.misc.imsave('/home/ziyangcheng/python_save_file/temp_gan/output/train/' +
                                          str(self.chunk_size * e + batch_i) + '-' + str(0) + 'train.png',
                                          check_imgs[0])

                        scipy.misc.imsave('/home/ziyangcheng/python_save_file/temp_gan/output/train/' +
                                          str(self.chunk_size * e + batch_i) + '-' + str(0) + 'lens.png',
                                          whatever[0])

            print('train done')
            # save sess
            saver.save(sess, '/home/ziyangcheng/python_save_file/temp_gan/save_model/4/temp_gan.ckpt')

    def generate(self, num):
        noise_imgs = tf.placeholder(tf.float32, [None, self.noise_size], name='noise_images')
        sample_imgs = self.generate_net(noise_imgs, train=False, reuse=tf.AUTO_REUSE)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, '/home/ziyangcheng/python_save_file/temp_gan/save_model/4/temp_gan.ckpt')
            sample_noise = np.random.uniform(-1.0, 1.0, size=(num, self.noise_size)).astype(np.float32)
            n_batch = num // self.batch_size
            for j in range(n_batch):
                samples = sess.run(sample_imgs,
                                   feed_dict={noise_imgs: sample_noise[(j * self.batch_size):((j + 1) * self.batch_size)]})
                for i in range(len(samples)):
                    print('index', j * self.batch_size + i)
                    scipy.misc.imsave(
                        '/home/ziyangcheng/python_save_file/temp_gan/output/generate/4/' + str(
                            j * self.batch_size + i) + 'generate.png', samples[i])
        print('generate done!')


if __name__ == '__main__':
    # lsun_path = '/home/ziyangcheng/datasets/lsun/church_outdoor_train_whole/'
    celebA_folder = '/home/ziyangcheng/datasets/celebA/img_align_celeba'
    a_temp_gan = TemperedGAN(celebA_folder, [96, 96, 3], 32, 0.0001, 1024, 10, 64, 512)
    # a_temp_gan.start_train()
    a_temp_gan.generate(100)
