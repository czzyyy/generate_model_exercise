import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# add some noise
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
img_height = 28
img_width = 28
img_size = img_height * img_width
noise_size = mnist.train.labels.shape[1] - 5

max_epoch = 200

h_size = 200
z_size = 100
batch_size = 64

g_w1 = tf.Variable(tf.truncated_normal([z_size + noise_size, h_size], stddev=0.05), name="g_w1", dtype=tf.float32)
g_b1 = tf.Variable(tf.zeros([h_size]), name="g_b1", dtype=tf.float32)
g_w2 = tf.Variable(tf.truncated_normal([h_size, img_size], stddev=0.05), name="g_w2", dtype=tf.float32)
g_b2 = tf.Variable(tf.zeros([img_size]), name="g_b2", dtype=tf.float32)
g_params = [g_w1, g_b1, g_w2, g_b2]

d_w1 = tf.Variable(tf.truncated_normal([img_size + noise_size, h_size], stddev=0.05), name="d_w1", dtype=tf.float32)
d_b1 = tf.Variable(tf.zeros([h_size]), name="d_b1", dtype=tf.float32)
d_w2 = tf.Variable(tf.truncated_normal([h_size, 1], stddev=0.05), name="d_w2", dtype=tf.float32)
d_b2 = tf.Variable(tf.zeros([1]), name="d_b2", dtype=tf.float32)
d_params = [d_w1, d_b1, d_w2, d_b2]


# generate (model 1)
def build_generator(z_prior, n):
    with tf.name_scope('generator'):
        inputs = tf.concat(axis=1, values=[z_prior, n])
        h1 = tf.nn.relu(tf.matmul(inputs, g_w1) + g_b1)
        h2 = tf.matmul(h1, g_w2) + g_b2
        x_generate = tf.nn.sigmoid(h2)
        return x_generate


# discriminator (model 2)
def build_discriminator(x_data, n, keep_prob):
    with tf.name_scope('discriminator'):
        inputs = tf.concat(axis=1, values=[x_data, n])
        h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(inputs, d_w1) + d_b1), keep_prob)
        h3 = tf.matmul(h1, d_w2) + d_b2
        y_data = h3
        return y_data


def train():
    with tf.Session() as sess:
        with tf.name_scope('inputs'):
            x_data = tf.placeholder(tf.float32, [None, img_size], name="x_data")
            z_prior = tf.placeholder(tf.float32, [None, z_size], name="z_prior")
            noise = tf.placeholder(tf.float32, shape=[None, noise_size])
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        x_generated = build_generator(z_prior, noise)
        y_data = build_discriminator(x_data, noise, keep_prob)
        y_generated = build_discriminator(x_generated, noise, keep_prob)

        with tf.name_scope('loss'):
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_generated, labels=tf.ones_like(y_generated)))
            d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_generated, labels=tf.zeros_like(y_generated)))
            d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=y_data, labels=tf.ones_like(y_data)))
            d_loss = tf.add(d_fake_loss, d_real_loss)
            tf.summary.scalar('g_loss', g_loss)
            tf.summary.scalar('d_fake_loss', d_fake_loss)
            tf.summary.scalar('d_real_loss', d_real_loss)
            tf.summary.scalar('d_loss', d_loss)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(0.001)
            d_trainer = optimizer.minimize(d_loss, var_list=d_params)
            g_trainer = optimizer.minimize(g_loss, var_list=g_params)

        saver = tf.train.Saver()
        # merge summary
        merged = tf.summary.merge_all()
        # choose dir
        writer = tf.summary.FileWriter('F:/tf_board/basic_gan_mnist', sess.graph)
        sess.run(tf.global_variables_initializer())
        for e in range(max_epoch):
            for batch_i in range(mnist.train.num_examples//batch_size):
                batch_data, y_data = mnist.train.next_batch(batch_size)
                # noise
                y_noise = np.random.uniform(-1.0, 1.0, size=(batch_size, noise_size))
                # generator noise
                batch_noise = np.random.uniform(-1.0, 1.0, size=(batch_size, z_size))

                # Run optimizers
                sess.run(d_trainer, feed_dict={x_data: batch_data, z_prior: batch_noise, noise: y_noise, keep_prob: 0.7})
                sess.run(g_trainer, feed_dict={z_prior: batch_noise, noise: y_noise, keep_prob: 0.7})

                if ((mnist.train.num_examples//batch_size) * e + batch_i) % (mnist.train.num_examples//batch_size) == 0:
                    train_loss_d = sess.run(d_loss, feed_dict={x_data: batch_data, z_prior: batch_noise, noise: y_noise,
                                                               keep_prob: 1.0})
                    fake_loss_d = sess.run(d_fake_loss, feed_dict={z_prior: batch_noise, noise: y_noise, keep_prob: 1.0})
                    real_loss_d = sess.run(d_real_loss, feed_dict={x_data: batch_data, noise: y_noise, keep_prob: 1.0})
                    # generator loss
                    train_loss_g = sess.run(g_loss, feed_dict={z_prior: batch_noise, noise: y_noise, keep_prob: 1.0})

                    merge_result = sess.run(merged, feed_dict={x_data: batch_data, z_prior: batch_noise, noise: y_noise,
                                                               keep_prob: 1.0})
                    writer.add_summary(merge_result, (mnist.train.num_examples//batch_size) * e + batch_i)

                    print("Epoch {}/{}...".format(e+1, max_epoch),
                          "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(
                              train_loss_d, real_loss_d, fake_loss_d), "Generator Loss: {:.4f}".format(train_loss_g))

            if e % 10 == 0:
                n_sample = 16
                sample_noise = np.random.uniform(-1.0, 1.0, size=(n_sample, z_size))
                y_sample = np.random.uniform(-1.0, 1.0, size=(n_sample, noise_size))
                check_imgs = sess.run(x_generated, feed_dict={z_prior: sample_noise, noise: y_sample}
                                      ).reshape((n_sample, 28, 28))[:2]

                plt.imsave('F:/tf_board/basic_gan_mnist/' + str(e) + '-' + str(0) + '.png', check_imgs[0],
                           cmap='Greys_r')
                plt.imsave('F:/tf_board/basic_gan_mnist/' + str(e) + '-' + str(1) + '.png', check_imgs[1],
                           cmap='Greys_r')

        print('train done')
        n_sample = 16
        sample_noise = np.random.uniform(-1.0, 1.0, size=(n_sample, z_size))
        y_sample = np.random.uniform(-1.0, 1.0, size=(n_sample, noise_size))
        check_imgs = sess.run(x_generated, feed_dict={z_prior: sample_noise, noise: y_sample}
                              ).reshape((n_sample, 28, 28))[:5]
        for i in range(5):
            plt.imsave('F:/tf_board/basic_gan_mnist/' + 'final-' + str(i) + '.png', check_imgs[i], cmap='Greys_r')

        # save sess
        saver.save(sess, '/root/basic_gan_mnist.ckpt')


if __name__ == '__main__':
    train()
