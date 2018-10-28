# Adversarial Auto-encoder ->semi-supervised
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
from math import *
import random
from sklearn.cluster import KMeans

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
test_label = np.argmax(mnist.test.labels, 1)
test_data = mnist.test

img_height = 28
img_width = 28
img_size = img_height * img_width

max_epoch = 20
learning_rate = 0.001
h_size = 10
batch_size = 64
display_step = 800
generate_num = 10
class_num = 10


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.02)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


# without popling downsample by strides
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME')


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak * x, name=name)


# generate (model 1)
def build_encoder(x):
    with tf.name_scope('encoder'):
        x_image = tf.reshape(x, [-1, img_height, img_width, 1])

        W_conv1 = weight_variable([5, 5, 1, 16], 'W_conv1')
        b_conv1 = bias_variable([16], 'b_conv1')
        conv1 = lrelu(tf.nn.bias_add(conv2d(x_image, W_conv1), b_conv1), name='conv1')

        W_conv2 = weight_variable([5, 5, 16, 32], 'W_conv2')
        b_conv2 = bias_variable([32], 'b_conv2')
        conv2 = lrelu(tf.nn.bias_add(conv2d(conv1, W_conv2), b_conv2), name='conv2')

        conv2_flat = tf.reshape(conv2, [-1, int(img_height/2/2) * int(img_width/2/2) * 32])

        # full connect layer
        W_full1 = weight_variable([int(img_height/2/2) * int(img_width/2/2) * 32, h_size], 'W_full1')
        b_full1 = bias_variable([h_size], 'b_full1')
        z_mean = tf.nn.bias_add(tf.matmul(conv2_flat, W_full1), b_full1, name='z_mean')
        W_full2 = weight_variable([int(img_height/2/2) * int(img_width/2/2) * 32, h_size], 'W_full2')
        b_full2 = bias_variable([h_size], 'b_full2')
        z_stddev = tf.nn.bias_add(tf.matmul(conv2_flat, W_full2), b_full2, name='z_stddev')

        return z_mean, z_stddev


def build_decoder(z):
    # deconv
    with tf.name_scope('decoder'):
        W_full = weight_variable([h_size, int(img_height/2/2) * int(img_width/2/2) * 32], 'W_full')
        b_full = bias_variable([int(img_height/2/2) * int(img_width/2/2) * 32], 'b_full')

        z_full = tf.nn.bias_add(tf.matmul(z, W_full), b_full, name='z_full')
        z_matrix = tf.nn.relu(tf.reshape(z_full, [batch_size, int(img_height/2/2), int(img_width/2/2), 32]),
                              name='z_matrix')
        W_h1 = weight_variable([5, 5, 16, 32], 'W_h1')
        W_h2 = weight_variable([5, 5, 1, 16], 'W_h2')
        # attention 5 5 16 32 not 5 5 32 16
        h1 = tf.nn.relu(tf.nn.conv2d_transpose(z_matrix, W_h1, [batch_size, int(img_height/2), int(img_width/2), 16],
                                               strides=[1, 2, 2, 1], padding="SAME"), name='h1')

        # attention sigmoid
        h2 = tf.nn.sigmoid(tf.nn.conv2d_transpose(h1, W_h2, [batch_size, img_height, img_width, 1],
                                                  strides=[1, 2, 2, 1], padding="SAME"), name='h2')

        return h2


# discriminator (model 2)
def build_discriminator(z_data, y_onehot):
    with tf.name_scope('discriminator'):
        inputs = tf.concat(axis=1, values=[z_data, y_onehot])
        d_w1 = tf.Variable(tf.truncated_normal([h_size + class_num + 1, (h_size + class_num + 1) * 2],
                                               stddev=0.05), name="d_w1", dtype=tf.float32)
        d_b1 = tf.Variable(tf.zeros([(h_size + class_num + 1) * 2]), name="d_b1", dtype=tf.float32)
        d_w2 = tf.Variable(tf.truncated_normal([(h_size + class_num + 1) * 2, 1], stddev=0.05),
                           name="d_w2", dtype=tf.float32)
        d_b2 = tf.Variable(tf.zeros([1]), name="d_b2", dtype=tf.float32)
        h1 = tf.nn.relu(tf.matmul(inputs, d_w1) + d_b1)
        h3 = tf.matmul(h1, d_w2) + d_b2
        return h3


def kmeans_cluster(data):
    d2_data = None
    if h_size > 2:
        pca = PCA(n_components=2)
        d2_data = pca.fit_transform(data)
    elif h_size == 2:
        d2_data = data
    y_pred = KMeans(n_clusters=2, random_state=170).fit_predict(d2_data)
    plt.scatter(d2_data[:, 0], d2_data[:, 1], c=y_pred)
    plt.show()


def cluster(data):
    # pca more faster
    d2_data = None
    if h_size > 2:
        pca = PCA(n_components=2)
        d2_data = pca.fit_transform(data)
    elif h_size == 2:
        d2_data = data
    plt.scatter(d2_data[:, 0], d2_data[:, 1], s=5, c=test_label)
    plt.colorbar()
    plt.show()


def supervised_gaussian_mixture(batchsize, ndim, label_indices, num_labels):
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    def sample(x, y, label, num_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(num_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], num_labels)
    return z


def gaussian_mixture(batchsize, ndim, num_labels):
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    def sample(x, y, label, num_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(num_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], random.randint(0, num_labels - 1), num_labels)
    return z


def start_train(datas):
    with tf.name_scope('inputs'):
        input_imgs_l = tf.placeholder(tf.float32, [None, img_size], name='input_imgs_l')
        input_imgs_u = tf.placeholder(tf.float32, [None, img_size], name='input_imgs_u')
        input_z_l = tf.placeholder(tf.float32, [None, h_size], name='input_z_l')
        input_z_u = tf.placeholder(tf.float32, [None, h_size], name='input_z_u')
        input_onehot_l = tf.placeholder(tf.float32, [None, class_num + 1], name='input_onehot_l')
        input_onehot_u = tf.placeholder(tf.float32, [None, class_num + 1], name='input_onehot_u')

    # encode l
    encode_z_mean_l, encode_z_stddev_l = build_encoder(input_imgs_l)
    encode_z_mean_u, encode_z_stddev_u = build_encoder(input_imgs_u)
    with tf.name_scope('sample'):
        samples_l = tf.random_normal([batch_size, h_size], 0, 1, dtype=tf.float32)
        fake_z_l = tf.add(tf.multiply(samples_l, encode_z_stddev_l), encode_z_mean_l, name='fake_z_l')

        samples_u = tf.random_normal([batch_size, h_size], 0, 1, dtype=tf.float32)
        fake_z_u = tf.add(tf.multiply(samples_u, encode_z_stddev_u), encode_z_mean_u, name='fake_z_u')

    generated_images_l = build_decoder(fake_z_l)
    generated_flat_l = tf.reshape(generated_images_l, [batch_size, img_height * img_width * 1])

    generated_images_u = build_decoder(fake_z_u)
    generated_flat_u = tf.reshape(generated_images_u, [batch_size, img_height * img_width * 1])

    fake_logits_l = build_discriminator(fake_z_l, input_onehot_l)
    fake_logits_u = build_discriminator(fake_z_u, input_onehot_u)

    real_logits_l = build_discriminator(input_z_l, input_onehot_l)
    real_logits_u = build_discriminator(input_z_u, input_onehot_u)

    with tf.name_scope('loss'):
        encoder_loss_l = -tf.reduce_sum(input_imgs_l * tf.log(1e-7 + generated_flat_l) + (1 - input_imgs_l)
                                        * tf.log(1e-7 + 1 - generated_flat_l), 1,)
        encoder_loss_u = -tf.reduce_sum(input_imgs_u * tf.log(1e-7 + generated_flat_u) + (1 - input_imgs_u)
                                        * tf.log(1e-7 + 1 - generated_flat_u), 1,)
        encoder_loss = tf.reduce_mean(encoder_loss_l + encoder_loss_u, name='encoder_loss')

        # gan
        g_loss_l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits_l, labels=tf.ones_like(fake_logits_l)))

        g_loss_u = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits_u, labels=tf.ones_like(fake_logits_u)))

        g_loss = g_loss_l + g_loss_u

        d_fake_loss_l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits_l, labels=tf.zeros_like(fake_logits_l)))

        d_fake_loss_u = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits_u, labels=tf.zeros_like(fake_logits_u)))

        d_real_loss_l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_logits_l, labels=tf.ones_like(real_logits_l)))

        d_real_loss_u = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_logits_u, labels=tf.ones_like(real_logits_u)))

        d_loss = d_fake_loss_l + d_fake_loss_u + d_real_loss_l + d_real_loss_u
        tf.summary.scalar('encoder_loss', encoder_loss)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('d_fake_loss', d_fake_loss_l + d_fake_loss_u)
        tf.summary.scalar('d_real_loss', d_real_loss_l + d_real_loss_u)
        tf.summary.scalar('d_loss', d_loss)
    with tf.name_scope('optimizer'):
        d_trainer = tf.train.AdamOptimizer(learning_rate).minimize(d_loss)
        g_trainer = tf.train.AdamOptimizer(learning_rate).minimize(g_loss)
        encoder_trainer = tf.train.AdamOptimizer(learning_rate).minimize(encoder_loss)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # merge summary
        merged = tf.summary.merge_all()
        # choose dir
        writer = tf.summary.FileWriter('F:/tf_board/aae_mnist_y', sess.graph)
        sess.run(tf.global_variables_initializer())
        # cal total_batch
        total_batch = int(datas.num_examples/batch_size)

        # for unlabeled y
        y_onehot_u = np.zeros((1, class_num + 1), dtype=np.float32)
        y_onehot_u[0, -1] = 1  # turn on the extra class
        y_onehot_u = np.repeat(y_onehot_u, batch_size, axis=0)

        for e in range(max_epoch):
            for batch_i in range(total_batch):
                batch_xs_l, batch_ys_l = datas.next_batch(batch_size)  # labeled
                batch_xs_u, _ = test_data.next_batch(batch_size)  # unlabeled

                batch_ys = np.argmax(batch_ys_l, 1)

                # add 0 to onehot to be 11 classes
                y_onehot_l = np.zeros(shape=[batch_size, class_num + 1])
                y_onehot_l[:,0:class_num] = batch_ys_l[:,:]

                # real_z
                real_z_l = supervised_gaussian_mixture(batch_size, h_size, batch_ys, class_num)
                real_z_u = gaussian_mixture(batch_size, h_size, class_num)

                # Run optimizers
                check_imgs_l, check_imgs_u, _ = sess.run([generated_images_l, generated_images_u, encoder_trainer],
                                                         feed_dict={input_imgs_l: batch_xs_l, input_imgs_u: batch_xs_u})

                sess.run(d_trainer, feed_dict={input_imgs_l: batch_xs_l, input_imgs_u: batch_xs_u, input_z_l: real_z_l,
                                               input_z_u: real_z_u, input_onehot_l: y_onehot_l,
                                               input_onehot_u: y_onehot_u})

                sess.run(g_trainer, feed_dict={input_imgs_l: batch_xs_l, input_imgs_u: batch_xs_u,
                                               input_onehot_l: y_onehot_l, input_onehot_u: y_onehot_u})

                if (total_batch * e + batch_i) % display_step == 0:
                    train_encoder_loss = sess.run(encoder_loss, feed_dict={input_imgs_l: batch_xs_l,
                                                                           input_imgs_u: batch_xs_u})
                    train_loss_d = sess.run(d_loss, feed_dict={input_imgs_l: batch_xs_l, input_imgs_u: batch_xs_u,
                                                               input_z_l: real_z_l, input_z_u: real_z_u,
                                                               input_onehot_l: y_onehot_l,
                                                               input_onehot_u: y_onehot_u})
                    fake_loss_d = sess.run(d_fake_loss_l + d_fake_loss_u, feed_dict={input_imgs_l: batch_xs_l,
                                                                                     input_imgs_u: batch_xs_u,
                                                                                     input_onehot_l: y_onehot_l,
                                                                                     input_onehot_u: y_onehot_u})
                    real_loss_d = sess.run(d_real_loss_l + d_real_loss_u, feed_dict={input_z_l: real_z_l,
                                                                                     input_z_u: real_z_u,
                                                                                     input_onehot_l: y_onehot_l,
                                                                                     input_onehot_u: y_onehot_u})
                    train_loss_g = sess.run(g_loss, feed_dict={input_imgs_l: batch_xs_l, input_imgs_u: batch_xs_u,
                                                               input_onehot_l: y_onehot_l, input_onehot_u: y_onehot_u})

                    merge_result = sess.run(merged, feed_dict={input_imgs_l: batch_xs_l, input_imgs_u: batch_xs_u,
                                                               input_z_l: real_z_l, input_z_u: real_z_u,
                                                               input_onehot_l: y_onehot_l,
                                                               input_onehot_u: y_onehot_u})
                    writer.add_summary(merge_result, total_batch * e + batch_i)

                    print("step {}/of epoch {}/{}...".format(total_batch * e + batch_i, e,max_epoch),
                          "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(
                              train_loss_d,real_loss_d,fake_loss_d),"Encoder Loss: {:.4f}".format(train_encoder_loss),
                          "Generator Loss: {:.4f}".format(train_loss_g))

                    # show pic
                    plt.imsave('F:/tf_board/aae_mnist_y/' + str(total_batch * e + batch_i)
                               + '-' + str(0) + 'l.png', check_imgs_l[0][:,:,0],cmap='Greys_r')
                    plt.imsave('F:/tf_board/aae_mnist_y/' + str(total_batch * e + batch_i)
                               + '-' + str(1) + 'u.png', check_imgs_u[1][:,:,0],cmap='Greys_r')

        print('train done')
        # save sess
        saver.save(sess, '/root/condition_aae_mnist/aae_y_mnist.ckpt')

        # draw
        # nx = ny = 20
        # x_values = np.linspace(-3, 3, nx)
        # y_values = np.linspace(-3, 3, ny)
        # canvas = np.empty((28*ny, 28*nx))
        # for i, yi in enumerate(x_values):
        #     for j, xi in enumerate(y_values):
        #         z_mu = np.array([[xi, yi]]*batch_size)
        #         x_mean = sess.run(generated_images_l, feed_dict={fake_z_l: z_mu})
        #         canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)
        #
        # plt.figure(figsize=(8, 10))
        # Xi, Yi = np.meshgrid(x_values, y_values)
        # plt.imshow(canvas, origin="upper", cmap="gray")
        # plt.tight_layout()
        # plt.show()

        # cluster
        samples_cluster = tf.random_normal([len(test_data.images), h_size], 0, 1, dtype=tf.float32)
        z_cluster = tf.add(tf.multiply(samples_cluster, encode_z_stddev_l), encode_z_mean_l)  # multiply not matmul
        cluster_data = sess.run(z_cluster, feed_dict={input_imgs_l: test_data.images})
        cluster(cluster_data)

        # generate
        samples = sess.run(generated_images_l, feed_dict={input_imgs_l: test_data.images[0:batch_size]})[:generate_num]
        for i in range(len(samples)):
            plt.imsave('F:/tf_board/aae_mnist_y/' + str(i) + 'fake_generate.png', samples[i][:,:,0],cmap='Greys_r')
        for i in range(class_num):
            z_l = supervised_gaussian_mixture(batch_size, h_size, np.zeros(shape=[batch_size,1]) + i, class_num)
            gaussian_samples = sess.run(generated_images_l, feed_dict={fake_z_l: z_l})
            plt.imsave('F:/tf_board/aae_mnist_y/' + str(i) + '0gaussian_generate.png', gaussian_samples[0][:, :, 0],
                       cmap='Greys_r')
            plt.imsave('F:/tf_board/aae_mnist_y/' + str(i) + '1gaussian_generate.png', gaussian_samples[1][:, :, 0],
                       cmap='Greys_r')
        print('generate done!')


if __name__ == '__main__':
    start_train(mnist.train)
