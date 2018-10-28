import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA


class Basic_VAE(object):
    def __init__(self, train_data, test_data, test_label, learning_rate, input_size,
                 training_epochs, batch_size, display_step, examples_to_show):
        self.train_data = train_data
        self.test_data = test_data
        self.test_label = test_label
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.display_step = display_step
        self.examples_to_show = examples_to_show

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    # without popling downsample by strides
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

    def encoder(self, X):
        with tf.name_scope('encoder'):
            x_image = tf.reshape(X, [-1, 28, 28, 1])

            W_conv1 = self.weight_variable([5, 5, 1, 16], 'W_conv1')
            b_conv1 = self.bias_variable([16], 'b_conv1')
            conv1 = tf.nn.relu(tf.nn.bias_add(self.conv2d(x_image, W_conv1), b_conv1), name='conv1')

            W_conv2 = self.weight_variable([5, 5, 16, 32], 'W_conv2')
            b_conv2 = self.bias_variable([32], 'b_conv2')
            conv2 = tf.nn.relu(tf.nn.bias_add(self.conv2d(conv1, W_conv2), b_conv2), name='conv2')

            conv2_flat = tf.reshape(conv2, [-1, 7 * 7 * 32])

            # full connect layer
            W_full1 = self.weight_variable([7 * 7 * 32, 2], 'W_full1')
            b_full1 = self.bias_variable([2], 'b_full1')
            z_mean = tf.nn.bias_add(tf.matmul(conv2_flat, W_full1), b_full1, name='z_mean')
            W_full2 = self.weight_variable([7 * 7 * 32, 2], 'W_full2')
            b_full2 = self.bias_variable([2], 'b_full2')
            z_stddev = tf.nn.bias_add(tf.matmul(conv2_flat, W_full2), b_full2, name='z_stddev')

            return z_mean, z_stddev

    def decoder(self, z):
        # deconv
        with tf.name_scope('decoder'):
            W_full = self.weight_variable([2, 7 * 7 * 32], 'W_full')
            b_full = self.bias_variable([7 * 7 * 32], 'b_full')
            z_full = tf.nn.bias_add(tf.matmul(z, W_full), b_full, name='z_full')
            z_matrix = tf.nn.relu(tf.reshape(z_full, [self.batch_size, 7, 7, 32]), name='z_matrix')
            # attention 5 5 16 32 not 5 5 32 16
            W_h1 = self.weight_variable([5, 5, 16, 32], 'W_h1')
            h1 = tf.nn.relu(tf.nn.conv2d_transpose(z_matrix, W_h1, [self.batch_size, 14, 14, 16],
                                                   strides=[1, 2, 2, 1], padding="SAME"), name='h1')

            W_h2 = self.weight_variable([5, 5, 1, 16], 'W_h2')
            # attention sigmoid
            h2 = tf.nn.sigmoid(tf.nn.conv2d_transpose(h1, W_h2, [self.batch_size, 28, 28, 1],
                                                      strides=[1, 2, 2, 1], padding="SAME"), name='h2')

            return h2

    def cluster(self, data):
        # pca = PCA(n_components=2)
        # pca_data = pca.fit_transform(data)
        pca_data = data
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=self.test_label)
        plt.colorbar()
        plt.show()

    def train(self):
        with tf.name_scope('input'):
            X = tf.placeholder(tf.float32, [None, self.input_size], name='input')
        z_mean, z_stddev = self.encoder(X)
        with tf.name_scope('sample'):
            samples = tf.random_normal([self.batch_size, 2], 0, 1, dtype=tf.float32)
            z = tf.add(tf.multiply(samples, z_stddev), z_mean, name='z')  # multiply not matmul
        generated_images = self.decoder(z)
        generated_flat = tf.reshape(generated_images, [self.batch_size, 28 * 28 * 1])

        with tf.name_scope('loss'):
            generation_loss = -tf.reduce_sum(X * tf.log(1e-7 + generated_flat) +
                                             (1 - X) * tf.log(1e-7 + 1 - generated_flat), 1, name='generation_loss')
            latent_loss = -0.5 * tf.reduce_sum(1 + tf.log(1e-7 + tf.square(z_stddev)) -
                                               tf.square(z_mean) - tf.square(z_stddev), 1, name='latent_loss')
            loss = tf.reduce_mean(generation_loss + latent_loss, name='loss')
            tf.summary.scalar('loss', loss)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        # saver
        saver = tf.train.Saver()
        best_loss = 10000
        best_sess = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # merge summary
            merged = tf.summary.merge_all()
            # choose dir
            writer = tf.summary.FileWriter('F:/tf_board/vae_mnist', sess.graph)
            # cal total_batch
            total_batch = int(self.train_data.num_examples/self.batch_size)
            for epoch in range(self.training_epochs):
                for i in range(total_batch):
                    batch_xs, batch_ys = self.train_data.next_batch(self.batch_size)  # y unused
                    sess.run(optimizer, feed_dict={X: batch_xs})
                    if (epoch * total_batch + i) % self.display_step == 0:
                        train_loss, merge_result = sess.run([loss, merged], feed_dict={X: batch_xs})
                        # train_loss = sess.run(loss, feed_dict={X: batch_xs})
                        print('step %d, training loss %g' %
                              (epoch * total_batch + i, train_loss))
                        # merge_result = sess.run(merged, feed_dict={X: batch_xs})
                        writer.add_summary(merge_result, epoch * total_batch + i)

                        if train_loss < best_loss:
                            best_loss = train_loss
                            best_sess = sess
            print('train done')
            # save sess
            saver.save(best_sess, '/root/vae_mnist.ckpt')

            # test
            example_y = sess.run(generated_flat, feed_dict={X: self.test_data[:self.batch_size]})
            f, a = plt.subplots(2, self.examples_to_show, figsize=(self.examples_to_show, 2))
            for i in range(self.examples_to_show):
                a[0][i].imshow(np.reshape(self.test_data[i], (28, 28)), cmap="gray")
                a[1][i].imshow(np.reshape(example_y[i], (28, 28)), cmap="gray")
            plt.show()

            # draw
            nx = ny = 20
            x_values = np.linspace(-3, 3, nx)
            y_values = np.linspace(-3, 3, ny)
            canvas = np.empty((28*ny, 28*nx))
            for i, yi in enumerate(x_values):
                for j, xi in enumerate(y_values):
                    z_mu = np.array([[xi, yi]]*self.batch_size)
                    x_mean = sess.run(generated_flat, feed_dict={z: z_mu})
                    canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)

            plt.figure(figsize=(8, 10))
            Xi, Yi = np.meshgrid(x_values, y_values)
            plt.imshow(canvas, origin="upper", cmap="gray")
            plt.tight_layout()
            plt.show()

            # cluster
            samples_cluster = tf.random_normal([len(self.test_data), 2], 0, 1, dtype=tf.float32)
            z_cluster = tf.add(tf.multiply(samples_cluster, z_stddev), z_mean)  # multiply not matmul
            cluster_data = sess.run(z_cluster, feed_dict={X: self.test_data})
            self.cluster(cluster_data)


if __name__ == '__main__':
    # load data mnist default
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    basic_AE = Basic_VAE(mnist.train, mnist.test.images, np.argmax(mnist.test.labels, 1), 0.001, 784, 30, 128, 200, 10)
    basic_AE.train()
