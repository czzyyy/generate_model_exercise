import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA


class BasicAE(object):
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
        initial = tf.random_normal(shape=shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    def encoder(self, X):
        # input_size->250->125->60
        with tf.name_scope('encoder'):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X,
                                                     self.weight_variable(shape=[self.input_size, 500],
                                                                          name='en_layer1_w')), self.bias_variable(
                shape=[500], name='en_layer1_b')), name='en_layer1')

            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,
                                                     self.weight_variable(shape=[500, 125],
                                                                          name='en_layer2_w')), self.bias_variable(
                shape=[125], name='en_layer2_b')), name='en_layer2')
            # without sigmoid layer_4 (-,+)
            layer_3 = tf.add(tf.matmul(layer_2, self.weight_variable(shape=[125, 10],
                                                                     name='en_layer3_w')), self.bias_variable(
                shape=[10], name='en_layer3_b'))

            return layer_3

    def decoder(self, X_code):
        # 60->125->250->input_size
        with tf.name_scope('decoder'):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X_code,
                                                     self.weight_variable(shape=[10, 125],
                                                                          name='de_layer1_w')), self.bias_variable(
                shape=[125], name='de_layer1_b')), name='de_layer1')
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,
                                                     self.weight_variable(shape=[125, 500],
                                                                          name='de_layer2_w')), self.bias_variable(
                shape=[500], name='de_layer2_b')), name='de_layer2')

            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2,
                                                     self.weight_variable(shape=[500, self.input_size],
                                                                          name='de_layer3_w')), self.bias_variable(
                shape=[self.input_size], name='de_layer3_b')), name='de_layer3')

            return layer_3

    def cluster(self, data):
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=self.test_label)
        plt.colorbar()
        plt.show()

    def train(self):
        with tf.name_scope('inputs'):
            X = tf.placeholder(tf.float32, [None, self.input_size], name='input_X')
        with tf.name_scope('encode_decode'):
            X_code = self.encoder(X)
            y_pred = self.decoder(X_code)
            y_true = X
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.pow(y_pred - y_true, 2))
            tf.summary.scalar('loss', loss)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        # accuracy
        with tf.name_scope('accuracy'):
            accuracy = tf.exp(-loss)
            tf.summary.scalar('accuracy', accuracy)
        # saver
        saver = tf.train.Saver()
        best_loss = 10000
        best_sess = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # merge summary
            merged = tf.summary.merge_all()
            # choose dir
            writer = tf.summary.FileWriter('F:/tf_board/basic_ae_mnist', sess.graph)
            # cal total_batch
            total_batch = int(self.train_data.num_examples/self.batch_size)
            for epoch in range(self.training_epochs):
                for i in range(total_batch):
                    batch_xs, batch_ys = self.train_data.next_batch(self.batch_size)  # y unused
                    sess.run(optimizer, feed_dict={X: batch_xs})
                    if (epoch * total_batch + i) % self.display_step == 0:
                        train_accuracy = sess.run(accuracy, feed_dict={X: batch_xs})
                        train_loss = sess.run(loss, feed_dict={X: batch_xs})
                        print('step %d, training accuracy %g , loss %g' %
                              (epoch * total_batch + i, train_accuracy, train_loss))
                        merge_result = sess.run(merged, feed_dict={X: batch_xs})
                        writer.add_summary(merge_result, epoch * total_batch + i)

                        if train_loss < best_loss:
                            best_loss = train_loss
                            best_sess = sess
            print('train done')
            # save sess
            saver.save(best_sess, '/root/basic_ae_mnist.ckpt')

            # test
            example_y = sess.run(y_pred, feed_dict={X: self.test_data[:self.examples_to_show]})
            f, a = plt.subplots(2, self.examples_to_show, figsize=(self.examples_to_show, 2))
            for i in range(self.examples_to_show):
                a[0][i].imshow(np.reshape(self.test_data[i], (28, 28)))
                a[1][i].imshow(np.reshape(example_y[i], (28, 28)))
            plt.show()


            # cluster
            test_code = sess.run(X_code, feed_dict={X: self.test_data})
            self.cluster(test_code)


if __name__ == '__main__':
    # load data mnist default
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    basic_AE = BasicAE(mnist.train, mnist.test.images, np.argmax(mnist.test.labels, 1), 0.01, 784, 40, 256, 500, 10)
    basic_AE.train()
