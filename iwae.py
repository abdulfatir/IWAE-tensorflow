import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow import keras


class IWAE:
    def __init__(self, z_dim=50, k=5, test_k=5000, n_steps=200000, batch_size=100):
        self.z_dim = z_dim
        self.k = k
        self.test_k = test_k
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.data_dim = 784
        self._build_model()
        self._build_test_graph()
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(init_op)
        self._load_data()

    def _load_data(self):
        mnist = keras.datasets.mnist
        (self.train_images, _), (self.test_images, _) = mnist.load_data()
        self.train_images = self.train_images/255.
        self.test_images = self.test_images/255.

    def _encoder(self, x, z_dim=20, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):
            l1 = tf.layers.dense(x, 200, activation=tf.nn.relu)
            l2 = tf.layers.dense(l1, 200, activation=tf.nn.relu)
            mu = tf.layers.dense(l2, z_dim, activation=None)
            sigma = 1e-6 + \
                tf.nn.softplus(tf.layers.dense(l2, z_dim, activation=None))
            return mu, sigma

    def _decoder(self, z, z_dim=20, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):
            l1 = tf.layers.dense(z, 200, activation=tf.nn.relu)
            l2 = tf.layers.dense(l1, 200, activation=tf.nn.relu)
            x_hat = tf.layers.dense(
                l2, self.data_dim, activation=tf.nn.sigmoid)
            return x_hat

    def _objective(self, z, mu, sigma, x, x_hat, training=True):
        log2pi = tf.log(2 * np.pi)
        log_QzGx = (-(self.z_dim / 2) * log2pi
                    + tf.reduce_sum(- tf.log(sigma) - 0.5 * tf.squared_difference(z, mu) / (2 * tf.square(sigma)), -1))
        log_PxGz = tf.reduce_mean(tf.reduce_sum(
            x * tf.log(x_hat + 1e-8) + (1 - x) * tf.log(1 - x_hat + 1e-8), [1]))
        log_Pz = (-(self.z_dim / 2) * log2pi
                  + tf.reduce_sum(- 0.5 * tf.squared_difference(z, 0) / 2, -1))
        if training:
            log_weights = tf.reshape(
                log_PxGz + log_Pz - log_QzGx, [self.k, self.batch_size])
            weights = tf.exp(log_weights - tf.reduce_max(log_weights, 0))
            normalized_weights = weights / tf.reduce_sum(weights, 0)
            loss = - \
                tf.reduce_mean(tf.reduce_sum(
                    normalized_weights * log_weights, 0))
        else:
            log_weights = tf.reshape(
                log_PxGz + log_Pz - log_QzGx, [self.test_k, 1])
            log_wmax = tf.reduce_max(log_weights, 0)
            weights = tf.exp(log_weights - log_wmax)
            loss = - \
                tf.reduce_mean(tf.log(tf.reduce_mean(weights, 0))
                               ) - tf.reduce_mean(log_wmax)
        return loss

    def _build_model(self):
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.data_dim])
        x_k = tf.tile(self.x, [self.k, 1])
        mu, sigma = self._encoder(x_k, z_dim=self.z_dim)
        z = mu + sigma * \
            tf.random_normal(
                [self.k * self.batch_size, self.z_dim], 0, 1, dtype=tf.float32)
        x_hat = self._decoder(z)
        self.loss = self._objective(z, mu, sigma, x_k, x_hat)
        self.optim_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def _build_test_graph(self):
        self.x_test = tf.placeholder(tf.float32, [1, self.data_dim])
        x_k_test = tf.tile(self.x_test, [self.test_k, 1])
        mu_test, sigma_test = self._encoder(x_k_test, z_dim=self.z_dim, reuse=True)
        z_test = mu_test + sigma_test * \
            tf.random_normal([self.test_k * 1, self.z_dim],
                             0, 1, dtype=tf.float32)
        x_hat_test = self._decoder(z_test, reuse=True)
        self.test_loss = self._objective(
            z_test, mu_test, sigma_test, x_k_test, x_hat_test, training=False)

    def _batch_generator(self, X, batch_size):
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        while True:
            permutation = np.random.permutation(X.shape[0])
            for b in range(n_batches):
                batch_idx = permutation[b *
                                        batch_size:(b + 1) * batch_size]
                batch = X[batch_idx]
                if batch.shape[0] is not batch_size:
                    continue
                yield batch

    def compute_test_loss(self):
        running_test_loss = 0.
        for i, img in enumerate(self.test_images):
            x_np = img.reshape(1, self.data_dim)
            loss_np = self.sess.run(
                self.test_loss, feed_dict={self.x_test: x_np})
            running_test_loss += loss_np
        return running_test_loss / self.test_images.shape[0]

    def train(self):
        train_gen = self._batch_generator(self.train_images, self.batch_size)
        start_time = time.time()
        for stp in range(1, self.n_steps + 1):
            x_np = next(train_gen).reshape(self.batch_size, self.data_dim)
            _, loss_np = self.sess.run([self.optim_op, self.loss], feed_dict={self.x: x_np})
            if stp % 5000 == 0:
                end_time = time.time()
                print('Step: {:d} in {:.2f}s :: Loss: {:.3f}'.format(
                    stp, end_time - start_time, loss_np))
                start_time = end_time
