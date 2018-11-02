#!/usr/bin/python
from __future__ import print_function, division
# logistic or linear: look up to the comment
# update_w may have questions
# momentum method: finished
import numpy as np
import tensorflow as tf


class Rbm(object):
    def __init__(self, n_input, n_hidden, layer_name, learning_rate,
                 transfer_function=tf.nn.sigmoid, linear=False):
        """
        Initialize RBM
        :param n_input: input dimension: int 
        :param n_hidden: hidden dimension: int 
        :param layer_name: names of parameter: String
        :param learning_rate: learning rate of training RBM: float
        :param transfer_function: trans_function handle: function handle
        :param linear: linear or logistic: bool
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.layer_names = layer_name
        self.linear = linear
        self.learning_rate = learning_rate

        # utilizing momentum:
        self.former_update_w = np.zeros([self.n_input, self.n_hidden], np.float32)
        self.former_update_vb = np.zeros([self.n_input], np.float32)
        self.former_update_hb = np.zeros([self.n_hidden], np.float32)

        # initialize networks parameters
        network_parameters = self._initialize_parameters()
        self.parameters = network_parameters

        # place holders
        self.x = tf.placeholder(tf.float32, [1, self.n_input])
        self.rbm_w = tf.placeholder(tf.float32, [self.n_input, self.n_hidden])
        self.rbm_vb = tf.placeholder(tf.float32, [self.n_input])
        self.rbm_hb = tf.placeholder(tf.float32, [self.n_hidden])

        # initialize variables to small random values chosen from a zero-mean and Gaussian with
        # standard deviation of 0.1.
        self.n_w = np.zeros([self.n_input, self.n_hidden], np.float32)
        self.n_vb = np.zeros([self.n_input], np.float32)
        self.n_hb = np.zeros([self.n_hidden], np.float32)
        # these numpy variables are used to update weights and biases during training process

        self.o_vb = np.zeros([self.n_input], np.float32)
        self.o_hb = np.zeros([self.n_hidden], np.float32)
        self.o_w = np.random.normal(0.0, 0.1, [self.n_input, self.n_hidden])
        # these variables are used to compute transformed values, training process and so on.
        # 这些变量也是临时变量

        # Using Gibbs sampling method
        self.h0_prob = transfer_function(tf.add(tf.matmul(self.x, self.rbm_w), self.rbm_hb))
        if self.linear:
            self.h0 = self.h0_prob + tf.random_normal([self.n_hidden])
        else:
            self.h0 = self.sample_prob(self.h0_prob)
        # h0 is a logistic state

        # compute new visible unit(linear)
        self.v1 = transfer_function(tf.add(tf.matmul(self.h0_prob, tf.transpose(self.rbm_w)), self.rbm_vb))
        # compute new hidden unit(linear)
        self.h1 = transfer_function(tf.add(tf.matmul(self.v1, self.rbm_w), self.rbm_hb))

        # update parameters and biases:
        # compute gradients
        self.w_positive_grad = tf.matmul(tf.transpose(self.x), self.h0_prob)
        # self.w_positive_grad = tf.matmul(self.x, tf.transpose(self.h0))
        self.w_negative_grad = tf.matmul(tf.transpose(self.v1), self.h1)
        # self.w_negative_grad = tf.matmul(self.v1, tf.transpose(self.h1))

        # stochastic steepest ascent, compute the increase value:
        self.increase_w = learning_rate * (self.w_positive_grad - self.w_negative_grad) / tf.to_float(
            tf.shape(self.x)[0])  # division here !!!shouldn't been here
        self.increase_vb = learning_rate * tf.reduce_mean(self.x - self.v1, 0)
        self.increase_hb = learning_rate * tf.reduce_mean(self.h0_prob - self.h1, 0)

        # sample to get reconstruction
        # 计算重构的结果
        self.h_sample = transfer_function(tf.add(tf.matmul(self.x, self.rbm_w), self.rbm_hb))
        self.v_sample = transfer_function(tf.add(tf.matmul(self.h_sample, tf.transpose(self.rbm_w)), self.rbm_vb))

        # cost function
        self.err_sum = tf.reduce_mean(tf.square(self.x - self.v_sample))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def compute_cost(self, batch):
        """
        compute cost under current parameters
        :param batch: input batch: vector
        :return: cost: float
        """
        return self.sess.run(self.err_sum, feed_dict={self.x: batch,
                                                      self.rbm_w: self.o_w,
                                                      self.rbm_vb: self.o_vb,
                                                      self.rbm_hb: self.o_hb})

    @staticmethod
    def sample_prob(probs):
        """
        perform sample operation to get logistic state of visible unit or hidden unit of RBM
        :param probs: probability: float
        :return: 1 or 0: bool
        """
        # if self.linear:
        return tf.nn.relu(tf.sign(probs-tf.random_uniform(tf.shape(probs))))
        # else:
        #    return tf.nn.re

    def _initialize_parameters(self):
        """
        initialize RBM parameters
        :return: parameters of current RBM: dictionary
        """
        all_parameters = dict()
        all_parameters['w'] = tf.Variable(tf.random_normal([self.n_input, self.n_hidden],
                                                           stddev=0.1, dtype=tf.float32), name=self.layer_names[0])
        all_parameters['vb'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32), name=self.layer_names[1])
        all_parameters['hb'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32), name=self.layer_names[2])
        return all_parameters

    def transform(self, batch_x):
        """
        compute output,实际中这个方法没啥用，可以用来调试啥的
        :param batch_x: input data: vector 
        :return: output: if linear==True: float else: bool 
        """
        if self.linear:
            return self.sess.run(self.h_sample, feed_dict={self.x: batch_x,
                                                           self.rbm_w: self.o_w,
                                                           self.rbm_hb: self.o_hb,
                                                           self.rbm_vb: self.o_vb})
        else:
            return self.sess.run(self.sample_prob(self.h_sample), feed_dict={self.x: batch_x,
                                                                             self.rbm_w: self.o_w,
                                                                             self.rbm_hb: self.o_hb,
                                                                             self.rbm_vb: self.o_vb})

    def restore_parameters(self, path):
        """
        restore parameters from saver offed by tensorflow
        :param path: file path: String 
        :return: None
        """
        saver = tf.train.Saver({self.layer_names[0]: self.parameters['w'],
                                self.layer_names[1]: self.parameters['vb'],
                                self.layer_names[2]: self.parameters['hb']})
        saver.restore(self.sess, path)
        self.o_w = self.parameters['w'].eval(self.sess)
        self.o_vb = self.parameters['vb'].eval(self.sess)
        self.o_hb = self.parameters['hb'].eval(self.sess)

    def save_parameters(self, path):
        """
        save parameters to file using tf.saver() 
        :param path: file path: String
        :return: None
        """
        self.sess.run(self.parameters['w'].assign(self.o_w))
        self.sess.run(self.parameters['vb'].assign(self.o_vb))
        self.sess.run(self.parameters['hb'].assign(self.o_hb))

        saver = tf.train.Saver({self.layer_names[0]: self.parameters['w'],
                                self.layer_names[1]: self.parameters['vb'],
                                self.layer_names[2]: self.parameters['hb']})
        saver.save(self.sess, path)

    def get_parameters(self):
        return self.parameters

    def get_hidden_weights_as_np(self):
        return self.n_w

    def train_step(self, batch_x, momentum):
        """
        main training process
        :param batch_x: input: numpy vector 
        :param momentum: momentum of current training process
        :return: error or cost: tf.float32
        """
        inc_w, inc_vb, inc_hb = self.sess.run([self.increase_w, self.increase_vb, self.increase_hb],
                                              feed_dict={self.x: batch_x,
                                                         self.rbm_w: self.o_w,
                                                         self.rbm_vb: self.o_vb,
                                                         self.rbm_hb: self.o_hb})
        # compute total increase:
        self.former_update_w = momentum * self.former_update_w + inc_w
        self.former_update_vb = momentum * self.former_update_vb + inc_vb
        self.former_update_hb = momentum * self.former_update_hb + inc_hb

        # refresh:
        self.n_w = self.o_w + self.former_update_w
        self.n_vb = self.o_vb + self.former_update_vb
        self.n_hb = self.o_hb + self.former_update_hb

        self.o_w = self.n_w
        self.o_vb = self.n_vb
        self.o_hb = self.n_hb

        return self.sess.run(self.err_sum, feed_dict={self.x: batch_x,
                                                      self.rbm_w: self.n_w,
                                                      self.rbm_vb: self.n_vb,
                                                      self.rbm_hb: self.n_hb})

    def get_reconstruct(self, batch_x):
        # 获取重构值，调试可以用一下
        return self.sess.run(self.v_sample, feed_dict={self.x: batch_x,
                                                       self.rbm_w: self.o_w,
                                                       self.rbm_vb: self.o_vb,
                                                       self.rbm_hb: self.o_hb})
