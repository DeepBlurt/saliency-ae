#!/usr/bin/python
from __future__ import print_function, division
import numpy as np
import tensorflow as tf


def xavier_init(fan_in, fan_out, trans_function):
    """
    xavier_init function: to initialize weights in auto-encoder
    :param fan_in: input dimension: int 
    :param fan_out: output dimension: int
    :param trans_function: transfer function: function handle
    :return: initialized variables: tf.Variables
    """
    if trans_function is tf.nn.sigmoid:
        low = -4.0 * np.sqrt(6.0 / (fan_in + fan_out))
        high = 4.0 * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)
    elif trans_function is tf.nn.tanh:
        low = -1 * np.sqrt(6.0 / (fan_in + fan_out))
        high = 1 * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class Autoencoder(object):
    def __init__(self, input_dimension, layer_sizes, layer_names, infer_layer_output_size, tied_weights=False,
                 optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001), transfer_function=tf.nn.sigmoid):
        """
        init the autoencoder layers and variables
        :param input_dimension: input dimensionality: int
        :param layer_sizes: every layer's output dimensionality: list of integers
        :param layer_names:  layers names: list of String 
        :param infer_layer_output_size: inference layer output size: int 
        :param tied_weights: using tied weights or not: bool
        :param optimizer: optimizer function: function handle
        :param transfer_function: transfer function: function handle
        """
        self.layer_names = layer_names
        self.layer_size = layer_sizes
        self.tied_weights = tied_weights
        self.input_dimension = input_dimension
        self.transfer_function = transfer_function

        # Build the encoding layers:

        self.x = tf.placeholder(tf.float32, [1, input_dimension])
        next_layer_input = self.x
        # use the variable to iterate

        # check parameters
        assert len(layer_sizes) == len(layer_names)
        self.encode_part_weights = list()
        self.encode_part_biases = list()
        # 编码部分变量
        for i in range(len(self.layer_size)):
            # get the current layer's output size
            dimension = self.layer_size[i]
            # compute current layer's input dimensionality
            input_dim = int(next_layer_input.get_shape()[1])

            # initialize weights using xavier initialization
            weights = tf.Variable(xavier_init(input_dim, dimension, self.transfer_function), name=layer_names[i][0])
            biases = tf.Variable(tf.zeros([dimension]), name=layer_names[i][1])

            # store weights for latter reference
            self.encode_part_weights.append(weights)
            self.encode_part_biases.append(biases)

            output = transfer_function(tf.add(tf.matmul(next_layer_input, weights), biases))
            next_layer_input = output

        self.encoded_x = next_layer_input

        # Building decoding layers:
        # 解码部分变量
        layer_sizes.reverse()
        self.encode_part_weights.reverse()

        self.decode_part_weights = list()
        self.decode_part_biases = list()
        for i, dim in enumerate(layer_sizes[1:] + [int(self.x.get_shape()[1])]):
            # when using tied weights, we can just transpose the encoding part's weights matrix
            if self.tied_weights:
                weights = tf.identity(tf.transpose(self.encode_part_weights[i]))
            else:
                weights = tf.Variable(xavier_init(self.encode_part_weights[i].get_shape()[1].value,
                                                  self.encode_part_weights[i].get_shape()[0].value,
                                                  trans_function=self.transfer_function))
            biases = tf.Variable(tf.zeros([dim]))
            self.decode_part_weights.append(weights)
            self.decode_part_biases.append(biases)

            output = self.transfer_function(tf.add(tf.matmul(next_layer_input, weights), biases))
            next_layer_input = output

        # ultimate output is decoded_x:
        self.decoded_x = next_layer_input

        # add the inference layer:
        self.infer_output_size = infer_layer_output_size
        self.infer_input = self.decoded_x
        self.infer_weights = tf.Variable(tf.random_normal([self.input_dimension, self.infer_output_size], mean=0,
                                                          stddev=0.1, dtype=tf.float32), name='inference_layer_weights')
        self.infer_biases = tf.Variable(tf.zeros([self.infer_output_size], dtype=tf.float32),
                                        name='inference_layer_biases')
        self.final_output = self.transfer_function(tf.add(tf.matmul(self.infer_input, self.infer_weights),
                                                          self.infer_biases))
        # final_output is the output of generative model

        # reverse back encode_part_weights
        self.encode_part_weights.reverse()
        self.decode_part_weights.reverse()

        # compute cost, definition of optimizer: this is for only training the autoencoder
        # not for training the whole network.
        self.ground_truth = tf.placeholder(tf.float32, [1, infer_layer_output_size])
        self.cost = tf.abs(tf.reduce_mean(tf.add(tf.multiply(self.ground_truth, tf.log(self.final_output)),
                                                 tf.multiply((1-self.ground_truth), tf.log(1-self.final_output)))))
        # self.cost = tf.reduce_mean(tf.square(self.final_output-self.ground_truth))

        self.optimizer = optimizer.minimize(self.cost)

        # initialize all variables
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def get_encoded_x(self, input_x):
        """
        get the encoded value of input X
        :param input_x: input: tf.placeholder
        :return: encoded_x: tf.Variable
        """
        return self.sess.run(self.encoded_x, feed_dict={self.x: input_x})

    def get_decoded_x(self, input_x):
        """
        get the decoded value of input X
        :param input_x: input: tf.placeholder
        :return: decoded_x: tf.Variable
        """
        return self.sess.run(self.decoded_x, feed_dict={self.x: input_x})

    def get_center(self, input_x):
        """
        get the predicted center patch of the given image
        # 得到推断曾的输出值
        :param input_x: input_data: vector
        :return: self.final_output
        """
        return self.sess.run(self.final_output, feed_dict={self.x: input_x})

    def load_rbm_weights(self, path, layer_names, layer):
        """
        load RBM weights to auto-encoder
        :param path: file path: String
        :param layer_names: name of encoding part and decoding part: String list
        :param layer: indices of layer to load weights: int
        :return: None
        """
        # loading RBM weights does not included the inference layer
        saver = tf.train.Saver({layer_names[0]: self.encode_part_weights[layer]},
                               {layer_names[1]: self.encode_part_biases[layer]})
        saver.restore(self.sess, path)
        if not self.tied_weights:
            self.sess.run(self.decode_part_weights[layer].assign(tf.transpose(self.encode_part_weights[layer])))

    def load_weights(self, path):
        """
        load all weights from path
        加载自编码器的参数
        :param path: file path: string
        :return:None 
        """
        dict_weights = self.get_parameter_dict()
        saver = tf.train.Saver(dict_weights)
        saver.restore(self.sess, path)

    def save_weights(self, path):
        """
        save all weights to path
        :param path: file path
        :return: save_path: path of saved file: a string
        """
        dict_parameter = self.get_parameter_dict()
        saver = tf.train.Saver(dict_parameter)
        save_path = saver.save(self.sess, path)
        return save_path
        # save everything in save_path

    def get_parameter_dict(self):
        """
        get a dictionary of layer names to weights and biases
        :return: weights and biases in a dictionary
        """
        dict_weights = dict()
        for i in range(len(self.layer_names)):
            dict_weights[self.layer_names[i][0]] = self.encode_part_weights[i]
            dict_weights[self.layer_names[i][1]] = self.encode_part_biases[i]
            if not self.tied_weights:
                dict_weights[self.layer_names[i][0]+'d'] = self.decode_part_weights[i]
                dict_weights[self.layer_names[i][1]+'d'] = self.decode_part_biases[i]

        dict_weights['inference layer weights'] = self.infer_weights
        dict_weights['inference layer biases'] = self.infer_biases
        return dict_weights

    def train_au_only(self, input_x):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: input_x})
        return cost

    def print_weights(self):
        """
        print weights and biases
        :return: None
        """
        for i in range(len(self.encode_part_weights)):
            print('encode_part weights:', i)
            print(self.encode_part_weights[i].eval(self.sess).shape)
            print(self.encode_part_weights[i].eval(self.sess))
            if not self.tied_weights:
                print('decode_part weights:', i)
                print(self.decode_part_weights[i].eval(self.sess).shape)
                print(self.decode_part_weights[i].eval(self.sess))

        for i in range(len(self.encode_part_biases)):
            print('encode_part biases:', i)
            print(self.encode_part_biases[i].eval(self.sess).shape)
            print(self.encode_part_biases[i].eval(self.sess))
            if not self.tied_weights:
                print('decode_part biases:', i)
                print(self.decode_part_biases[i].eval(self.sess).shape)
                print(self.decode_part_biases[i].eval(self.sess))
        print('inference layer: weights:')
        print(self.infer_weights.eval(self.sess).shape)
        print(self.infer_weights.eval(self.sess))

        print('inference layer: biases:')
        print(self.infer_biases.eval(self.sess).shape)
        print(self.infer_biases.eval(self.sess))

    def train_whole_net(self, input_x, ground_truth):
        """
        train the whole network using cross-entropy error and BP algorithm
        :param input_x: input image: vector
        :param ground_truth: true label: vector
        :return: the value of cost function when session is running
        """
        error, _ = self.sess.run([self.cost, self.optimizer], feed_dict={self.x: input_x,
                                                                         self.ground_truth: ground_truth})
        return error
