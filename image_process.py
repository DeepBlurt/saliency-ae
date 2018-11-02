#!/usr/bin/python
from __future__ import print_function, division
import tensorflow as tf
import os
import numpy as np

PATCH_SIZE = 15
CENTER_SIZE = 7
DIMEN1 = PATCH_SIZE ** 2 * 3
DIMEN2 = CENTER_SIZE ** 2 * 3
BIAS = 4


def image_process(image_path, image_type='jpeg'):
    """
    
    :param image_path: image file path or name: string
    :param image_type: type of image(png or jpg): String
    :return: processed image: three dimension matrix tensor 
    """
    # read and decode:
    image_raw = tf.gfile.FastGFile(image_path, 'r').read()
    if image_type == 'png':
        image_data = tf.image.decode_png(image_raw)
    else:
        image_data = tf.image.decode_jpeg(image_raw)
    # change data type and standardize:
    image = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    image_stand = tf.div(image, tf.reduce_max(image))

    return image_stand


def mask_process(mask_path, mask_channels, mask_type='jpeg'):
    """

    :param mask_path: image file path or name: string
    :param mask_channels: channels of mask images: integer
    :param mask_type: type of image(png or jpg): String
    :return: processed images: three dimension matrix tensor dictionary
    """
    # read and decode:
    mask_raw = tf.gfile.FastGFile(mask_path, 'r').read()
    if mask_type == 'png':
        mask_data = tf.image.decode_image(mask_raw)
    else:
        mask_data = tf.image.decode_jpeg(mask_raw, mask_channels)
    # change data type
    mask = tf.image.convert_image_dtype(mask_data, dtype=tf.float32)
    mask_stand = tf.div(mask, tf.reduce_max(mask))

    return mask_stand


def make_examples(image_name, image_path='Data_set/train/color stimuli', m=8000, image_shape=(400, 300)):
    """
    make examples for train
    :param image_name: name of image :string
    :param image_path: path to store images: String
    :param m: sampling numbers: integer
    :param image_shape: image width and height: tuple
    :return: list of examples: list
    """
    x = os.getcwd()
    image_path = x + '/' + image_path
    example_for_train = list()

    image_raw = image_process(image_path+'/'+image_name, image_type='jpeg')
    image = tf.reshape(image_raw, [image_shape[1], image_shape[0], 3])
    x = tf.random_uniform([1], 0, image_shape[1] - PATCH_SIZE, dtype=tf.int32)
    y = tf.random_uniform([1], 0, image_shape[0] - PATCH_SIZE, dtype=tf.int32)
    image_patch = tf.image.crop_to_bounding_box(image, x[0], y[0], PATCH_SIZE, PATCH_SIZE)
    gt_patch = tf.image.crop_to_bounding_box(image, x[0] + BIAS, y[0] + BIAS, CENTER_SIZE, CENTER_SIZE)
    image_vector = tf.reshape(image_patch, [1, DIMEN1])
    gt_vector = tf.reshape(gt_patch, [1, DIMEN2])
    # sampling images patches and ground truth patches:
    print('processing img:', image_name)
    with tf.Session() as sess:
        for i in range(m):
            image_array, gt_array = sess.run([image_vector, gt_vector])
            example_for_train.append((image_array, gt_array))
    return example_for_train


def saliency(au, image_name, image_path='Data_set/test/color stimuli', image_shape=(400, 300)):
    """
    read images and its every pixels' surroundings:
    :param au: auto-encoder object
    :param image_name: name of image:String
    :param image_path: path to store images: String
    :param image_shape: image width and height: tuple
    :return: list of examples: list
    """
    x = os.getcwd()
    image_path = x + '/' + image_path
    salient_map = np.zeros([image_shape[1], image_shape[0]], np.float32)
    # reconstruct = np.zeros([image_shape[1], image_shape[0], 3])
    image_raw = image_process(image_path+'/'+image_name, image_type='jpeg')
    image = tf.reshape(image_raw, [image_shape[1], image_shape[0], 3])

    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.int32)
    re_vector = tf.placeholder(tf.float32, [1, DIMEN2])

    image_patch = tf.image.crop_to_bounding_box(image, x, y, PATCH_SIZE, PATCH_SIZE)
    image_vector = tf.reshape(image_patch, [1, DIMEN1])
    gt_patch = tf.image.crop_to_bounding_box(image, x + BIAS, y + BIAS, CENTER_SIZE, CENTER_SIZE)
    gt_vector = tf.reshape(gt_patch, [1, DIMEN2])

    residual = tf.reduce_mean(tf.square(re_vector - gt_vector))
    print('predicting saliency of img:', image_name)
    with tf.Session() as sess:
        for x_val in range(image_shape[1]-PATCH_SIZE):
            for y_val in range(image_shape[0]-PATCH_SIZE):
                image_patch = sess.run(image_vector, feed_dict={x: x_val, y: y_val})
                real = au.get_center(image_patch)
                # recon = au.get_decoded_x(image_patch)
                # if (x_val == 100 and y_val == 100) or (x_val == 120 and y_val == 120):
                # print(real)
                # compute the residual:
                res = sess.run(residual, feed_dict={x: x_val, y: y_val, re_vector: real})
                salient_map[x_val+CENTER_SIZE, y_val+CENTER_SIZE] = res
        # maybe I should normalize the salient map's value to [0, 1]
    a = salient_map.max()
    salient_map = salient_map/a

    return salient_map*255
