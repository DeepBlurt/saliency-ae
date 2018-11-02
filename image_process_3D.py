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


def mask_process(mask_path, channels, mask_type='jpeg'):
    """
    
    :param mask_path: image file path or name: string
    :param mask_type: type of image(png or jpg): String
    :param channels: channels of mask images
    :return: processed images: three dimension matrix tensor dictionary
    """
    # read and decode:
    mask_raw = tf.gfile.FastGFile(mask_path, 'r').read()
    if mask_type == 'png':
        mask_data = tf.image.decode_image(mask_raw, channels)
    else:
        mask_data = tf.image.decode_jpeg(mask_raw)
    # change data type
    mask = tf.image.convert_image_dtype(mask_data, dtype=tf.float32)
    mask_stand = tf.div(mask, tf.reduce_max(mask))

    return mask_stand


def make_examples_3d(image_name, image_path='Data_set/train/color stimuli', depth_path='Data_set/train/depth',
                     m=8000, image_shape=(400, 300)):
    """
    make examples for 3D training 
    :param image_name: name of image:String
    :param image_path: path to store images: String
    :param m: sampling numbers: integer
    :param depth_path: path of depth images: String
    :param image_shape: image width and height: tuple
    :return: list of examples: list
    """
    x = os.getcwd()
    image_path = x + '/' + image_path
    depth_path = x + '/' + depth_path
    example_for_train = list()

    image_raw = image_process(image_path+'/'+image_name, image_type='jpeg')
    depth_map = depth_process(depth_path + '/' + 'depth' + image_name[5:], depth_type='jpeg')
    image = tf.reshape(image_raw, [image_shape[1], image_shape[0], 3])
    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.int32)
    image_patch = tf.image.crop_to_bounding_box(image, x, y, PATCH_SIZE, PATCH_SIZE)
    gt_patch = tf.image.crop_to_bounding_box(image, x + BIAS, y + BIAS, CENTER_SIZE, CENTER_SIZE)
    image_vector = tf.reshape(image_patch, [1, DIMEN1])
    gt_vector = tf.reshape(gt_patch, [1, DIMEN2])
    # sampling images patches and ground truth patches:
    print('processing img:', image_name)
    with tf.Session() as sess:
        depth_array = sess.run(depth_map)
        for i in range(m):
            x_value, y_value = depth_guided_sample(depth_array, sess)
            val1, val2 = sess.run([image_vector, gt_vector], feed_dict={x: x_value, y: y_value})
            example_for_train.append((val1, val2))
    return example_for_train


def depth_process(depth_path, depth_type='jpeg'):
    """
    depth map
    :param depth_path: image file path or name: string
    :param depth_type: type of image(png or jpg): String
    :return: processed images: three dimension matrix tensor dictionary
    """
    # read and decode:
    depth_raw = tf.gfile.FastGFile(depth_path, 'r').read()
    if depth_type == 'png':
        mask_data = tf.image.decode_image(depth_raw, channels=1)
    else:
        mask_data = tf.image.decode_jpeg(depth_raw, channels=1)
    # change data type
    depth_map = tf.image.convert_image_dtype(mask_data, dtype=tf.float32)

    return depth_map


def depth_guided_sample(depth_map, sess, method=0, z0=0.5):
    """
    sampling pixels following distribution of depth priors
    :param depth_map: depth map array: np.ndarray
    :param sess: tf.Session()
    :param method: 
    :param z0: 
    :return: 
    """
    rand_depth = 256
    x = 0
    y = 0
    while int(depth_map[x, y] * 255) < rand_depth:
        rand_depth = np.random.randint(0, 256)
        x = np.random.randint(0, depth_map.shape[0]-PATCH_SIZE)
        y = np.random.randint(0, depth_map.shape[1]-PATCH_SIZE)
    # print(x, y, depth_array[x, y], rand_depth/256)
    return x, y


if __name__ == '__main__':
    x = os.getcwd()
    print(x)
    for k in range(20):
        with tf.Session() as sess:
            make_examples_3d(m=10)
