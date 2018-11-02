#!/usr/bin/python
from __future__ import print_function, division

from RBM import Rbm
from image_process import *
from image_process_3D import *


PATCH_SIZE = 15


def pretrain(examples, names, hidden_size=(256, 128, 64, 32, 8), batch=200, epoch=40):
    """
    Training a stack of RBM for auto-encoder: list
    预先训练限制玻尔兹曼机函数
    :param: input_x: input list
    :param: names: names of rbm
    :return: None    :param: hidden_size: hidden_size of networks, choose the architecture of the paper suggested.

    """
    # define a stack of rbm:
    rbm_stack = list()
    current_input = PATCH_SIZE ** 2 * 3
    for i, dim in enumerate(hidden_size):
        current_output = dim
        if i == 4:
            rbm = Rbm(current_input, current_output, names[i], 0.001, linear=True)
            rbm_stack.append(rbm)
        else:
            rbm = Rbm(current_input, current_output, names[i], 0.1)
            rbm_stack.append(rbm)
        current_input = dim

    # train the rbm:
    momentum = [0.5, 0.6, 0.7, 0.8, 0.9]
    for i in range(epoch-5):
        momentum.append(0.9)
    err_sum = 0.0
    path = os.getcwd()
    with tf.Session() as sess:
        for i, rbm in enumerate(rbm_stack):
            print('training rbm', i)
            for j in range(epoch):
                for k in range(batch):
                    current_input = examples[batch*j+k][0]
                    output = current_input
                # compute needed input:
                for x in range(i):
                    output = rbm_stack[x].transform(current_input)
                    current_input = output
                err_sum = rbm.train_step(output, momentum[j])
                # print('after ', j, 'epochs, cost is :', err_sum)
            rbm.save_parameters(path+'/'+'rbm/rbm'+str(i)+'.para')

    # test the result:


if __name__ == '__main__':
    with tf.Session() as sess:
        example = make_examples('dataset for test/img', 'dataset for test/mask_jpg', 3, m=20)
        pretrain(example, batch=20, epoch=10)
