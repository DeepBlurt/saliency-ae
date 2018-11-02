#!/usr/bin/python
from autoencoder import *
from pretrain import *
# from __future__ import print_function, division
# with tf.device('/gpu:1'):
# define autoencoder objects:
layer_names = [['rbm1w', 'rbm1b'],
               ['rbm2w', 'rbm2b'],
               ['rbm3w', 'rbm3b'],
               ['rbm4w', 'rbm4b'],
               ['rbm5w', 'rbm5b']]
au_2d = Autoencoder(DIMEN1, [256, 128, 64, 32, 8], layer_names, CENTER_SIZE)
au_3d = Autoencoder(DIMEN1, [256, 128, 64, 32, 8], layer_names, CENTER_SIZE)

# pretrain:

# 2d version:
print('2D version:')
examples = make_examples('dataset for test/img', 'dataset for test/mask_jpg', 3, m=20)
print('pre_training  process:')
pretrain(examples)
# load rbm weights:
for i, name in enumerate(layer_names):
    au_2d.load_rbm_weights('rbm/rbm'+str(i)+'.para', name, i)

# train the auto-encoder:
BATCH = 800
EPOCH = 200
error = 0
for i in range(EPOCH):
    for j in range(BATCH):
        index = i * BATCH + j
        error = au_2d.train_whole_net(examples[index][0], examples[index][1])
    print('after :', i, 'epochs, cost is :', error)
# save weights:
x = os.getcwd()
au_2d.save_weights(x+'/'+'auto-encoder-2d.para')

del examples

# 3d version:
print('2D version:')
examples = make_examples_3d()
print('pre_training  process:')
pretrain(examples)
# load rbm weights:
for i, name in enumerate(layer_names):
    au_3d.load_rbm_weights('rbm/rbm'+str(i)+'.para', name, i)

# train the auto-encoder:
BATCH = 800
EPOCH = 200
error = 0
for i in range(EPOCH):
    for j in range(BATCH):
        index = i * BATCH + j
        error = au_3d.train_whole_net(examples[index][0], examples[index][1])
    print('after :', i, 'epochs, cost is :', error)
# save weights:
x = os.getcwd()
au_2d.save_weights(x+'/'+'auto-encoder-3d.para')
