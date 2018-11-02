#!/usr/bin/python
from __future__ import print_function, division
from autoencoder import *
from pretrain import *
from gaussian import *

BATCH = 200
EPOCH = 40

# define autoencoder objects:
rbm_names = [['rbm1w', 'rbm1vb', 'rbm1hb'],
             ['rbm2w', 'rbm2vb', 'rbm2hb'],
             ['rbm3w', 'rbm3vb', 'rbm3hb'],
             ['rbm4w', 'rbm4vb', 'rbm4hb'],
             ['rbm5w', 'rbm5vb', 'rbm5hb']]
layer_names = [['rbm1w', 'rbm1hb'],
               ['rbm2w', 'rbm2hb'],
               ['rbm3w', 'rbm3hb'],
               ['rbm4w', 'rbm4hb'],
               ['rbm5w', 'rbm5hb']]
au_2d = Autoencoder(DIMEN1, [256, 128, 64, 32, 8], layer_names, DIMEN2)
au_3d = Autoencoder(DIMEN1, [256, 128, 64, 32, 8], layer_names, DIMEN2)

# pretrain:

# 2d version:
print('2D version:')
image_path = 'Data_set/test/color stimuli'
image_list = os.listdir(image_path)

for image_name in image_list:
    examples = make_examples(image_name, image_path)
    print('pre_training  process:')
    pretrain(examples, rbm_names)
    # load rbm weights:
    for i, name in enumerate(layer_names):
        au_2d.load_rbm_weights('rbm/rbm'+str(i)+'.para', name, i)

    # train the auto-encoder:
    print('training the auto-encoder2d:')
    error = 0
    for i in range(EPOCH):
        for j in range(BATCH):
            index = i * BATCH + j
            error = au_2d.train_whole_net(examples[index][0], examples[index][1])
        print('after :', i, 'epochs, cost is :', error)
    # save weights:
    saliency_map = saliency(au_2d, image_name, image_path)
    im = Image.fromarray(saliency_map.astype(np.uint8))
    # im = im.filter(MyGaussianBlur(radius=7))
    x = os.getcwd()
    im.save(x+'/saliency/2D/'+image_name+'saliency2d.jpg')
    # au_2d.save_weights(x+'/auto-encoder/'+image_name+'auto-encoder-2d.para')

# 3d version:
print('3D version:')
image_path = 'Data_set/test/color stimuli'
depth_path = 'Data_set/test/depth'
image_list = os.listdir(image_path)
for image_name in image_list:
    examples = make_examples_3d(image_name, image_path, depth_path)
    print('pre_training  process:')
    pretrain(examples, rbm_names)
    # load rbm weights:
    for i, name in enumerate(layer_names):
        au_3d.load_rbm_weights('rbm/rbm'+str(i)+'.para', name, i)

    # train the auto-encoder:
    print('training the auto-encoder-3d:')
    error = 0
    for i in range(EPOCH):
        for j in range(BATCH):
            index = i * BATCH + j
            error = au_3d.train_whole_net(examples[index][0], examples[index][1])
        print('after :', i, 'epochs, cost is :', error)
    # save weights:
    saliency_map = saliency(au_3d, image_name, image_path)
    im = Image.fromarray(saliency_map.astype(np.uint8))
    x = os.getcwd()
    im = Image.fromarray(saliency_map.astype(np.uint8))
    # im = im.filter(MyGaussianBlur(radius=7))
    im.save(x + '/saliency/3D/' + image_name + 'saliency3d.jpg')
    # au_3d.save_weights(x+'/auto-encoder/'+image_name+'auto-encoder-3d.para')
