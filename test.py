from gaussian import *
import numpy as np
import os
import matplotlib.pyplot as plt


def evaluate(image_name, pred_array, path='saliency/2D fixation', shape=(300, 400)):
    gt = Image.open(path+'/'+image_name[5:-14])
    gt_array = np.array(gt)

    tp = np.zeros(257, dtype=np.float32)
    fp = np.zeros(257, dtype=np.float32)
    fn = np.zeros(257, dtype=np.float32)
    tn = np.zeros(257, dtype=np.float32)
    for threshold in range(257):
        for x in range(shape[0]):
            for y in range(shape[1]):
                if pred_array[x, y] > threshold-0.5:
                    if gt_array[x, y] > 0:
                        tp[256-threshold] += 1
                    elif gt_array[x, y] <= 0 :
                        fp[256-threshold] += 1
                elif pred_array[x, y] < threshold-0.5:
                    if gt_array[x, y] > 0:
                        fn[256-threshold] += 1
                    elif gt_array[x, y] <= 0:
                        tn[256-threshold] += 1
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    array = gt_array * np.log(gt_array / (pred_array + 0.0001) + 0.0001)
    kl = array.sum()/120000.0
    print(kl)
    return tpr, fpr, kl


pred_path_2 = 'saliency/2D'
pred_path_3 = 'saliency/3D'

pred_list_2 = os.listdir(pred_path_2)
pred_list_3 = os.listdir(pred_path_3)

filter_path_2 = 'saliency/filtered/2D'
filter_path_3 = 'saliency/filtered/3D'
tp = np.zeros(257)
fp = np.zeros(257)
kl_vector = np.zeros(len(pred_list_2))

print('2D evaluation:')
for i, image in enumerate(pred_list_2):
    print('processing img', image)
    im = Image.open(pred_path_2+'/'+image)
    im = im.filter(MyGaussianBlur(radius=3))
    im_array = np.array(im)
    a = im_array.max()
    im_array = (im_array.astype(np.float32)/a)*255
    im_array = im_array.astype(np.uint8)
    # print(im_array.max(), im_array.min())

    temp1, temp2, k = evaluate(image, im_array)
    tp = tp + temp1
    fp = fp + temp2
    kl_vector[i] = k

tp_rate = tp/len(pred_list_2)
fp_rate = fp/len(pred_list_2)
np.savetxt('2d_fpr.txt', fp_rate)
np.savetxt('2d_tpr.txt', tp_rate)
np.savetxt('2dkl.txt', kl_vector)

# need to delete:
plt.figure(0)
plt.plot(fp_rate, tp_rate)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.xlabel('false positive rate')
plt.title('ROC of 2D prediction')
plt.ylabel('true positive rate')
plt.show()
# plt.savefig('2droc.png')

kl_vector = np.zeros(len(pred_list_3))

tp = np.zeros(257)
fp = np.zeros(257)
print('3D evaluation:')
for i, image in enumerate(pred_list_3):
    print('processing img', image)
    im = Image.open(pred_path_3+'/'+image)
    im = im.filter(MyGaussianBlur(radius=3))
    im_array = np.array(im)
    a = im_array.max()
    im_array = (im_array.astype(np.float32) / a) * 255
    im_array = im_array.astype(np.uint8)

    temp1, temp2, k = evaluate(image, im_array, 'saliency/3D fixation')
    kl_vector[i] = k
    tp = tp + temp1
    fp = fp + temp2


tp_rate = tp/len(pred_list_3)
fp_rate = fp/len(pred_list_3)
np.savetxt('3d_fpr.txt', fp_rate)
np.savetxt('3d_tpr.txt', tp_rate)
np.savetxt('3dkl.txt', kl_vector)

plt.figure(1)
plt.plot(fp_rate, tp_rate)
plt.xlabel('false positive rate')
plt.title('3D evaluation')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.ylabel('true positive rate')
plt.show()
