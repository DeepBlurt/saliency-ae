import numpy as np
import matplotlib.pyplot as plt

fpr = np.loadtxt('2d_fpr.txt')
tpr = np.loadtxt('2d_tpr.txt')

# plot 2D roc curve
plt.figure(0)
plt.plot(fpr, tpr)
plt.xlabel('false positive rate')
plt.title('2D evaluation')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.ylabel('true positive rate')
plt.show()


fpr = np.loadtxt('3d_fpr.txt')
tpr = np.loadtxt('3d_tpr.txt')
# plot 3D roc curve
plt.figure(1)
plt.plot(fpr, tpr)
plt.xlabel('false positive rate')
plt.title('3D evaluation')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.ylabel('true positive rate')
plt.show()

kl = np.loadtxt('2dkl.txt')
print('mean of 2d kl is:', kl.mean())

kl = np.loadtxt('3dkl.txt')
print('mean of 3d kl is:', kl.mean())
