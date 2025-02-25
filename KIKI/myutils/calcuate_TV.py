import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

path = r"D:\code\sparseMRI_v0.2\TV_3X.mat"

matdata = sio.loadmat(path)
# print(matdata)
data = matdata['all_data']
data = np.flip(np.rot90(np.reshape(data, [-1, 84, 96]), axes=(1, 2)), axis=2)
# test = np.squeeze(data[10])
for i in range(np.shape(data)[0]):
    plt.figure()
    plt.imshow(np.abs(np.squeeze(data[i])), cmap='gray')
    plt.show()



