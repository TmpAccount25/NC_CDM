import numpy as np
import matplotlib.pyplot as plt
import os

path = r'E:\dataset\cc-359\try\Train'
savepath = r'E:\dataset\cc-359\try_select\Train'
filelist = os.listdir(path)
for file in filelist:
    filepath = os.path.join(path, file)
    data = np.load(filepath)
    new_data = data[30:-30, :, :, :]
    # print('data.shape', data.shape)
    # check = new_data[-1, :, :, 0] + 1j * new_data[-1, :, :, 1]
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(np.abs(np.fft.ifft2(check)), cmap='gray')
    # plt.subplot(122)
    # plt.imshow(np.abs(check), cmap='gray')
    # plt.show()
    np.save(os.path.join(savepath, file), new_data)

