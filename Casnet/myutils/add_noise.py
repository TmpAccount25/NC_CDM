import numpy as np
import SimpleITK as itk
import torch
import os

def add_noise(input, sigm):
    shape = input.shape
    noise = (np.random.normal(0, sigm, shape).astype(np.float32))
    # noise = np.random.normal(0, sigm, shape).astype(np.float32)
    x = input + noise
    return x, noise


path = r'E:\dataset\data_xe\select\test'
savepath = r'E:\dataset\data_xe\select\noise_test\sigma=0.8'

files = os.listdir(path)
for file in files:
    data = itk.ReadImage(os.path.join(path, file))
    img = itk.GetArrayFromImage(data)
    k = np.fft.fft2(img)
    k_noise, noise = add_noise(k, 0.8)
    img_noise = np.abs(np.fft.ifft2(k_noise))
    data_noise = itk.GetImageFromArray(img_noise)
    itk.WriteImage(data_noise, os.path.join(savepath, file))


