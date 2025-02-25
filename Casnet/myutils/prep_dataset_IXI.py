import os
import SimpleITK as itk
import numpy as np

path = r'E:\dataset\IXI\T1'
savepath = r'E:\dataset\IXI\select\T1'


def normalize(x):
    shape = x.shape
    x_norm = np.zeros(shape)
    for i in range(x.shape[0]):
        slice = x[i]
        xmax = np.max(slice)
        xmin = np.min(slice)
        x_norm[i, :, :] = (slice - xmin)/(xmax - xmin)
    return x_norm

files = os.listdir(path)
for file in files:
    filepath = os.path.join(path, file)
    filedata = itk.ReadImage(filepath)
    data = itk.GetArrayFromImage(filedata)
    data = np.transpose(data, (1, 2, 0))
    data_select = data[120:200, :, :]
    norm_data = normalize(data_select)
    norm_img_data = itk.GetImageFromArray(norm_data)
    itk.WriteImage(norm_img_data, os.path.join(savepath,file))

