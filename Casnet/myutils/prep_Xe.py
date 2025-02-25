import SimpleITK as itk
from scipy.io import loadmat
import numpy as np
import os
# path = r'E:\dataset\data_xe\lungmask'
# savepath = r'E:\dataset\data_xe\select'
# files = os.listdir(path)
# for file in files:
#     filepath = os.path.join(path, file)
#     imgs = os.listdir(filepath)
#     for img in imgs:
#         if img[-6:-4] == 'Xe':
#             imgpath = os.path.join(filepath, img)
#             # print(imgpath)
#             imgdata = itk.ReadImage(imgpath)
#             data = itk.GetArrayFromImage(imgdata)
#             data_select = data[4:19]
#             imgdata_select = itk.GetImageFromArray(data_select)
#             itk.WriteImage(imgdata_select, os.path.join(savepath, img))

def normalized(x):
    """
    输入复值
    """
    x = np.abs(x)
    shape = x.shape
    n_x = np.zeros(shape)
    for i in range(shape[0]):
        xmax = np.max(x[i])
        xmin = np.min(x[i])
        norm_img = (x[i] - xmin) / (xmax - xmin)
        print('i=', i)
        n_x[i] = np.fft.ifft2(np.fft.fft2(norm_img))
    return n_x



path = r'E:\dataset\data_xe\select\data'
savepath = r'E:\dataset\data_xe\select\hsnr_data_nii'
files = os.listdir(path)
for file in files:
    data = loadmat(os.path.join(path, file))['k_xe']
    shape = data.shape
    # data = loadmat(os.path.join(path, file))
    if len(shape) == 3:
        data = np.transpose(data, (2, 0, 1))
        shape = data.shape
        print('data.shape', data.shape)
        if shape[0] != 0 :
            slice_data = np.fft.fft2(data)
            norm_data = normalized(slice_data)
            # img_slicedata = itk.GetImageFromArray(slice_data)
            file_py = file.replace('.mat', '.nii')
            norm_img_data = itk.GetImageFromArray(norm_data)
            itk.WriteImage(norm_img_data, os.path.join(savepath, file_py))
            # np.save(os.path.join(savepath, file_py), slice_data)
