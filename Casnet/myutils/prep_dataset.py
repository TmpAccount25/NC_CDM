import os
import shutil
import numpy as np
from skimage import transform
import SimpleITK as itk

path = r'E:\dataset\MICCAI_BraTS_2019_Data_Training\my_try\test'
savepath = r'E:\dataset\MICCAI_BraTS_2019_Data_Training\try\No_norm\test'

start_slice = 60
end_slice = 100
file_floders = os.listdir(path)

def normalize(x):
    shape = x.shape
    x_norm = np.zeros(shape)
    for i in range(x.shape[0]):
        slice = x[i]
        xmax = np.max(slice)
        xmin = np.min(slice)
        x_norm[i, :, :] =  (slice - xmin)/(xmax - xmin)
    return x_norm


def save_file(root, filefloder, fname, file):
    filepath = os.path.join(root, filefloder)
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    img = itk.GetImageFromArray(file)
    itk.WriteImage(img, os.path.join(filepath, fname))

for file_floder in file_floders:
    if not os.path.exists(os.path.join(savepath, file_floder)):
        os.mkdir(os.path.join(savepath, file_floder))
    filepath = os.path.join(path, file_floder)
    modalities = os.listdir(filepath)
    for modality in modalities:
        fname_T2 = (file_floder + '_t2.nii')
        fname_T1 = (file_floder + '_t1.nii')
        if modality == fname_T2:
            T2_file = itk.ReadImage(os.path.join(filepath, fname_T2))
            T2_data = itk.GetArrayFromImage(T2_file)
            T2 = T2_data[start_slice:end_slice, :, :]
            T2_resize = transform.resize(T2, (T2.shape[0], 256, 256),preserve_range=True)
            # T2_norm = normalize(T2_resize)
            save_file(savepath, file_floder, fname_T2, T2_resize)

        if modality == fname_T1:
            T1_file = itk.ReadImage(os.path.join(filepath, fname_T1))
            T1_data = itk.GetArrayFromImage(T1_file)
            T1 = T1_data[start_slice:end_slice, :, :]
            T1_resize = transform.resize(T1, (T1.shape[0], 256, 256), preserve_range=True)
            # T1_norm = normalize(T1_resize)
            save_file(savepath, file_floder, fname_T1, T1_resize)

