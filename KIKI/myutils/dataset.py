import os

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as itk
from torch.utils.data import Dataset
from skimage import transform


class SliceData(Dataset):
    def __init__(self, root, transform, batch_size):
        self.transformer = transform
        self.t2_data = []
        self.t1_data = []
        filesfloders = os.listdir(root)
        for filefloder in filesfloders:
            nii_file_t1, nii_file_t2= self.get_fname(root, filefloder)
            t2_file = itk.ReadImage(nii_file_t2)
            t2_data = itk.GetArrayFromImage(t2_file)
            num_slices = t2_data.shape[0]
            self.t2_data +=[(nii_file_t2, slice) for slice in range((num_slices // batch_size) * batch_size)]

            t1_file = itk.ReadImage(nii_file_t1)
            t1_data = itk.GetArrayFromImage(t1_file)
            num_slices = t1_data.shape[0]
            self.t1_data += [(nii_file_t1, slice) for slice in range(((num_slices // batch_size) * batch_size))]


    def  __len__(self):
        return len(self.t2_data)

    def get_fname(self, root, filefloder):
        filepath = os.path.join(root, filefloder)
        modalities = os.listdir(filepath)
        for file in modalities:
            fname_T2 = (filefloder + '_t2.nii')
            fname_T1 = (filefloder + '_t1.nii')
            if file == fname_T1:
                nii_file_t1 = os.path.join(root, filefloder, fname_T1)
                flag = 'T1'
            elif file == fname_T2:
                nii_file_t2 = os.path.join(root, filefloder, fname_T2)
                flag = 'T2'
            # else:
            #     nii_file_t1 = 'None'
            #     nii_file_t2 = 'None'
            #     flag = 'Not need'

        return nii_file_t1, nii_file_t2

    def __getitem__(self, i):
        fname_t2, slice_t2 = self.t2_data[i]
        filedata_t2 = itk.ReadImage(fname_t2)
        imgdata_t2 = itk.GetArrayFromImage(filedata_t2)
        imgdata_t2_resize = transform.resize(imgdata_t2, (imgdata_t2.shape[0], 256, 256))

        fname_t1, slice_t1 = self.t1_data[i]
        filedata_t1 = itk.ReadImage(fname_t1)
        imgdata_t1 = itk.GetArrayFromImage(filedata_t1)
        imgdata_t1_resize = transform.resize(imgdata_t1, (imgdata_t1.shape[0], 256, 256))

        # plt.figure()
        # plt.title(fname_t1)
        # plt.imshow(np.abs(imgdata_t1_resize[80, :, :]), cmap='gray')
        # plt.show()


        kdata_t1 = (np.fft.fft2(imgdata_t1_resize))
        kdata_t2 = (np.fft.fft2(imgdata_t2_resize))

        # imgdata_t1 = imgdata_t1[slice_t1]
        # imgdata_t2 = imgdata_t2[slice_t2]

        kspace_t2 = kdata_t2[slice_t2]
        kspace_t1 = kdata_t1[slice_t1]

        return self.transformer(kspace_t2, kspace_t1, slice_t2, slice_t1)



