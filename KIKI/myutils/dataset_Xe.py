import os
import SimpleITK as itk
import numpy as np
from torch.utils.data import Dataset

class SliceData(Dataset):
    def __init__(self, root, transform, batch_size):
        self.transform = transform
        self.example = []
        files = os.listdir(root)
        for file in files:
            fname = os.path.join(root, file)
            # data = np.load(fname)
            imgdata = itk.ReadImage(fname)
            data = itk.GetArrayFromImage(imgdata)
            num_slices = data.shape[0]
            self.example += [(fname, slice) for slice in range((num_slices // batch_size) * batch_size)]

    def __len__(self):
        return len(self.example)

    def __getitem__(self, i):
        fname, slice = self.example[i]
        # data = np.load(fname)
        imgdata = itk.ReadImage(fname)
        data = itk.GetArrayFromImage(imgdata)
        # data = transform.resize(data, (data.shape[0], 256, 150))
        ks = np.fft.fft2(data)
        kspace = ks[slice]
        return self.transform(kspace, slice)


