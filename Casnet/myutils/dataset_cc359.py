import pathlib
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class SliceData(Dataset):
    def __init__(self, root, transform, batch_size):
        self.transform = transform
        self.example = []
        files = list(pathlib.Path(root).iterdir())
        for fname in sorted(files):
            kspace = np.load(fname)
            num_slices = kspace.shape[0]
            self.example += [(fname, slice) for slice in range( (num_slices // batch_size) * batch_size)]

    def __len__(self):
        return len(self.example)

    def __getitem__(self, i):
        fname, slice = self.example[i]
        data = np.load(fname)
        kspace = data[slice]
        return self.transform(kspace, fname.name, slice)

def to_complex(data):
    data = data[:, :, 0] + 1j * data[:, :, 1]
    return data

