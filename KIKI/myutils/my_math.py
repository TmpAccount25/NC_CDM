import torch
import numpy as np

def complex2real_torch(x):
    """
    复值[nx, ny]转为实值[2, nx, ny]
    """
    size = x.shape
    y = torch.zeros((2, size[0], size[1]), dtype=torch.float32)
    y[0, :, :] = x.real
    y[1, :, :] = x.imag
    return y

def real2complex_torch(x):
    """
    [batch_size, 2, nx, ny]转成复值[batch_size, nx, ny]
    """
    y = x[:, 0, :, :] + 1j * x[:, 1, :, :]
    return y

def complex2real_b_troch(x):
    """
    [batch_size, nx, ny]转成[batch_size, 2, nx, ny]
    """
    size = x.shape
    y = torch.zeros((size[0], 2, size[1], size[2]), dtype=torch.float32).to('cuda')
    y[:, 0, :, :] = x.real
    y[:, 1, :, :] = x.imag
    return y

def complex2real_numpy(x):
    size = x.shape
    y = np.zeros((size[0], 2, size[1], size[2]))
    y[:, 0, :, :] = x.real
    y[:, 1, :, :] = x.imag
    return y

def real2complex_numpy(x):
    y = x[:, 0, :, :] + 1j * x[:, 1, :, :]
    return y