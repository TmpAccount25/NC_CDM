__author__ = 'Jo Schlemper'

import numpy as np


def mse(x, y):
    return np.mean(np.abs(x - y)**2)


def psnr(x, y):
    '''
    Measures the PSNR of recon w.r.t x.
    Image must be of either integer (0, 256) or float value (0,1)
    :param x: [m,n]
    :param y: [m,n]
    :return:
    '''
    assert x.shape == y.shape
    assert x.dtype == y.dtype or np.issubdtype(x.dtype, np.float) \
        and np.issubdtype(y.dtype, np.float)
    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1

    mse = np.sum((x - y) ** 2).astype(float) / x.size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)


def complex_psnr(x, y, peak='normalized'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max

    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).

    '''
    mse = np.mean(np.abs(x - y)**2)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/mse)
    else:
        return 10*np.log10(1./mse)


def normalized(x):
    shape = x.shape
    if len(shape) == 3:
        nx = np.zeros(shape)
        for i in range(shape[0]):
            nx[i, :, :] = (x[i] - np.min(x[i]))/(np.max(x[i]) - np.min(x[i]))
    else:
        nx = (x - np.min(x)) / (np.max(x) - np.min(x))
    return nx

def mse(x, y):
    return np.means(np.abs(x - y) ** 2)

def psnr(x, y):
    assert x.shape == y.shape
    assert x.dtype == y.dtype or np.issubdtype(x.dtype, np.float32) \
           and np.issubdtype(y.dtype, np.float32)
    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1

    mse = np.sum((x - y) ** 2).astype(float) / x.size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)

def psnr_mask(x, y, mask):
    assert x.shape == y.shape
    assert x.dtype == y.dtype or np.issubdtype(x.dtype, np.float32) \
           and np.issubdtype(y.dtype, np.float32)
    if x.dtype == np.uint8:
        max_intensity = 256
    else:
        max_intensity = 1

    size = np.sum(mask)

    mse = np.sum((x - y) ** 2).astype(float) / size
    return 20 * np.log10(max_intensity) - 10 * np.log10(mse)