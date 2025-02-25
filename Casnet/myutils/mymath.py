import numpy as np
from numpy.fft import fft, fft2, ifft, ifft2, ifftshift, fftshift


sqrt = np.sqrt

def fftc(x, axis=-1, norm='ortho'):
    """x - [m,n]"""
    return fftshift(fft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)

def ifftc(x, axis=-1, norm='ortho'):
    return fftshift(ifft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)

def fft2c(x, norm):
    """
    centered fft
    默认对后两维使用 fft
    """
    axes = (-2, -1)
    res = fftshift(fft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res

def ifft2c(x, norm):
    axes = (-2, -1)
    res = fftshift(ifft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res

def fourier_matrix(rows, cols):
    """
    :param rows: 行
    :param cols: 列
    :return:rows * cols fourier matrix
    """
    col_range = np.arange(cols)
    row_range = np.arange(rows)
    scale = 1 / np.sqrt(cols)

    coeffs = np.outer(row_range, col_range)
    fouriermatrix = np.exp(coeffs * (-2. * np.pi *1j /cols)) *scale
    return fouriermatrix

def inverse_fourier_matrix(row, cols):
    return np.array(np.matrix(fourier_matrix(row, cols)).getH())

def flip(m, axis):
    """
    沿选定轴翻转
    m : array_like
        输入数组
    axis : int
        选定的轴
    """

    #hascatter: Return whether the object has an attribute with the given name.
    if not hasattr(m, 'ndim'):
        m = np.asarry(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]

def rot90_d(x, axes=(-2, -1), k=1):
    "旋转选定轴"
    def flipud(x):
        return flip(x, axes[0])

    def fliplr(x):
        return flip(x, axes[1])

    x = np.asanyarray(x)
    if x.ndim<2:
        raise ValueError("Input must >=2-d")
    k = k%4
    if k==0:
        return x
    elif k == 1:
        #return fliplr(x).swapaxes(*axes)
        return np.swapaxes(fliplr(x), *axes)
    elif k == 2:
        return fliplr(flipud(x))
    else:
        #return fliplr(x.swapaxes(*axes))
        return fliplr(np.swapaxes(x, *axes))