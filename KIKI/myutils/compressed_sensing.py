import numpy as np
import random
from utils import mymath
from numpy.lib.stride_tricks import  as_strided

def soft_thresh(u, lmda):
    """soft-threshing operator for complex value input"""
    Su = (abs(u) - lmda) / abs(u) * u
    Su[abs(u) < lmda ] = 0
    return Su

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

def var_dens_mask(shape, ivar, sample_high_freq=True):
    """Variable Density Mask (2D undersampling),变密度采样掩码"""
    if len(shape) == 3:
        Nt, Nx, Ny = shape
    else:
        Nx, Ny = shape
        Nt = 1

    pdf_x = normal_pdf(Nx, ivar)
    pdf_y = normal_pdf(Ny, ivar)
    pdf = np.outer(pdf_x, pdf_y)

    size = pdf.itemsize #生成元素所占空间大小
    strided_pdf = as_strided(pdf, (Nt, Nx, Ny),(0, Ny*size, size))
    if sample_high_freq:
        strided_pdf = strided_pdf / 1.25 + 0.02
    mask = np.random.binomial(1, strided_pdf)

    xc = int(Nx / 2)
    yc = int(Ny / 2)
    mask[:, xc - 10:xc+11, yc -10:yc + 11] =True

    if Nt == 1:
        return mask.reshape((Nx, Ny))

    return mask

def cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - 维度(..., nx, ny)
    acc: float - 加速倍数
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)

    pdf_x += lmda * 1./Nx

    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = mymath.ifftshift(mask, axes=(-1, -2))

    return mask

def shear_grid_mask(shape, acceleration_rate, sample_low_freq=True,
                    centred=False, sample_n=10):
    """
    shape: (nt, nx, ny)
    acceleration_rate: int
    """
    Nt, Nx, Ny = shape
    start = np.random.randint(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in range(Nt):
        mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

    xc = int(Nx / 2)
    xl = int(sample_n / 2)
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0 :
            xh += 1
            mask[:, xc - xc:xc + xh+1] =1

    elif sample_low_freq:
        xh =xl
        if sample_n % 2 == 1:
            xh -= 1

        if xl > 0:
            mask[:, :xl] = 1
        if xh > 0:
            mask[:, -xh:] = 1

    mask_rep = np.repeat(mask[..., np.newaxis], Ny, axis=-1)
    return mask_rep

def perturbed_shear_grid_mask(shape, acceleration_rate, sample_low_freq=True,
                              centred=False,
                              sample_n=10):
    Nt, Nx, Ny = shape
    start = np.random.randint(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in range(Nt):
        mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

    rand_code = np.random.randint(0 , 3, size=Nt*Nx)
    shift = np.array([-1, 0, 1])[rand_code]
    new_mask = np.zeros_like(mask)
    for t in range(Nt):
        for x in range(Nx):
            if mask[t, x]:
                new_mask[t, (x + shift[t*x])%Nx] = 1

    xc = int(Nx / 2)
    xl = int(sample_n / 2)
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1
        new_mask[:, xc - xl:xc + xh+1] = 1
    elif sample_low_freq:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1

        new_mask[:, :xl] = 1
        new_mask[:, -xh:] = 1
    mask_rep = np.repeat(new_mask[..., np.newaxis], Ny, axis=-1)

    return mask_rep

def undersample(x, mask, centred=False, norm='ortho', noise=0):
    """
    x: array_like
        data
    mask: array_like
        undersampling mask in fourier domain
    norm: 'ortho' or None
        if 'ortho', performs unitary transform, otherwise normal dft
    noise:float
        模拟获得噪声，复数AWG噪声，为峰值信号的百分比
    返回:
    xu； array_like
        图像域欠采样图像，为复值
    x_fu；array_like
        k-sapce欠采样数据
    """
    assert x.shape ==mask.shape
    noise_power = noise
    nz = np.sqrt(.5)*(np.random.normal(0, 1, x.shape) + 1j * np.random.normal(0, 1, x.shape))
    nz = nz * np.sqrt(noise_power)

    if norm == 'ortho':
        nz = nz * np.sqrt(np.prod(mask.shape[-2:]))
    else:
        nz = nz * np.prod(mask.shape[-2:])

    if centred:
        x_f = mymath.fft2c(x, norm=norm)
        x_fu = mask * (x_f + nz)
        x_u = mymath.ifft2c(x_fu, norm=norm)
        return x_u,x_fu
    else:
        x_f = mymath.fft2c(x, norm=norm)
        x_fu = mask * (x_f + nz)
        x_u = mymath.ifft2c(x_fu, norm=norm)
        return x_u, x_fu

def data_consistency(x, y, mask, centered=False, norm='ortho'):
    """
    x - 图像域
    y- k-space
    """
    if centered:
        xf = mymath.fft2c(x, norm=norm)
        xm = (1 - mask) * xf + y
        xd = mymath.ifft2c(xm, norm=norm)
    else:
        xf = mymath.fft2(x, norm=norm)
        xm = (1-mask) * xf + y
        xd = mymath.ifft2(xm, norm=norm)
    return xd

def get_phase(x):
    xr = np.real(x)
    xi = np.imag(x)
    phase = np.arctan(xi / (xr + 1e-12))
    return phase

def undersampling_rate(mask):
    return float(mask.sum()) / mask.size


def random_sampling(x, acc=4):
    shape = x.shape
    mask = np.zeros(shape)
    sample_num = int(shape[-1]/acc)
    for i in range(shape[0]):
        list = random.sample(range(0, shape[-1]), sample_num)
        for j in range(sample_num):
            mask[i, list[j], :] = 1
    return mask

def divided_mask(mask):
    """
    为256×256图片，加速为4，sample_n=10设计
    """
    centerlist = list(range(124, 133))
    shape = mask.shape
    mask1 = np.zeros(shape)
    mask2 = np.zeros(shape)
    samplepos = np.zeros((shape[0], int(shape[1]/2) - 9))
    #获得除中心9行外的采样位置
    for i in range(shape[0]):
        m = 0
        for j in range(shape[1]):
            if j not in centerlist:
                if mask[i, j, 0] == 1:
                    samplepos[i, m] = j
                    m = m + 1
    length = samplepos.shape[1]
    for i in range(shape[0]):
        check = []
        # sample_num = np.random.randint(low=0, high=length, size=int(length / 2))
        sample_num = random.sample(range(0, length), int(length/2))
        for j in range(length):
            if j in sample_num:
                pahse = int(samplepos[i, j])
                mask1[i, pahse, :] = 1
            else:
                pahse = int(samplepos[i, j])
                mask2[i, pahse, :] = 1


    # sum1 = np.sum(mask1[1, :, 1])
    # sum2 = np.sum(mask2[1, :, 1])
    #中心校正
    Ny = shape[1]
    mask1[:, int(Ny/2)-4:int(Ny/2)+5, :] = 1
    mask2[:, int(Ny/2)-4:int(Ny/2)+5, :] = 1
    return mask1, mask2





