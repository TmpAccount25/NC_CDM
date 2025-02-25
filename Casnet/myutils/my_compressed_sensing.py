import random
import torch
import numpy as np
from numpy.lib.stride_tricks import  as_strided

# def create_mask(shape, acc, sample_n=10):
#     """
#     输入shape应为[bs, nx, ny]
#     """
#     nx = shape[-2]
#     ny = shape[-1]
#     sample_line = int(nx / acc)
#     for i in range(shape[0]):


def get_mask_subset(mask, acc):
    shape = mask.shape
    sample_num = int(shape[1]*shape[2] / acc)
    mask1 = np.zeros(shape)
    mask2 = np.zeros(shape)
    for i in range(shape[0]):
        coord = random_coordinates(mask[i], sample_num)
        for j in range(len(coord)):
            x, y = coord[j]
            mask1[i, x, y] = 1
        mask2[i] = mask[i] - mask1[i]
    return mask1, mask2


def random_coordinates(mask, num, range_x=(0, 255), range_y=(0, 255)):
    coordiante = []
    while len(coordiante) < num:
        x = random.randint(range_x[0], range_x[1])
        y = random.randint(range_y[0], range_y[1])
        new_coor = (x, y)
        if mask[x, y] == 1:
            if new_coor not in coordiante:
                coordiante.append(new_coor)
    return coordiante


import numpy as np


# def split_matrix(matrix):
#     # 获取矩阵中元素为1的坐标
#     ones_indices = np.argwhere(matrix == 1)
#
#     # 随机打乱元素为1的坐标顺序
#     np.random.shuffle(ones_indices)
#
#     # 计算每个矩阵中元素为1的个数
#     total_ones = len(ones_indices)
#     ones_per_matrix = total_ones // 2
#
#     # 从打乱的坐标中取出相应个数的坐标
#     matrix1_indices = ones_indices[:ones_per_matrix]
#     matrix2_indices = ones_indices[ones_per_matrix:]
#
#     # 创建两个新矩阵，并根据坐标将元素置为1
#     matrix1 = np.zeros_like(matrix)
#     matrix2 = np.zeros_like(matrix)
#
#     for index in matrix1_indices:
#         matrix1[index[0], index[1]] = 1
#
#     for index in matrix2_indices:
#         matrix2[index[0], index[1]] = 1
#
#     return matrix1, matrix2

def random_split_matrix(matrix):
    shape = matrix.shape
    if len(shape) == 3:
        bs, nx, ny = shape
        matrix1 = np.zeros(shape, dtype=int)
        matrix2 = np.zeros(shape, dtype=int)

        for slice in range(shape[0]):
            for i in range(shape[1]):
                for j in range(shape[2]):
                    if matrix[slice, i, j] == 1:
                        # 随机决定将元素1分配给哪个矩阵
                        if np.random.choice([True, False]):
                            matrix1[slice, i, j] = 1
                        else:
                            matrix2[slice, i, j] = 1

        matrix1[:, int(nx/2-5):int(nx/2+6), int(ny/2-5):int(ny/2+6)] = 1
        matrix2[:, int(nx / 2 - 5):int(nx / 2 + 6), int(ny / 2 - 5):int(ny / 2 + 6)] = 1
    else:
        nx, ny = shape
        # matrix1 = np.zeros(shape, dtype=int)
        # matrix2 = np.zeros(shape, dtype=int)
        #
        #
        # for i in range(shape[0]):
        #     for j in range(shape[1]):
        #         if matrix[i, j] == 1:
        #             # 随机决定将元素1分配给哪个矩阵
        #             if np.random.choice([True, False]):
        #                 matrix1[i, j] = 1
        #             else:
        #                 matrix2[i, j] = 1

        shape = matrix.shape
        p = np.random.uniform(0, 1, shape)
        maskp = np.where(p > 0.5, 1, 0)

        matrix1 = maskp * matrix
        matrix2 = matrix - matrix1

        matrix1[int(nx / 2 - 5):int(nx / 2 + 6), int(ny / 2 - 5):int(ny / 2 + 6)] = 1
        matrix2[int(nx / 2 - 5):int(nx / 2 + 6), int(ny / 2 - 5):int(ny / 2 + 6)] = 1

    return matrix1, matrix2


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


def spilt_motherfucker(mask):
    shape = mask.shape
    p = np.random.uniform(0, 1, shape)
    maskp = torch.from_numpy(np.where(p > 0.5, 1, 0)).to('cuda')
    spilt_mask = maskp * mask
    return spilt_mask

