import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
生成的结果请自行归一化
函数分为torch版本和numpy版本
输入输出均为2D数据: [h,w]
'''
def image_fft2_k_torch(input):

    # 输入数据通道数为二维,输出通道为二维

    # input=input/torch.max(torch.abs(input))
    img_shift=torch.fft.ifftshift(input)   # 移位使其符合变换规则
    k=torch.fft.fft2(img_shift)  # IFFT变换获得图像
    k_recon=torch.fft.ifftshift(k)  #　k_recon
    k_recon= (1/len(k_recon[:]))*k_recon

    # img_complex = real2complex(input)
    # k_com = torch.fft.fft2(img_complex)
    # k = complex2real(img_complex) 

    return k_recon

def k_ifft2_image_torch(input):

    k_shift=torch.fft.fftshift(input)   # 移位使其符合变换规则
    image=torch.fft.ifft2(k_shift)  # IFFT变换获得图像
    # img =image
    img=torch.fft.ifftshift(image)  # 再移位回来
    img= (len(img[:]))*img

    # k_complex = real2complex(input)
    # img_com = torch.fft.ifft2(k_complex)
    # img = complex2real(img_com) 
    return img

def image_fft2_k_np(input):
    # input=input/np.max(np.abs(input))
    img_shift=np.fft.ifftshift(input)   # 移位使其符合变换规则
    k=np.fft.fft2(img_shift)  # IFFT变换获得图像
    k_recon=np.fft.fftshift(k)  #　k_recon
    k_recon= (1/len(k_recon[:]))*k_recon
    return k_recon

def k_ifft2_image_np(input):
    k_shift=np.fft.fftshift(input)   # 移位使其符合变换规则
    image=np.fft.ifft2(k_shift)  # IFFT变换获得图像
    img=np.fft.ifftshift(image)  # 再移位回来
    img= (len(img[:]))*img
    return img

def complex2real(input_data):
    """
    Parameters
    ----------
    input_data : row x col
    dtype :The default is np.float32.

    Returns
    -------
    output : row x col x 2

    """
    if input_data.shape[1] == 1:
        input_data=input_data
    else:
        input_data=input_data.unsqueeze(1)

    input_data_Real=torch.real(input_data)
    input_data_Imag=torch.imag(input_data)

    input=torch.cat([input_data_Real,input_data_Imag],dim=1).float()

    return input


def real2complex(input_data):
    """
    Parameters
    ----------
    input_data : row x col x 2

    Returns
    -------
    output : row x col

    """
    if len(input_data.shape) == 4:
        output = input_data[:,0,:,:] + 1j * input_data[:,1,:,:]
    elif len(input_data.shape) == 5:
        output = input_data[:,0,:,:,:] + 1j * input_data[:,1,:,:,:]

    return output


def Image2Kspace_bchw(image_data):
    # input_shape:[batchsize, c, h, w]

    input_shape = image_data.shape
    kspace_tensor = torch.empty(input_shape, dtype=torch.complex64).to(device)
    for ii in range(np.shape(image_data)[0]):
        kspace_tensor[ii,0,:,:] = image_fft2_k_torch(image_data[ii,0,:,:])

    # output_shape: [batchsize, c, h, w]
    return kspace_tensor

def Kspace2Image_bchw(kspace_data):
    # input_shape:[batchsize, c, h, w]

    input_shape = kspace_data.shape
    image_tensor = torch.empty(input_shape, dtype=torch.complex64).to(device)
    for ii in range(np.shape(kspace_data)[0]):
        image_tensor[ii,0,:,:] = k_ifft2_image_torch(kspace_data[ii,0,:,:])

    # output_shape: [batchsize, c, h, w]
    return image_tensor
