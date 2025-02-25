import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy
from torch.nn import functional as F
import myFFT2
import matplotlib.pyplot as plt
class   ConvBlock(nn.Module):
    def __init__(self, in_chans=2, out_chans=64):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, in_chans, kernel_size=3, padding=1),
        )

    def forward(self, input):
        return self.layers(input)

def data_consistency(recon_img, us_img, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    recon_img = myFFT2.real2complex(recon_img)
    us_img = myFFT2.real2complex(us_img)
    # k = myFFT2.image_fft2_k_torch(recon_img)
    # k0 = myFFT2.image_fft2_k_torch(us_img)
    k = torch.fft.ifftshift(torch.fft.fft2(recon_img))
    k0 = torch.fft.ifftshift(torch.fft.fft2(us_img))
    k = myFFT2.complex2real(k)
    k0 = myFFT2.complex2real(k0)

    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0

    out = myFFT2.real2complex(out)
    # out = myFFT2.k_ifft2_image_torch(out)
    out = torch.fft.ifft2(torch.fft.fftshift(out))
    out = myFFT2.complex2real(out)

    return out

class   KIKInet(nn.Module):
    """
    U_Net
    """
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        in_chans:输入通道数
        out_chans:输出通道数
        chans:中间层通道数
        num_pool_layers:上采样或下采样的层数
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.conv1 = ConvBlock(self.in_chans,self.out_chans)
        self.conv2 = ConvBlock(self.in_chans,self.out_chans)
        self.conv3 = ConvBlock(self.in_chans,self.out_chans)
        self.conv4 = ConvBlock(self.in_chans,self.out_chans)

    def forward(self, us_img, mask):
        # plt.subplot(121)
        # plt.imshow(torch.abs(mask[0,0,:,:]).cpu().detach().numpy(),'gray')
        # plt.subplot(122)
        # plt.imshow(torch.abs(us_img[0,:,:]).cpu().detach().numpy(),'gray')
        # plt.show()
        
        inputs_k_c = torch.fft.ifftshift(torch.fft.fft2(us_img))
        # inputs_k_c = torch.zeros(us_img.shape,dtype=torch.complex64).cuda()
        # for i in range(us_img.shape[0]):
        #     inputs_k_c[i,:,:] = myFFT2.image_fft2_k_torch(us_img[i,:,:])
        # plt.subplot(122)
        # plt.imshow(torch.abs(inputs_k_c[0,:,:]).cpu().detach().numpy(),'gray')
        # plt.show()
        us_img = myFFT2.complex2real(us_img)
        inputs_k_r = myFFT2.complex2real(inputs_k_c)
        temp = inputs_k_r
        for j in range(5):
            temp = self.conv1(temp)
        k_net_out = myFFT2.real2complex(temp)
        # temp1 = torch.zeros(k_net_out.shape,dtype=torch.complex64).cuda()
        # for i in range(us_img.shape[0]):
        #     temp1[i,:,:] = myFFT2.k_ifft2_image_torch(k_net_out[i,:,:])
        temp1 = torch.fft.ifft2(torch.fft.fftshift(k_net_out))
        # plt.subplot(122)
        # plt.imshow(torch.abs(us_img[0,:,:]).cpu().detach().numpy(),'gray')
        # plt.show()
        temp1 = myFFT2.complex2real(temp1)
        for j in range(5):
            res = self.conv2(temp1)
            temp1 = res + temp1
            temp1 = data_consistency(temp1, us_img, mask)
        

        temp1 = myFFT2.real2complex(temp1)
        temp2 = torch.fft.ifftshift(torch.fft.fft2(temp1))
        temp2 = myFFT2.complex2real(temp2)
        out1 = temp1

        for j in range(5):
            temp2 = self.conv3(temp2)
        k_net_out_1 = myFFT2.real2complex(temp2)
        temp3 = torch.fft.ifft2(torch.fft.fftshift(k_net_out_1))
        temp3 = myFFT2.complex2real(temp3)

        for j in range(5):
            res = self.conv4(temp3)
            temp3 = res + temp3
            temp3 = data_consistency(temp3, us_img, mask)
        temp3 = myFFT2.real2complex(temp3)
        out = temp3

        return out, out1, k_net_out, k_net_out_1