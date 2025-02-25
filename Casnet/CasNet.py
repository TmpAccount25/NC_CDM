import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy
from torch.nn import functional as F
import myFFT2


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans,drop_prob):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        return self.layers(input)

    def __repr___(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
                f'drop_prob={self.drop_prob})'



class   UNetModel(nn.Module):   # model = UNetModel(2, 2, 64, 4, 0).to(args.device)
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
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch*2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch*2, ch//2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch//2, kernel_size=1),
            nn.Conv2d(ch//2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1)
        )

    def forward(self, input):
        stack = []
        output = myFFT2.complex2real(input)

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            pool = nn.MaxPool2d(kernel_size=2)
            output = pool(output)

        output = self.conv(output)

        for layer in self.up_sample_layers:
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)
        return self.conv2(output)