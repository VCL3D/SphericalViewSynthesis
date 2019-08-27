import torch
from torch import nn
import torch.nn.functional as F

'''
    Code adapted from https://github.com/uber-research/coordconv 
    accompanying the paper "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution" (NeurIPS 2018)
'''

class AddCoords360(nn.Module):
    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        super(AddCoords360, self).__init__()
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]

        xx_ones = torch.ones([1, self.y_dim], dtype=torch.float32, device=input_tensor.device)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.float32, device=input_tensor.device).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)

        yy_ones = torch.ones([1, self.x_dim], dtype=torch.float32, device=input_tensor.device)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.float32, device=input_tensor.device).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        xx_channel = xx_channel.float() / (self.x_dim - 1)
        yy_channel = yy_channel.float() / (self.y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CoordConv360(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, with_r, in_channels, out_channels, kernel_size, *args, **kwargs):
        super(CoordConv360, self).__init__()
        self.addcoords = AddCoords360(x_dim=x_dim, y_dim=y_dim, with_r=with_r)                
        in_size = in_channels+2
        if with_r:
            in_size += 1            
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size, **kwargs)

    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret


def create_conv(in_size, out_size, conv_type, padding=1, stride=1, kernel_size=3, width=512):
    if conv_type == 'standard':
        return nn.Conv2d(in_channels=in_size, out_channels=out_size, \
            kernel_size=kernel_size, padding=padding, stride=stride)
    elif conv_type == 'coord':
        return CoordConv360(x_dim=width / 2.0, y_dim=width,\
            with_r=False, kernel_size=kernel_size, stride=stride,\
            in_channels=in_size, out_channels=out_size, padding=padding)    

def create_activation(activation):
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'elu':
        return nn.ELU(inplace=True)

class Identity(nn.Module):
    def forward(self, x):
        return x

def create_normalization(out_size, norm_type):
    if norm_type == 'batchnorm':
        return nn.BatchNorm2d(out_size)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(out_size // 4, out_size)
    elif norm_type == 'none':
        return Identity()

def create_downscale(out_size, down_mode):
    if down_mode == 'pool':
        return torch.nn.modules.MaxPool2d(2)
    elif down_mode == 'downconv':
        return nn.Conv2d(in_channels=out_size, out_channels=out_size, kernel_size=3,\
            stride=2, padding=1, bias=False)
    elif down_mode == 'gaussian':
        print("Not implemented")