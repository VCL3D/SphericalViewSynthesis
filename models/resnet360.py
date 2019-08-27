import torch
import torch.nn as nn

import functools

from .modules import *

# adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py 

class ResNet360(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        depth=5,
        wf=32,
        conv_type='coord',
        padding='kernel',
        norm_type='none',
        activation='elu',
        up_mode='upconv',
        down_mode='downconv',
        width=512,
        use_dropout=False,
        padding_type='reflect',
    ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(depth >= 0)
        super(ResNet360, self).__init__()
        model = (
            [
                create_conv(in_channels, wf, conv_type, \
                    kernel_size=7, padding=3, stride=1, width=width),
                create_normalization(wf, norm_type),
                create_activation(activation)
            ]
        )

        n_downsampling = 2
        for i in range(n_downsampling): 
            mult = 2 ** i
            model += (
                [
                    create_conv(wf * mult, wf * mult * 2, conv_type, \
                        kernel_size=3, stride=2, padding=1, width=width // (i+1)),
                    create_normalization(wf * mult * 2, norm_type),
                    create_activation(activation)
                ]
            )

        mult = 2 ** n_downsampling
        for i in range(depth):
            model += [ResnetBlock(wf * mult, activation=activation, \
                norm_type=norm_type, conv_type=conv_type, \
                width=width // (2 ** n_downsampling))]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += (
                [
                    nn.ConvTranspose2d(wf * mult, int(wf * mult / 2),
                        kernel_size=3, stride=2,
                        padding=1, output_padding=1),
                    create_normalization(int(wf * mult / 2), norm_type),
                    create_activation(activation)
                ]
            )
        
        model += [create_conv(wf, out_channels, conv_type, \
            kernel_size=7, padding=3, width=width)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, norm_type, conv_type, activation, width):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block +=(
            [
                create_conv(dim, dim, conv_type, width=width),
                create_normalization(dim, norm_type),
                create_activation(activation),
            ]
        )
        conv_block +=(
            [
                create_conv(dim, dim, conv_type, width=width),
                create_normalization(dim, norm_type),
            ]
        )

        self.block = nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.block(x)  # add skip connections
        return out