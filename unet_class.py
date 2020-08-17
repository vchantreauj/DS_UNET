"""unet class file and usefull functions"""

import numpy as np
from skimage import io
import torch
from torch import nn
import torch.nn.functional as F

class ImProcess():
    """class to process image previously to unet training"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.nb_im = 10
        self.crop_size = 44

    def load_set(self, repim, repmask, nb_im):
        """load images and get the train set
        width and height are required for the image to support the successives
        convolutions
        """
        self.nb_im = nb_im
        images = io.imread_collection(repim, plugin='tifffile')
        masks = io.imread_collection(repmask, plugin='tifffile')

        im_process = []
        labels = []
        for i in range(self.nb_im):
            im_process.append(images[i][:self.width, :self.height])
            labels.append(masks[i][self.crop_size:self.width - self.crop_size,
                                   self.crop_size:self.height - self.crop_size])

        im_process = np.array(im_process) / 255
        im_process = np.transpose(im_process, (0, 3, 1, 2))
        labels = np.array(labels) / 255
        return im_process, labels


class UNet(nn.Module):
    """encoder decoder method to analyse medical images"""

    def forward(self, x):
        """encoding block is 2 convolutions layers each followed by a max pooling layers
        with a stride of 2 for downsampling
        for each two layers conv/maxpool, Hout = Hin/2 - (k+1)
        ***
        decoding block is upsampling, concatenation and convolutions
        for each two layers crop and conv, Hout = 2Hin - 3k + 1"""
        # encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        # Decode
        decode_block3 = crop_and_concat(
            bottleneck1, encode_block3, crop=True)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = crop_and_concat(
            cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = crop_and_concat(
            cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return final_layer


    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        self.crop_size = 44
        self.conv_encode1 = contracting_block(
            in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = contracting_block(64, 128)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = contracting_block(128, 256)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.ConvTranspose2d(
                in_channels=512, out_channels=256,
                kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.conv_decode3 = expansive_block(512, 256, 128)
        self.conv_decode2 = expansive_block(256, 128, 64)
        self.final_layer = final_block(128, 64, out_channel)


def contracting_block(in_channels, out_channels, kernel_size=3):
    """convolution layers for unet contracting block"""
    block = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel_size,
                        in_channels=in_channels, out_channels=out_channels),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.Conv2d(kernel_size=kernel_size,
                        in_channels=out_channels, out_channels=out_channels),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(out_channels),
    )
    return block

def expansive_block(in_channels, mid_channel, out_channels, kernel_size=3):
    """convolution layers for expansive block"""
    block = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel_size,
                        in_channels=in_channels, out_channels=mid_channel),
        # conv2d Hout = Hin - k + 1
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(mid_channel),  # normalization over 4D input
        torch.nn.Conv2d(kernel_size=kernel_size,
                        in_channels=mid_channel, out_channels=mid_channel),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(mid_channel),
        torch.nn.ConvTranspose2d(
            in_channels=mid_channel,
            out_channels=out_channels,
            kernel_size=3, stride=2, padding=1, output_padding=1)
        # convtranspose2d Hout = 2Hin - 3 + k
    )
    return block

def final_block(in_channels, mid_channel, out_channels, kernel_size=3):
    """layers for output pixel labels"""
    block = torch.nn.Sequential(
        torch.nn.Conv2d(kernel_size=kernel_size,
                        in_channels=in_channels, out_channels=mid_channel),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(mid_channel),
        torch.nn.Conv2d(kernel_size=kernel_size,
                        in_channels=mid_channel, out_channels=mid_channel),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(mid_channel),
        torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel,
                        out_channels=out_channels, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(),  # add this to have binary mask as output torch.nn.Sigmoid()
    )
    return block

def crop_and_concat(upsampled, bypass, crop=False):
    """crop and concat input to prepare the expansion block"""
    if crop:
        crop_im_size = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (
            -crop_im_size,
            -crop_im_size,
            -crop_im_size,
            -crop_im_size))
    # concatenate on row (add element on line)
    return torch.cat((upsampled, bypass), 1)
