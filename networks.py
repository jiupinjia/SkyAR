import torch
import torch.nn as nn
from torch.nn import init
import functools
from torchvision import models
from torch.optim import lr_scheduler
import math
import utils
import matplotlib.pyplot as plt
import numpy as np

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PI = math.pi

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net




def define_G(input_nc, output_nc, ngf, netG, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type='none')
    if netG == 'resnet50':
        net = ResNet50FCN()
    elif netG == 'coord_resnet50':
        net = ResNet50FCN(coordconv=True)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)



class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, y_dim, x_dim = input_tensor.size()

        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2).type_as(input_tensor)
        yy_channel = yy_channel.float() / y_dim
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        ret = torch.cat([input_tensor, yy_channel], dim=1)

        return ret

class CoordConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        in_size = in_channels + 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = AddCoords()(x)
        ret = self.conv(ret)
        return ret


class ResNet50FCN(torch.nn.Module):
    def __init__(self, coordconv=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet50FCN, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2)
        self.coordconv = coordconv

        if coordconv:
            self.conv_in = CoordConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv_fpn1 = CoordConv2d(2048, 1024, kernel_size=3, padding=1)
            self.conv_fpn2 = CoordConv2d(1024, 512, kernel_size=3, padding=1)
            self.conv_fpn3 = CoordConv2d(512, 256, kernel_size=3, padding=1)
            self.conv_fpn4 = CoordConv2d(256, 64, kernel_size=3, padding=1)
            self.conv_pred_1 = CoordConv2d(64, 64, kernel_size=3, padding=1)
            self.conv_pred_2 = CoordConv2d(64, 1, kernel_size=3, padding=1)
        else:
            self.conv_fpn1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
            self.conv_fpn2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
            self.conv_fpn3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.conv_fpn4 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
            self.conv_pred_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv_pred_2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # resnet layers
        if self.coordconv:
            x = self.conv_in(x)
        else:
            x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4) # 1/8, in=64, out=128
        x_16 = self.resnet.layer3(x_8) # 1/16, in=128, out=256
        x_32 = self.resnet.layer4(x_16) # 1/32, in=256, out=512

        # FPN layers
        x = self.upsample(self.relu(self.conv_fpn1(x_32)))
        x = self.upsample(self.relu(self.conv_fpn2(x + x_16)))
        x = self.upsample(self.relu(self.conv_fpn3(x + x_8)))
        x = self.upsample(self.relu(self.conv_fpn4(x + x_4)))

        # output layers
        x = self.upsample(self.relu(self.conv_pred_1(x)))
        x = self.sigmoid(self.conv_pred_2(x))

        return x

