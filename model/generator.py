import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils import spectral_norm


def sndeconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    )


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    )


def upsampleLayer(dim_in, dim_out, upsample='basic'):
    if upsample == 'basic':
        upconv = [sndeconv2d(dim_in, dim_out, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  snconv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError('upsample layer [%s] not implemented' % upsample)
    return upconv


def get_normalization(norm_type='instance'):
    """
    Return a normalization layer
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
        norm_layer = None
    else:
        raise NotImplementedError('Normalization type [%s] is not found' % norm_type)
    return norm_layer


def get_activation(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError('Non-linearity activation type [%s] is not found' % layer_type)

    return nl_layer


class GCN(nn.Module):
    """
    Graph Convolution Layer
    """
    def __init__(self, C_in, N):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(N, N, kernel_size=1)
        self.conv2 = nn.Conv1d(C_in, C_in, kernel_size=1)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.conv2(F.relu(h))
        return h


class GRM(nn.Module):
    def __init__(self, C_in, C_mid, ConvLayer, BatchNorm):
        super(GRM, self).__init__()

        self.C_mid = C_mid
        # projection phi
        self.conv_phi = ConvLayer(C_in, C_mid, kernel_size=1)
        # projection theta
        self.conv_theta = ConvLayer(C_in, C_mid, kernel_size=1)
        # gcn
        self.gcn = GCN(C_in=C_mid, N=C_mid)
        # raising dimension
        self.raising_ch = nn.Sequential(ConvLayer(C_mid, C_in, kernel_size=1, bias=False),
                                        BatchNorm(C_in))

    def forward(self, x):
        bs = x.size()[0]

        # x1: [bs, c_mid, h*w]
        x1 = self.conv_phi(x).view(bs, self.C_mid, -1)
        # x2: [bs, c_mid, h*w]
        x2 = self.conv_theta(x).view(bs, self.C_mid, -1)
        x2_copy = x2

        # M: (bs, c_mid, c_mid)
        M = x1 @ x2.permute(0, 2, 1).contiguous()
        M = M / M.size()[-1]
        # M_hat: [bs, c_mid, c_mid]
        M_hat = self.gcn(M)
        # reverse projection -> M_hat: [bs, c_mid, h*w]
        M_hat = M_hat @ x2_copy
        # x_out: [bs, c_mid, h, w]
        x_out = M_hat.view(bs, self.C_mid, x.size()[-2], x.size()[-1])

        # layer_out: [bs, c_in, h, w]
        layer_out = x + self.raising_ch(x_out)
        return layer_out


class UnetBlock(nn.Module):
    def __init__(self, dim_input, dim_outter, dim_inner, dim_z=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 upsample='basic'):
        super(UnetBlock, self).__init__()
        downconv = []
        downconv += [snconv2d(dim_input, dim_inner, kernel_size=4, stride=2, padding=1)]
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(dim_inner * 2, dim_outter, upsample=upsample)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            self.glore = GRM(C_in=dim_input, C_mid=dim_input // 4, ConvLayer=nn.Conv2d, BatchNorm=nn.InstanceNorm2d)
            upconv = upsampleLayer(dim_inner + dim_z + 3, dim_outter, upsample=upsample)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(dim_outter)]
        else:
            self.glore = GRM(C_in=dim_input, C_mid=dim_input//4, ConvLayer=nn.Conv2d, BatchNorm=nn.InstanceNorm2d)
            upconv = upsampleLayer(dim_inner * 2, dim_outter, upsample=upsample)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(dim_inner)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(dim_outter)]

            if use_dropout:
                up += [nn.Dropout(0.5)]

        self.outermost = outermost
        self.innermost = innermost
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        if self.outermost:
            x1 = self.down(x)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            # x: [bs, 512, 2, 2]
            x_and_z = torch.cat([self.down(x), z], 1)
            x1 = self.up(x_and_z)
            return torch.cat([x1, self.glore(x)], 1)
        else:
            x1 = self.down(x)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), self.glore(x)], 1)


class Unet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_z, norm_layer=None,
                 nl_layer=None, use_dropout=False, upsample='basic'):
        super(Unet, self).__init__()

        # construct U-net structure from inner to outer part
        # 1. bottleneck: 512 -> 320 -> 512, 2x2 -> 1x1 -> 2x2
        unet_block = UnetBlock(512, 512, 320, dim_z, None, innermost=True,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        # 2. 512 -> 512 -> 512, 4x4 -> 2x2 -> 4x4
        unet_block = UnetBlock(512, 512, 512, dim_z, unet_block, norm_layer=norm_layer,
                               nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)

        # 3. 512 -> 512 -> 512, 8x8 -> 4x4 -> 8x8
        unet_block = UnetBlock(512, 512, 512, dim_z, unet_block, norm_layer=norm_layer,
                               nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)

        # 4. 256 -> 512 -> 256, 8x8 -> 4x4 -> 8x8
        unet_block = UnetBlock(256, 256, 512, dim_z, unet_block, norm_layer=norm_layer,
                               nl_layer=nl_layer, upsample=upsample)

        unet_block = UnetBlock(128, 128, 256, dim_z, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        unet_block = UnetBlock(64, 64, 128, dim_z, unet_block, norm_layer=norm_layer,
                               nl_layer=nl_layer, upsample=upsample)

        unet_block = UnetBlock(dim_in, dim_out, 64, dim_z, unet_block, outermost=True,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)


def init_net(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    # define the initialization function
    def init_func(m):
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
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

    return net


def define_G(dim_in, dim_out, dim_z, norm='instance', nl='relu', use_dropout=False,
             init_type='normal', init_gain=0.02, upsample='basic'):
    norm_layer = get_normalization(norm_type=norm)
    nl_layer = get_activation(layer_type=nl)
    net = Unet(dim_in, dim_out, dim_z, norm_layer, nl_layer, use_dropout, upsample)

    return init_net(net, init_type, init_gain)


class Generator(nn.Module):
    def __init__(self, dim_in=1, dim_z=100, norm_type='instance', nl='lrelu', dropout=False,
                 init_type="normal", init_gain=0.02, upsample='basic'):
        super(Generator, self).__init__()
        self.netG = define_G(dim_in, dim_in, dim_z, norm_type, nl, dropout, init_type, init_gain, upsample)

    def forward(self, x, z0):
        return self.netG(x, z0)
