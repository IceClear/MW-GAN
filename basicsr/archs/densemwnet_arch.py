import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from .densemwnet_util import initialize_weights, BBlock
from .wavelet_util import DWT, IWT, default_conv
import torch.nn.init as init
from torch.nn.parameter import Parameter
from .pac_util import PacConv2d
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn.parameter import Parameter
import collections.abc
from itertools import repeat
from torch.nn.init import constant_
import math
from basicsr.utils import get_root_logger
from copy import deepcopy
from .arch_util import flow_warp, make_layer
# try:
#     from models.modules.dcn.deform_conv import ModulatedDeformConvPack as DCN
# except ImportError:
#     raise ImportError('Failed to import DCNv2 module.')
# from pytorch_wavelets import SWTForward, DWTForward # (or import DWT, IDWT)

class DWT_CNN(nn.Module):
    def __init__(self, ch):
        super(DWT_CNN, self).__init__()
        self.DWT= DWT()
        self.Conv = nn.Conv2d(ch*4, ch, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return(self.lrelu(self.Conv(self.DWT(x))))


class IWT_CNN(nn.Module):
    def __init__(self, ch):
        super(IWT_CNN, self).__init__()
        self.IWT = IWT()
        self.Conv = nn.Conv2d(ch//4, ch, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return(self.lrelu(self.Conv(self.IWT(x))))

class Motion_fea_5c(nn.Module):
    def __init__(self, nf=2, out=2, gc=32, bias=True, use_snorm=False):
        super(Motion_fea_5c, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        if use_snorm:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nf, gc, 3, 1, 1, bias=bias))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(gc, gc, 3, 1, 1, bias=bias))
            self.conv3 = nn.utils.spectral_norm(nn.Conv2d(gc, gc, 3, 1, 1, bias=bias))
            self.conv4 = nn.utils.spectral_norm(nn.Conv2d(gc, gc, 3, 1, 1, bias=bias))
            self.conv5 = nn.utils.spectral_norm(nn.Conv2d(gc, out, 3, 1, 1, bias=bias))
        else:
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
            self.conv2 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
            self.conv3 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
            self.conv4 = nn.Conv2d(gc, gc, 3, 1, 1, bias=bias)
            self.conv5 = nn.Conv2d(gc, out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        x4 = self.lrelu(self.conv4(x3))
        x5 = self.lrelu(self.conv5(x4))
        return x5

class DWT_motion_Pyramid(nn.Module):
    def __init__(self, gc=64, bias=True, use_snorm=False):
        super(DWT_motion_Pyramid, self).__init__()

        self.DWT= DWT()
        self.IWT = IWT()

        self.warp_level_1 = Motion_fea_5c(nf=6+3, out=6, gc=gc, bias=bias, use_snorm=use_snorm)
        self.warp_level_2 = Motion_fea_5c(nf=6*4*2+3, out=6*4 ,gc=gc, bias=bias, use_snorm=use_snorm)
        self.warp_level_3 = Motion_fea_5c(nf=6*4*4, out=6*4*4, gc=gc, bias=bias, use_snorm=use_snorm)

        self.conv_1 = nn.Conv2d(6, 2, 1, 1, 0, bias=bias)
        self.conv_2 = nn.Conv2d(6, 2, 1, 1, 0, bias=bias)
        self.conv_3_1 = nn.Conv2d(6*4, 6*4, 1, 1, 0, bias=bias)
        self.conv_3_2 = nn.Conv2d(6, 2, 1, 1, 0, bias=bias)
        self.conv_3_3 = nn.Conv2d(3, 3, 3, 2, 1, bias=bias)
        self.conv_last = nn.Conv2d(3, 3, 1, 1, 0, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_to_warp = x[:,3:,:,:]
        x_2 = self.DWT(x)
        x_3 = self.DWT(x_2)

        fea_3 = self.warp_level_3(x_3)
        fea_3 = self.tanh(self.conv_3_1(self.IWT(fea_3)))
        fea_3_to_warp = self.conv_3_2(self.IWT(fea_3))

        warp_3 = flow_warp(x_to_warp,fea_3_to_warp)
        warp_3_downsample = self.lrelu(self.conv_3_3(warp_3))
        fea_2 = torch.cat([x_2, fea_3, warp_3_downsample], dim=1)
        fea_2 = self.warp_level_2(fea_2)
        fea_2 = self.tanh(self.conv_2(self.IWT(fea_2)))
        warp_2 = flow_warp(x_to_warp,fea_2)
        fea_1 = torch.cat([x, warp_2], dim=1)
        fea_1 = self.warp_level_1(fea_1)
        fea_1 = self.tanh(self.conv_1(fea_1))
        warp_1 = flow_warp(x_to_warp,fea_1)

        return self.conv_last(warp_1)

class MWP_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(MWP_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_pcnpack = PacConv2d(nf, nf, 3, stride=1, padding=1, dilation=1)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_pcnpack = PacConv2d(nf, nf, 3, stride=1, padding=1, dilation=1)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_pcnpack = PacConv2d(nf, nf, 3, stride=1, padding=1, dilation=1)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_pcnpack = PacConv2d(nf, nf, 3, stride=1, padding=1, dilation=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_pcnpack(nbr_fea_l[2], L3_offset))
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_pcnpack(nbr_fea_l[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_pcnpack(nbr_fea_l[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_pcnpack(L1_fea, offset))

        return L1_fea


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True, use_snorm=False):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        if use_snorm:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nf, gc, 3, 1, 1, bias=bias))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias))
            self.conv3 = nn.utils.spectral_norm(nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias))
            self.conv4 = nn.utils.spectral_norm(nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias))
            self.conv5 = nn.utils.spectral_norm(nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias))
        else:
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class WDRB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, bias=False, use_snorm=False):
        super(WDRB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc, bias, use_snorm=use_snorm)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc, bias, use_snorm=use_snorm)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc, bias, use_snorm=use_snorm)
        self.dwt = DWT()
        self.iwt = IWT()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input):
        x = input
        x_dwt1 = self.dwt(x)
        out = self.RDB1(x_dwt1)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = self.iwt(out)
        out = out * 0.2 + x
        return out

class WDRB_Mini(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, use_snorm=False):
        super(WDRB_Mini, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc, use_snorm=use_snorm)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc, use_snorm=use_snorm)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc, use_snorm=use_snorm)
        self.dwt = DWT_CNN(nf)
        self.iwt = IWT_CNN(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input):
        x = input
        x_dwt1 = self.dwt(x)
        out = self.RDB1(x_dwt1)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = self.iwt(out)
        out = out * 0.2 + x
        return out

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, use_snorm=False):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc, use_snorm)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc, use_snorm)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc, use_snorm)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, input):
        x = input
        x_dwt1 = F.interpolate(x, scale_factor=0.5, mode='nearest')
        out = self.RDB1(x_dwt1)
        out = self.RDB2(out)
        out = self.RDB3(out)
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = out * 0.2 + x
        return out

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class Multi_extfea(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True, use_snorm=False):
        super(Multi_extfea, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        if use_snorm:
            self.conv0 = nn.utils.spectral_norm(nn.Conv2d(nf, gc, 1, 1, 1, bias=bias))
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nf, gc, 3, 1, 1, bias=bias))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(nf, gc, 5, 1, 2, bias=bias))
            self.conv3 = nn.utils.spectral_norm(nn.Conv2d(nf, gc, 7, 1, 3, bias=bias))
        else:
            self.conv0 = nn.Conv2d(nf, gc, 1, 1, 1, bias=bias)
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
            self.conv2 = nn.Conv2d(nf, gc, 5, 1, 2, bias=bias)
            self.conv3 = nn.Conv2d(nf, gc, 7, 1, 3, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv_out = nn.Conv2d(gc*4, gc, 3, 1, 1, bias=bias)
        # initialization
        initialize_weights([self.conv0, self.conv1, self.conv2, self.conv3, self.conv_out], 0.1)

    def forward(self, x):
        x0 = self.lrelu(self.conv1(x))
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x))
        x3 = self.lrelu(self.conv3(x))
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_out = self.conv_out(x_cat)
        return x_out

@ARCH_REGISTRY.register()
class DenseMWNet_Mini(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, nframes=5, groups=8, front_RBs=3, gc=32, use_snorm=False, center=None):
        super(DenseMWNet_Mini, self).__init__()
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        act = nn.LeakyReLU(True)
        conv=default_conv
        self.center = nframes // 2 if center is None else center
        self.nframes = nframes

        self.motion_align = MWP_Align(nf=nf, groups=groups)

        self.attention_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        self.conv_first_1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.DWT_1= DWT_CNN(nf)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.DWT_2= DWT_CNN(nf)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.feature_extraction = make_layer(ResidualBlock_noBN_f, front_RBs)
        self.multi_fea_l1 = Multi_extfea(nf=nf, gc=nf, use_snorm=use_snorm)
        self.DWT_3= DWT_CNN(nf)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.DWT_4= DWT_CNN(nf)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        ResidualBlock = []
        for i in range(nb):
            ResidualBlock.append(WDRB_Mini(nf=nf, gc=gc, use_snorm=use_snorm))
        self.ResidualBlock = nn.Sequential(*ResidualBlock)

        ### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.IWT_1 = IWT_CNN(nf)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.IWT_2 = IWT_CNN(nf)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames

        pad_len = self.nframes // 2
        # temporal zero padding, same as 3DCNN
        # padding_fea = x.new_zeros(B, pad_len, C, H, W)
        # x = torch.cat([padding_fea, x, padding_fea], dim=1)

        # flip padding for temporal axis, better than zero padding
        assert N-1>=pad_len
        padding_fea_head = x[:,1:(pad_len+1),:,:,:]
        padding_fea_last = x[:,(-1-pad_len):-1,:,:,:]
        padding_fea_head = torch.flip(padding_fea_head, dims=[1])
        padding_fea_last = torch.flip(padding_fea_last, dims=[1])
        x = torch.cat([padding_fea_head, x, padding_fea_last], dim=1)

        out_list = []
        for i in range(pad_len, N+pad_len):
            cur_input = x[:, i-pad_len:i+pad_len+1, :, :, :]
            B, _, C, H, W = cur_input.size()

            x_center = cur_input[:, self.center, :, :, :].contiguous()

            assert torch.max(x_center)>0
            #### extract LR features
            # L1
            L1_fea = self.lrelu(self.conv_first_1(cur_input.reshape(-1, C, H, W)))
            L1_fea = self.DWT_1(L1_fea)
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.DWT_1(L1_fea)
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
            H, W = H // 4, W // 4

            L1_fea = self.multi_fea_l1(L1_fea)
            L1_fea = self.feature_extraction(L1_fea)
            # L2
            L2_fea = self.lrelu(self.fea_L2_conv1(self.DWT_3(L1_fea)))
            L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
            # L3
            L3_fea = self.lrelu(self.fea_L3_conv1(self.DWT_4(L2_fea)))
            L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

            L1_fea = L1_fea.view(B, self.nframes, -1, H, W)
            L2_fea = L2_fea.view(B, self.nframes, -1, H // 2, W // 2)
            L3_fea = L3_fea.view(B, self.nframes, -1, H // 4, W // 4)

            ref_fea_l = [
                L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
                L3_fea[:, self.center, :, :, :].clone()
            ]
            aligned_fea = []
            for i in range(self.nframes):
                nbr_fea_l = [
                    L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                    L3_fea[:, i, :, :, :].clone()
                ]
                aligned_fea.append(self.motion_align(nbr_fea_l, ref_fea_l))
            aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
            aligned_fea = aligned_fea.view(B, -1, H, W)
            aligned_fea = self.attention_fusion(aligned_fea)

            fea = self.ResidualBlock(aligned_fea)
            out = self.lrelu(self.upconv1(self.IWT_1(fea)))
            out = self.lrelu(self.upconv2(self.IWT_2(out)))
            out = self.conv_last(out)
            out += x_center
            out_list.append(out.unsqueeze(1))
        out_list = torch.cat(out_list, dim=1)

        return out_list

class DenseMWNet_PSNR_Pretrained(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, nframes=5, groups=8, front_RBs=3, gc=32, use_snorm=False, center=None, pretrained=None):
        super(DenseMWNet_PSNR_Pretrained, self).__init__()
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        act = nn.LeakyReLU(True)
        conv=default_conv
        self.center = nframes // 2 if center is None else center
        self.nframes = nframes

        self.motion_align = MWP_Align(nf=nf, groups=groups)

        self.attention_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        self.conv_first_1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.DWT_1= DWT_CNN(nf)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.DWT_2= DWT_CNN(nf)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.feature_extraction = make_layer(ResidualBlock_noBN_f, front_RBs)
        self.multi_fea_l1 = Multi_extfea(nf=nf, gc=nf, use_snorm=use_snorm)
        self.DWT_3= DWT_CNN(nf)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.DWT_4= DWT_CNN(nf)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        ResidualBlock = []
        for i in range(nb):
            ResidualBlock.append(WDRB_Mini(nf=nf, gc=gc, use_snorm=use_snorm))
        self.ResidualBlock = nn.Sequential(*ResidualBlock)

        ### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.IWT_1 = IWT_CNN(nf)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.IWT_2 = IWT_CNN(nf)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if pretrained:
            self.init_weights(pretrained)

    def init_weights(self, pretrained=None, strict=True, param_key='params'):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        logger = get_root_logger()
        load_net = torch.load(pretrained, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading PSNR pretrained model from {pretrained}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self.load_state_dict(load_net, strict=strict)
        logger.info(f'Loaded PSNR pretrained model.')

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        # temporal zero padding, same as 3DCNN
        # padding_fea = x.new_zeros(B, pad_len, C, H, W)
        # x = torch.cat([padding_fea, x, padding_fea], dim=1)

        # flip padding for temporal axis, better than zero padding

        cur_input = x
        B, _, C, H, W = cur_input.size()

        x_center = cur_input[:, self.center, :, :, :].contiguous()

        assert torch.max(x_center)>0
        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first_1(cur_input.reshape(-1, C, H, W)))
        L1_fea = self.DWT_1(L1_fea)
        L1_fea = self.lrelu(self.conv_first_2(L1_fea))
        L1_fea = self.DWT_1(L1_fea)
        L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        H, W = H // 4, W // 4

        L1_fea = self.multi_fea_l1(L1_fea)
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(self.DWT_3(L1_fea)))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(self.DWT_4(L2_fea)))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, self.nframes, -1, H, W)
        L2_fea = L2_fea.view(B, self.nframes, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, self.nframes, -1, H // 4, W // 4)

        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(self.nframes):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.motion_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        aligned_fea = aligned_fea.view(B, -1, H, W)
        aligned_fea = self.attention_fusion(aligned_fea)

        fea = self.ResidualBlock(aligned_fea)
        out = self.lrelu(self.upconv1(self.IWT_1(fea)))
        out = self.lrelu(self.upconv2(self.IWT_2(out)))
        out = self.conv_last(out)
        out += x_center

        return out

@ARCH_REGISTRY.register()
class DenseMWNet_Mini_PSNRbased(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, nframes=5, groups=8, front_RBs=3, gc=32, use_snorm=False, center=None, pretrained=None):
        super(DenseMWNet_Mini_PSNRbased, self).__init__()
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        act = nn.LeakyReLU(True)
        conv=default_conv
        self.center = nframes // 2 if center is None else center
        self.nframes = nframes

        self.motion_align = MWP_Align(nf=nf, groups=groups)

        self.attention_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        self.conv_first_1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.DWT_1= DWT_CNN(nf)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.DWT_2= DWT_CNN(nf)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.feature_extraction = make_layer(ResidualBlock_noBN_f, front_RBs)
        self.multi_fea_l1 = Multi_extfea(nf=nf, gc=nf, use_snorm=use_snorm)
        self.DWT_3= DWT_CNN(nf)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.DWT_4= DWT_CNN(nf)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        ResidualBlock = []
        for i in range(nb):
            ResidualBlock.append(WDRB_Mini(nf=nf, gc=gc, use_snorm=use_snorm))
        self.ResidualBlock = nn.Sequential(*ResidualBlock)

        ### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.IWT_1 = IWT_CNN(nf)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.IWT_2 = IWT_CNN(nf)
        # self.HRconv = nn.Conv2d(in_nc*4, in_nc, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.PSNR_model = DenseMWNet_PSNR_Pretrained(in_nc=in_nc,
                                                     out_nc=out_nc,
                                                     nf=nf,
                                                     nb=nb,
                                                     nframes=nframes,
                                                     groups=groups,
                                                     front_RBs=front_RBs,
                                                     gc=gc,
                                                     use_snorm=False,
                                                     center=None,
                                                     pretrained=pretrained)
        self.PSNR_model.requires_grad_(False)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames

        pad_len = self.nframes // 2
        # temporal zero padding, same as 3DCNN
        # padding_fea = x.new_zeros(B, pad_len, C, H, W)
        # x = torch.cat([padding_fea, x, padding_fea], dim=1)

        # flip padding for temporal axis, better than zero padding
        assert N-1>=pad_len
        padding_fea_head = x[:,1:(pad_len+1),:,:,:]
        padding_fea_last = x[:,(-1-pad_len):-1,:,:,:]
        padding_fea_head = torch.flip(padding_fea_head, dims=[1])
        padding_fea_last = torch.flip(padding_fea_last, dims=[1])
        x = torch.cat([padding_fea_head, x, padding_fea_last], dim=1)

        psnr_list = []
        for i in range(pad_len, N+pad_len):
            cur_input = x[:, i-pad_len:i+pad_len+1, :, :, :]
            psnr_out = self.PSNR_model(cur_input)
            psnr_list.append(psnr_out.unsqueeze(1))
        psnr_list = torch.cat(psnr_list, dim=1)

        B, N, C, H, W = psnr_list.size()
        assert N-1>=pad_len
        padding_fea_head = psnr_list[:,1:(pad_len+1),:,:,:]
        padding_fea_last = psnr_list[:,(-1-pad_len):-1,:,:,:]
        padding_fea_head = torch.flip(padding_fea_head, dims=[1])
        padding_fea_last = torch.flip(padding_fea_last, dims=[1])
        psnr_list = torch.cat([padding_fea_head, psnr_list, padding_fea_last], dim=1)

        out_list = []
        for i in range(pad_len, N+pad_len):
            cur_input = psnr_list[:, i-pad_len:i+pad_len+1, :, :, :]
            B, _, C, H, W = cur_input.size()

            x_center = cur_input[:, self.center, :, :, :].contiguous()

            assert torch.max(x_center)>0
            #### extract LR features
            # L1
            L1_fea = self.lrelu(self.conv_first_1(cur_input.reshape(-1, C, H, W)))
            L1_fea = self.DWT_1(L1_fea)
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.DWT_1(L1_fea)
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
            H, W = H // 4, W // 4

            L1_fea = self.multi_fea_l1(L1_fea)
            L1_fea = self.feature_extraction(L1_fea)
            # L2
            L2_fea = self.lrelu(self.fea_L2_conv1(self.DWT_3(L1_fea)))
            L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
            # L3
            L3_fea = self.lrelu(self.fea_L3_conv1(self.DWT_4(L2_fea)))
            L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

            L1_fea = L1_fea.view(B, self.nframes, -1, H, W)
            L2_fea = L2_fea.view(B, self.nframes, -1, H // 2, W // 2)
            L3_fea = L3_fea.view(B, self.nframes, -1, H // 4, W // 4)

            ref_fea_l = [
                L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
                L3_fea[:, self.center, :, :, :].clone()
            ]
            aligned_fea = []
            for i in range(self.nframes):
                nbr_fea_l = [
                    L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                    L3_fea[:, i, :, :, :].clone()
                ]
                aligned_fea.append(self.motion_align(nbr_fea_l, ref_fea_l))
            aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
            aligned_fea = aligned_fea.view(B, -1, H, W)
            aligned_fea = self.attention_fusion(aligned_fea)

            fea = self.ResidualBlock(aligned_fea)
            out = self.lrelu(self.upconv1(self.IWT_1(fea)))
            out = self.lrelu(self.upconv2(self.IWT_2(out)))
            out = self.conv_last(out)
            out += x_center
            out_list.append(out.unsqueeze(1))
        out_list = torch.cat(out_list, dim=1)

        return out_list


@ARCH_REGISTRY.register()
class DenseMWNet_Mini_PSNR(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, nframes=5, groups=8, front_RBs=3, gc=32, use_snorm=False, center=None):
        super(DenseMWNet_Mini_PSNR, self).__init__()
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        act = nn.LeakyReLU(True)
        conv=default_conv
        self.center = nframes // 2 if center is None else center
        self.nframes = nframes

        self.motion_align = MWP_Align(nf=nf, groups=groups)

        self.attention_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        self.conv_first_1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.DWT_1= DWT_CNN(nf)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.DWT_2= DWT_CNN(nf)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.feature_extraction = make_layer(ResidualBlock_noBN_f, front_RBs)
        self.multi_fea_l1 = Multi_extfea(nf=nf, gc=nf, use_snorm=use_snorm)
        self.DWT_3= DWT_CNN(nf)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.DWT_4= DWT_CNN(nf)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        ResidualBlock = []
        for i in range(nb):
            ResidualBlock.append(WDRB_Mini(nf=nf, gc=gc, use_snorm=use_snorm))
        self.ResidualBlock = nn.Sequential(*ResidualBlock)

        ### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.IWT_1 = IWT_CNN(nf)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.IWT_2 = IWT_CNN(nf)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        # temporal zero padding, same as 3DCNN
        # padding_fea = x.new_zeros(B, pad_len, C, H, W)
        # x = torch.cat([padding_fea, x, padding_fea], dim=1)

        # flip padding for temporal axis, better than zero padding

        cur_input = x
        B, _, C, H, W = cur_input.size()

        x_center = cur_input[:, self.center, :, :, :].contiguous()

        assert torch.max(x_center)>0
        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first_1(cur_input.reshape(-1, C, H, W)))
        L1_fea = self.DWT_1(L1_fea)
        L1_fea = self.lrelu(self.conv_first_2(L1_fea))
        L1_fea = self.DWT_1(L1_fea)
        L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        H, W = H // 4, W // 4

        L1_fea = self.multi_fea_l1(L1_fea)
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(self.DWT_3(L1_fea)))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(self.DWT_4(L2_fea)))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, self.nframes, -1, H, W)
        L2_fea = L2_fea.view(B, self.nframes, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, self.nframes, -1, H // 4, W // 4)

        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(self.nframes):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.motion_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        aligned_fea = aligned_fea.view(B, -1, H, W)
        aligned_fea = self.attention_fusion(aligned_fea)

        fea = self.ResidualBlock(aligned_fea)
        out = self.lrelu(self.upconv1(self.IWT_1(fea)))
        out = self.lrelu(self.upconv2(self.IWT_2(out)))
        out = self.conv_last(out)
        out += x_center

        return out

@ARCH_REGISTRY.register()
class DenseMWNet_ECCV(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, groups=8, front_RBs=3, gc=32, use_snorm=False, center=None):
        super(DenseMWNet_ECCV, self).__init__()
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        act = nn.LeakyReLU(True)
        conv=default_conv
        self.center = 1 # 3 frames as input

        self.DWT = DWT()
        self.IWT = IWT()

        cov_path_0 = [BBlock(conv, in_nc*4, nf, 3, act=act, use_snorm=use_snorm)]
        cov_path_0.append(BBlock(conv, nf, in_nc*4, 3, act=act, use_snorm=use_snorm))
        cov_path_1 = [BBlock(conv, in_nc*4, nf, 3, act=act, use_snorm=use_snorm)]
        cov_path_1.append(BBlock(conv, nf, nf, 3, act=act, use_snorm=use_snorm))
        cov_path_2 = [BBlock(conv, in_nc*4, nf, 3, act=act, use_snorm=use_snorm)]
        cov_path_2.append(BBlock(conv, nf, nf, 3, act=act, use_snorm=use_snorm))
        cov_path_3 = [BBlock(conv, in_nc*4, nf, 3, act=act, use_snorm=use_snorm)]
        cov_path_3.append(BBlock(conv, nf, nf, 3, act=act, use_snorm=use_snorm))

        self.cov_path_1 = nn.Sequential(*cov_path_1)
        self.cov_path_2 = nn.Sequential(*cov_path_2)
        self.cov_path_3 = nn.Sequential(*cov_path_3)

        self.motion_align = DWT_motion_Pyramid(gc=nf)

        ResidualBlock = []
        for i in range(nb):
            ResidualBlock.append(WDRB_Mini(nf=nf, gc=gc, use_snorm=use_snorm))
        self.ResidualBlock = nn.Sequential(*ResidualBlock)

        if use_snorm:
            self.conv_first_ = nn.utils.spectral_norm(nn.Conv2d(nf*3, nf, 3, 1, 1, bias=True))
            self.conv_second = nn.utils.spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
            self.trunk_conv = nn.utils.spectral_norm(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
            self.conv_aft = nn.utils.spectral_norm(nn.Conv2d(nf//4, out_nc, 3, 1, 1, bias=True))
            self.conv_last_ = nn.utils.spectral_norm(nn.Conv2d(out_nc, out_nc, 3, 1, 1, bias=True))
        else:
            self.conv_first_ = nn.Conv2d(nf*3, nf, 3, 1, 1, bias=True)
            self.conv_second = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_aft = nn.Conv2d(nf//4, out_nc, 3, 1, 1, bias=True)
            self.conv_last_ = nn.Conv2d(out_nc, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames

        cur_input = x
        B, _, C, H, W = cur_input.size()

        x_center = cur_input[:, self.center, :, :, :].contiguous()
        assert torch.max(x_center)>0

        compx_1 = self.motion_align(torch.cat([x[:,self.center,:,:,:], x[:,0,:,:,:]], dim=1))
        compx_2 = self.motion_align(torch.cat([x[:,self.center,:,:,:], x[:,2,:,:,:]], dim=1))
        #### extract LR features
        fea_1 = self.cov_path_1(self.DWT(compx_1))
        fea_2 = self.cov_path_2(self.DWT(x[:,self.center,:,:,:]))
        fea_3 = self.cov_path_3(self.DWT(compx_2))

        fea = torch.cat((fea_1, fea_2, fea_3), 1)
        fea = self.lrelu(self.conv_first_(fea))
        fea = self.lrelu(self.conv_second(fea))

        trunk = self.trunk_conv(self.ResidualBlock(fea))
        fea = trunk
        fea = self.conv_aft(self.IWT(fea)) + x_center
        out = self.conv_last_(fea)

        return out, compx_1, compx_2
