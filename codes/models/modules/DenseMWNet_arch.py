import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil
import models.modules.Wavelet as common
import torch.nn.init as init
from torch.nn.parameter import Parameter
import models.modules.pac as pac
# try:
#     from models.modules.dcn.deform_conv import ModulatedDeformConvPack as DCN
# except ImportError:
#     raise ImportError('Failed to import DCNv2 module.')
# from pytorch_wavelets import SWTForward, DWTForward # (or import DWT, IDWT)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class DWT_motion_Pyramid(nn.Module):
    def __init__(self, gc=64, bias=True, use_snorm=False):
        super(DWT_motion_Pyramid, self).__init__()

        self.DWT= common.DWT()
        self.IWT = common.IWT()

        self.warp_level_1 = Motion_fea_5c(nf=2+1, out=2, gc=gc, bias=bias, use_snorm=use_snorm)
        self.warp_level_2 = Motion_fea_5c(nf=2*4*2+1, out=2*4 ,gc=gc, bias=bias, use_snorm=use_snorm)
        self.warp_level_3 = Motion_fea_5c(nf=2*4*4, out=2*4*4, gc=gc, bias=bias, use_snorm=use_snorm)

        self.conv_1 = nn.Conv2d(2, 2, 1, 1, 0, bias=bias)
        self.conv_2 = nn.Conv2d(2, 2, 1, 1, 0, bias=bias)
        self.conv_3_1 = nn.Conv2d(2*4, 2*4, 1, 1, 0, bias=bias)
        self.conv_3_2 = nn.Conv2d(2, 2, 1, 1, 0, bias=bias)
        self.conv_3_3 = nn.Conv2d(1, 1, 3, 2, 1, bias=bias)
        self.conv_last = nn.Conv2d(1, 1, 1, 1, 0, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_to_warp = x[:,1:,:,:]
        x_2 = self.DWT(x)
        x_3 = self.DWT(x_2)

        fea_3 = self.warp_level_3(x_3)
        fea_3 = self.tanh(self.conv_3_1(self.IWT(fea_3)))
        fea_3_to_warp = self.conv_3_2(self.IWT(fea_3))

        warp_3 = mutil.flow_warp(x_to_warp,fea_3_to_warp)
        warp_3_downsample = self.lrelu(self.conv_3_3(warp_3))
        fea_2 = torch.cat([x_2, fea_3, warp_3_downsample], dim=1)
        fea_2 = self.warp_level_2(fea_2)
        fea_2 = self.tanh(self.conv_2(self.IWT(fea_2)))
        warp_2 = mutil.flow_warp(x_to_warp,fea_2)
        fea_1 = torch.cat([x, warp_2], dim=1)
        fea_1 = self.warp_level_1(fea_1)
        fea_1 = self.tanh(self.conv_1(fea_1))
        warp_1 = mutil.flow_warp(x_to_warp,fea_1)

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
        self.L3_pcnpack = pac.PacConv2d(nf, nf, 3, stride=1, padding=1, dilation=1)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_pcnpack = pac.PacConv2d(nf, nf, 3, stride=1, padding=1, dilation=1)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_pcnpack = pac.PacConv2d(nf, nf, 3, stride=1, padding=1, dilation=1)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_pcnpack = pac.PacConv2d(nf, nf, 3, stride=1, padding=1, dilation=1)

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
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class WDRB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, use_snorm=False):
        super(WDRB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc, use_snorm)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc, use_snorm)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc, use_snorm)
        self.dwt = common.DWT()
        self.iwt = common.IWT()
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
        mutil.initialize_weights([self.conv0, self.conv1, self.conv2, self.conv3, self.conv_out], 0.1)

    def forward(self, x):
        x0 = self.lrelu(self.conv1(x))
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x))
        x3 = self.lrelu(self.conv3(x))
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_out = self.conv_out(x_cat)
        return x_out


class DenseMWNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, nframes=5, groups=8, front_RBs=3, gc=128, use_snorm=False, center=None):
        super(DenseMWNet, self).__init__()
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        act = nn.LeakyReLU(True)
        conv=common.default_conv
        self.center = nframes // 2 if center is None else center

        self.DWT= common.DWT()
        self.IWT = common.IWT()

        self.motion_align = MWP_Align(nf=nf, groups=groups)

        self.attention_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        self.conv_first_1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf*4, nf, 3, 1, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf*4, nf, 3, 1, 1, bias=True)

        self.feature_extraction = mutil.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.multi_fea_l1 = Multi_extfea(nf=nf, gc=nf, use_snorm=use_snorm)
        self.fea_L2_conv1 = nn.Conv2d(nf*4, nf, 3, 1, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf*4, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # self.motion_align = DWT_motion_Pyramid(gc=nf)
        
        ResidualBlock = []
        for i in range(nb):
            ResidualBlock.append(WDRB(nf=nf*4, gc=gc, use_snorm=use_snorm))
        self.ResidualBlock = nn.Sequential(*ResidualBlock)

        ### upsampling
        self.upconv1 = nn.Conv2d(nf//4, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf//4, nf, 3, 1, 1, bias=True)
        # self.HRconv = nn.Conv2d(in_nc*4, in_nc, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames

        x_center = x[:, self.center, :, :, :].contiguous()
        #### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
        L1_fea = self.DWT(L1_fea)
        L1_fea = self.lrelu(self.conv_first_2(L1_fea))
        L1_fea = self.DWT(L1_fea)
        L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        H, W = H // 4, W // 4

        L1_fea = self.multi_fea_l1(L1_fea)
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(self.DWT(L1_fea)))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(self.DWT(L2_fea)))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.motion_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        aligned_fea = aligned_fea.view(B, -1, H, W)
        aligned_fea = self.attention_fusion(aligned_fea)

        fea = self.ResidualBlock(aligned_fea)
        out = self.lrelu(self.upconv1(self.IWT(fea)))
        out = self.lrelu(self.upconv2(self.IWT(out)))
        out = self.conv_last(out)
        out += x_center

        return out
