import torch
import torch.nn as nn
import torchvision
from .wavelet_util import DWT
import functools
import math
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F

def get_norm_layer(layer_type='instance'):
  if layer_type == 'batch':
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
  elif layer_type == 'instance':
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
  elif layer_type == 'none':
    norm_layer = None
  else:
    raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
  return norm_layer


def get_activation_layer(layer_type='relu'):
  if layer_type == 'relu':
    nl_layer = functools.partial(nn.ReLU, inplace=True)
  elif layer_type == 'lrelu':
    nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
  elif layer_type == 'elu':
    nl_layer = functools.partial(nn.ELU, inplace=True)
  elif layer_type == 'none':
    nl_layer = None
  else:
    raise NotImplementedError('activitation [%s] is not found' % layer_type)
  return nl_layer

def init_weights(net, init_type='xavier', init_gain=1):
    def init_func(m):  # define the initialization function
      classname = m.__class__.__name__
      if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
          nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
          nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
          nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
          raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)
      elif classname.find(
              'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        nn.init.normal_(m.weight.data, 1.0, init_gain)
        nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class DWT_CNN(nn.Module):
    def __init__(self, ch):
        super(DWT_CNN, self).__init__()
        self.DWT= DWT()
        self.Conv = nn.Conv2d(ch*4, ch, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return(self.lrelu(self.Conv(self.DWT(x))))

@ARCH_REGISTRY.register()
class Discriminator_Multi_3DLIGHT(nn.Module):
    def __init__(self, in_nc, nf, n_scale=3):
        super(Discriminator_Multi_3DLIGHT, self).__init__()
        self.DWT = DWT_CNN(in_nc)
        self.n_scale = n_scale
        # self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        for i in range(self.n_scale):
            disc = TemporalDiscriminator(in_nc, nf)
            setattr(self, 'disc_{}'.format(i), disc)

    def forward(self, x):
        outs = []
        x = x.permute(0,2,1,3,4).contiguous()
        for i in range(self.n_scale):
            disc = getattr(self, 'disc_{}'.format(i))
            outD = disc(x)
            outs.append(outD)
            x_dic = []
            for i in range(x.size(2)):
                x_dic.append(self.DWT(x[:,:,i,:,:]).unsqueeze(2))
            x = torch.cat(x_dic, dim=2)
        return outs

class TemporalDiscriminator(nn.Module):

    def __init__(self, in_channel, chn=128):
        super().__init__()

        gain = 2 ** 0.5

        self.pre_conv = nn.Sequential(
            STConv3d(in_channel, 2*chn, 3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d((1,2,2))
        )
        self.pre_skip = STConv3d(in_channel, 2*chn, 1)

        self.res3d = Res3dBlock(2*chn, 4*chn, bn=False, upsample=False, downsample=True)

        self.res3d_1 = nn.Sequential(
            STConv3d(4*chn, 4*chn, 3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d((1,2,2)),
            STConv3d(4*chn, 2*chn, 3, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool3d((1,2,2))
        )

        self.final_conv = nn.utils.spectral_norm(nn.Conv2d(2*chn, 1, 3, 1, padding=1))

    def forward(self, x):
        # pre-process with avg_pool2d to reduce tensor size
        B, T, C, H, W = x.size()

        out = self.pre_conv(x)
        out = out + self.pre_skip(F.avg_pool3d(x, (1,2,2)))
        out = self.res3d(out) # B x C x T x W x H
        out = self.res3d_1(out) # B x C x T x W x H

        #reshape to BTxCxWxH
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        B, T, C, W, H = out.size()
        out = out.view(B*T, C, W, H)

        out = self.final_conv(out)
        _, _, W, H = out.size()

        return out.reshape(B, T, -1, W, H).squeeze(2)

class Res3dBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3,
                 padding=1, stride=1, bn=False,
                 activation=F.leaky_relu, upsample=True, downsample=False):
        super().__init__()

        gain = 2 ** 0.5

        self.conv0 = STConv3d(in_channel, out_channel,
                                            kernel_size, stride, padding,
                                            bias=True if bn else True)
        self.conv1 = STConv3d(out_channel, out_channel,
                                            kernel_size, stride, padding,
                                            bias=True if bn else True)

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_sc = STConv3d(in_channel, out_channel,1, 1, 0)
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.HyperBN = ConditionalNorm(in_channel, 148)
            self.HyperBN_1 = ConditionalNorm(out_channel, 148)

    def forward(self, input, condition=None):
        out = input

        if self.bn:
            # print('condition',condition.size()) #condition torch.Size([4, 148])
            out = self.HyperBN(out, condition)
        out = self.activation(out)
        if self.upsample:
            # TODO different form papers
            out = F.upsample(out, scale_factor=2)
        out = self.conv0(out)
        if self.bn:
            out = self.HyperBN_1(out, condition)
        out = self.activation(out)
        out = self.conv1(out)

        if self.downsample:
            out = F.avg_pool3d(out, (1,2,2))

        if self.skip_proj:
            skip = input
            if self.upsample:
                # TODO different form papers
                skip = F.upsample(skip, scale_factor=2)
            skip = self.conv_sc(skip)
            if self.downsample:
                skip = F.avg_pool3d(skip, (1,2,2))

        else:
            skip = input

        return out + skip

class STConv3d(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride=1,padding=0, bias=True):
        super(STConv3d, self).__init__()
        self.conv = nn.utils.spectral_norm(nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size),stride=(1,stride,stride),padding=(0,padding,padding),bias=bias))
        self.conv2 = nn.utils.spectral_norm(nn.Conv3d(out_planes,out_planes,kernel_size=(kernel_size,1,1),stride=(stride,1,1),padding=(padding,0,0),bias=bias))

        # self.bn=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

        # self.bn2=nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu2=nn.ReLU(inplace=True)

        nn.init.normal(self.conv2.weight,mean=0,std=0.01)
        nn.init.constant(self.conv2.bias,0)

    def forward(self,x):
        x=self.conv(x)
        #x=self.conv2(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.relu2(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256, scales=(0, 1, 4, 8, 12), sn=False):
      super(ASPP, self).__init__()
      self.scales = scales
      for dilate_rate in self.scales:
          if dilate_rate == -1:
              break
          if dilate_rate == 0:
              layers = [nn.AdaptiveAvgPool2d((1, 1))]
              if sn:
                layers += [nn.utils.spectral_norm(nn.Conv2d(in_channel, depth, 1, 1))]
              else:
                layers += [nn.Conv2d(in_channel, depth, 1, 1)]
              setattr(self, 'dilate_layer_{}'.format(dilate_rate), nn.Sequential(*layers))
          elif dilate_rate == 1:
              if sn:
                layers = [nn.utils.spectral_norm(nn.Conv2d(in_channel, depth, 1, 1))]
              else:
                layers = [nn.Conv2d(in_channel, depth, 1, 1)]
              setattr(self, 'dilate_layer_{}'.format(dilate_rate), nn.Sequential(*layers))
          else:
              if sn:
                layers = [nn.utils.spectral_norm(nn.Conv2d(in_channel, depth, 3, 1, dilation=dilate_rate, padding=dilate_rate))]
              else:
                layers = [nn.Conv2d(in_channel, depth, 3, 1, dilation=dilate_rate, padding=dilate_rate)]
              setattr(self, 'dilate_layer_{}'.format(dilate_rate), nn.Sequential(*layers))

      self.conv_1x1_output = nn.Conv2d(depth * len(scales), depth, 1, 1)


    def forward(self, x):
      dilate_outs = []
      for dilate_rate in self.scales:
          if dilate_rate == -1:
              return x
          if dilate_rate == 0:
              layer = getattr(self, 'dilate_layer_{}'.format(dilate_rate))
              size = x.shape[2:]
              tempout = F.interpolate(layer(x), size=size, mode='bilinear', align_corners=True)
              dilate_outs.append(tempout)
          else:
              layer = getattr(self, 'dilate_layer_{}'.format(dilate_rate))
              dilate_outs.append(layer(x))
      out = self.conv_1x1_output(torch.cat(dilate_outs, dim=1))
      return out

class NormActConv(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, act_layer, norm_layer, sn=False, padding_type = 'zero'):
        super(NormActConv, self).__init__()
        layers = []
        if padding_type == 'reflect':
            layers += [norm_layer(n_in),
                      act_layer(),
                      nn.ReflectionPad2d(padding)]
            p = 0
        elif padding_type == 'replicate':
            layers += [norm_layer(n_in),
                      act_layer(),
                    nn.ReplicationPad2d(padding)]
            p = 0
        elif padding_type == 'zero':
            layers += [norm_layer(n_in),
                      act_layer()]
            p = padding
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        if sn:
            layers += [nn.utils.spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=p))]
        else:
            layers += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=p)]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

@ARCH_REGISTRY.register()
class Discriminator_ECCV(nn.Module):
    def __init__(self, in_nc, nf, sn=True, n_scale=2, n_layer=3,
    actType='lrelu', normType='instance', dilate_scales=(1,2,4,6), cond=False):
        super(Discriminator_ECCV, self).__init__()
        self.DWT = DWT()
        self.n_scale = n_scale
        norm_layer = get_norm_layer(normType)
        act_layer = get_activation_layer(actType)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.cond = cond
        if cond:
            input_dim = input_dim + 1
        for i in range(self.n_scale):
            disc = self.singlescale_net(int(in_nc*math.pow(4,i)), nf, n_layer, act_layer, norm_layer, sn, dilate_scales)
            setattr(self, 'disc_{}'.format(i), disc)

    def singlescale_net(self, input_dim, ch, n_layer, act_layer, norm_layer, sn, dilate_scales):
        model = []
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(input_dim, ch, kernel_size=3, stride=2, padding=1))]
        else:
            model += [nn.Conv2d(input_dim, ch, kernel_size=3, stride=2, padding=1)]
        prech = ch
        tch = min(prech * 2, ch*8)
        for _ in range(1, n_layer):
            model += [NormActConv(prech, tch, 3, 2, 1, act_layer, norm_layer, sn)]
            prech = tch
            tch = min(prech * 2, ch*8)
        model += [ASPP(prech, prech // 2, dilate_scales, sn)]
        model += [NormActConv(prech // 2, 1, 3, 1, 1, act_layer, norm_layer, sn)]
        return nn.Sequential(*model)

    def forward(self, x):
        outs = []
        x2 = x
        nanoutD = 0
        infoutD = 0
        for i in range(self.n_scale):
            disc = getattr(self, 'disc_{}'.format(i))
            outD = disc(x)
            outs.append(outD)
            x = self.DWT(x)
        return outs
