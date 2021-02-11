import torch
import torch.nn as nn
import torchvision
import models.modules.Wavelet as common
import functools
import math

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

class singlescale_net(nn.Module):
    def __init__(self, input_dim, ch, n_layer, act_layer, norm_layer, sn, dilate_scales):
        super(singlescale_net, self).__init__()
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
        self.model = nn.Sequential(*model)
        self.final_cov = NormActConv(prech // 2, 1, 3, 1, 1, act_layer, norm_layer, sn)


    def forward(self, x):
        x = self.model(x)
        x = self.final_cov(x)

        return x

class Discriminator_Multi_light(nn.Module):
    def __init__(self, in_nc, nf, sn=True, n_scale=3, n_layer=3,
    actType='lrelu', normType='instance', dilate_scales=(1,2,4,6)):
        super(Discriminator_Multi_light, self).__init__()
        self.DWT = common.DWT()
        self.n_scale = n_scale
        norm_layer = get_norm_layer(normType)
        act_layer = get_activation_layer(actType)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        for i in range(self.n_scale):
            disc = singlescale_net(int(in_nc*math.pow(4,i)), nf, n_layer, act_layer, norm_layer, sn, dilate_scales)
            setattr(self, 'disc_{}'.format(i), disc)

    def forward(self, x):
        outs = []
        x2 = x
        for i in range(self.n_scale):
            disc = getattr(self, 'disc_{}'.format(i))
            outD = disc(x)
            outs.append(outD)
            x = self.DWT(x)
        return outs

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

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output
