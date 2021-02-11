import torch
import logging
import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.discriminator_mutil_arch as MWGAN_arch
import models.modules.discriminator_mutil_light as MWGAN_arch_light
import models.modules.DenseMWNet_arch as DenseMWNet_arch
logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    if which_model == 'DenseMWNet':
        netG = DenseMWNet_arch.DenseMWNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], use_snorm=opt_net['use_snorm'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'discriminator_Multi_128':
        netD = MWGAN_arch.Discriminator_Multi_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'discriminator_Multi_light':
        netD = MWGAN_arch_light.Discriminator_Multi_light(in_nc=opt_net['in_nc'], nf=opt_net['nf'], n_scale=opt_net['n_scale'], sn=opt_net['use_snorm'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
