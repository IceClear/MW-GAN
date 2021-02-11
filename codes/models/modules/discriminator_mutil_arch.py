import torch
import torch.nn as nn
import torchvision
import models.modules.Wavelet as common


class Discriminator_Multi_128(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_Multi_128, self).__init__()
        self.DWT = common.DWT()
        self.IWT = common.IWT()

        self.level0 = common.VGG_conv0(in_nc,nf)
        self.level1 = common.VGG_conv1(in_nc*4,nf)
        self.level2 = common.VGG_conv2(in_nc*16,nf)

        self.linear1 = nn.Linear(17152, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.DWT(x)
        x2 = self.DWT(x1)

        fea0 = self.level0(x)
        fea1 = self.level1(x1)
        fea2 = self.level2(x2)

        fea0 = fea0.view(fea0.size(0), -1)
        fea1 = fea1.view(fea1.size(0), -1)
        fea2 = fea2.view(fea2.size(0), -1)


        fea = torch.cat((fea0, fea1, fea2), 1)

        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out


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
