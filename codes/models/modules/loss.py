import torch
import torch.nn as nn

class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

class loss_Textures(nn.Module):

    def __init__(self, nc=1, alpha=1.2, margin=0):
        super(loss_Textures, self).__init__()
        self.nc = nc
        self.alpha = alpha
        self.margin = margin

    def forward(self, x, y):
        xi = x.contiguous().view(x.size(0), -1, self.nc, x.size(2), x.size(3))
        yi = y.contiguous().view(y.size(0), -1, self.nc, y.size(2), y.size(3))

        xi2 = torch.sum(xi * xi, dim=2)
        yi2 = torch.sum(yi * yi, dim=2)

        out = nn.functional.relu(yi2.mul(self.alpha) - xi2 + self.margin)

        return torch.mean(out)



# Define GAN loss: [ | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        if isinstance(input,list):
            all_losses = []
            for prediction in input:
                if self.gan_type in ['lsgan', 'gan','ragan']:
                    target_tensor = self.get_target_label(prediction, target_is_real)
                    loss_temp = self.loss(prediction, target_tensor)
                elif self.gan_type == 'wgangp':
                    if target_is_real:
                        loss_temp = -prediction.mean()
                    else:
                        loss_temp = prediction.mean()
                all_losses.append(loss_temp)
            loss = sum(all_losses)
        else:
            target_label = self.get_target_label(input, target_is_real)
            loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss
