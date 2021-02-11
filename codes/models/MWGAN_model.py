import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import GANLoss, CharbonnierLoss, loss_Textures, GramMatrix
import cv2
import models.modules.Wavelet as common
from models.modules.discriminator_mutil_light import init_weights
from utils import util
import torch.nn.functional as F
import lpips

logger = logging.getLogger('base')


class MWGANModel(BaseModel):
    def __init__(self, opt):
        super(MWGANModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.train_opt = opt['train']

        self.DWT = common.DWT()
        self.IWT = common.IWT()

        # define networks and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        # pretrained_dict = torch.load(opt['path']['pretrain_model_others'])
        # netG_dict = self.netG.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in netG_dict}
        # netG_dict.update(pretrained_dict)
        # self.netG.load_state_dict(netG_dict)

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        if self.is_train:
            if not self.train_opt['only_G']:
                self.netD = networks.define_D(opt).to(self.device)
                # init_weights(self.netD)
                if opt['dist']:
                    self.netD = DistributedDataParallel(self.netD,
                                                        device_ids=[torch.cuda.current_device()])
                else:
                    self.netD = DataParallel(self.netD)

                self.netG.train()
                self.netD.train()
            else:
                self.netG.train()
        else:
            self.netG.train()

        # define losses, optimizer and scheduler
        if self.is_train:

            # G pixel loss
            if self.train_opt['pixel_weight'] > 0:
                l_pix_type = self.train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'cb':
                    self.cri_pix = CharbonnierLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = self.train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None

            if self.train_opt['lpips_weight'] > 0:
                l_lpips_type = self.train_opt['lpips_criterion']
                if l_lpips_type == 'lpips':
                    self.cri_lpips = lpips.LPIPS(net='vgg').to(self.device)
                    if opt['dist']:
                        self.cri_lpips = DistributedDataParallel(self.cri_lpips,
                                                            device_ids=[torch.cuda.current_device()])
                    else:
                        self.cri_lpips = DataParallel(self.cri_lpips)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_lpips_type))
                self.l_lpips_w = self.train_opt['lpips_weight']
            else:
                logger.info('Remove lpips loss.')
                self.cri_lpips = None

            # G feature loss
            if self.train_opt['feature_weight'] > 0:
                self.fea_trans = GramMatrix().to(self.device)
                l_fea_type = self.train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                elif l_fea_type == 'cb':
                    self.cri_fea = CharbonnierLoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = self.train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    self.netF = DistributedDataParallel(self.netF,
                                                        device_ids=[torch.cuda.current_device()])
                else:
                    self.netF = DataParallel(self.netF)

            # GD gan loss
            self.cri_gan = GANLoss(self.train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = self.train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = self.train_opt['D_update_ratio'] if self.train_opt['D_update_ratio'] else 1
            self.D_init_iters = self.train_opt['D_init_iters'] if self.train_opt['D_init_iters'] else 0

            # optimizers
            # G
            wd_G = self.train_opt['weight_decay_G'] if self.train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=self.train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(self.train_opt['beta1_G'], self.train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)

            if not self.train_opt['only_G']:
                # D
                wd_D = self.train_opt['weight_decay_D'] if self.train_opt['weight_decay_D'] else 0
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.train_opt['lr_D'],
                                                    weight_decay=wd_D,
                                                    betas=(self.train_opt['beta1_D'], self.train_opt['beta2_D']))
                self.optimizers.append(self.optimizer_D)

            # schedulers
            if self.train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, self.train_opt['lr_steps'],
                                                         restarts=self.train_opt['restarts'],
                                                         weights=self.train_opt['restart_weights'],
                                                         gamma=self.train_opt['lr_gamma'],
                                                         clear_state=self.train_opt['clear_state']))
            elif self.train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, self.train_opt['T_period'], eta_min=self.train_opt['eta_min'],
                            restarts=self.train_opt['restarts'], weights=self.train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        if self.is_train:
            if not self.train_opt['only_G']:
                self.print_network()  # print network
        else:
            self.print_network()  # print network

        try:
            self.load()  # load G and D if needed
            print('Pretrained model loaded')
        except Exception as e:
            print('No pretrained model found')

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            # print(self.var_H.size())
            self.var_H = self.var_H.squeeze(1)
            # self.var_H = self.DWT(self.var_H)

            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)
            # print(self.var_ref.size())
            self.var_ref = self.var_ref.squeeze(1)
            # print(s)
            # self.var_ref = self.DWT(self.var_ref)


    def process_list(self, input1, input2):
        result = []
        for index in range(len(input1)):
            result.append(input1[index]-torch.mean(input2[index]))
        return result

    def optimize_parameters(self, step):
        # G
        if not self.train_opt['only_G']:
            for p in self.netD.parameters():
                p.requires_grad = False

        self.optimizer_G.zero_grad()

        self.fake_H = self.netG(self.var_L)

        # self.var_H = self.var_H.squeeze(1)

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix

            if self.cri_lpips:  # pixel loss
                l_g_lpips = torch.mean(self.l_lpips_w * self.cri_lpips.forward(self.fake_H, self.var_H))
                l_g_total += l_g_lpips

            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                real_fea_trans = self.fea_trans(real_fea)
                fake_fea_trans = self.fea_trans(fake_fea)
                l_g_fea_trans = self.l_fea_w * self.cri_fea(fake_fea_trans, real_fea_trans) * 10
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
                l_g_total += l_g_fea_trans

            if not self.train_opt['only_G']:
                pred_g_fake = self.netD(self.fake_H)

                if self.opt['train']['gan_type'] == 'gan':
                    l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                elif self.opt['train']['gan_type'] == 'ragan':
                    # self.var_ref = self.var_ref[:,1:,:,:]
                    pred_d_real = self.netD(self.var_ref)
                    pred_d_real = [ele.detach() for ele in pred_d_real]
                    l_g_gan = self.l_gan_w * (
                        self.cri_gan(self.process_list(pred_d_real,pred_g_fake), False) +
                        self.cri_gan(self.process_list(pred_g_fake,pred_d_real), True)) / 2
                elif self.opt['train']['gan_type'] == 'lsgan_ra':
                    # self.var_ref = self.var_ref[:,1:,:,:]
                    pred_d_real = self.netD(self.var_ref)
                    pred_d_real = [ele.detach() for ele in pred_d_real]
                    # l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                    l_g_gan = self.l_gan_w * (
                        self.cri_gan(self.process_list(pred_d_real,pred_g_fake), False) +
                        self.cri_gan(self.process_list(pred_g_fake,pred_d_real), True)) / 2
                elif self.opt['train']['gan_type'] == 'lsgan':
                    # self.var_ref = self.var_ref[:,1:,:,:]
                    l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
                l_g_total += l_g_gan

            l_g_total.backward()
            self.optimizer_G.step()
        else:
            self.var_ref = self.var_ref

        if not self.train_opt['only_G']:
            # D
            for p in self.netD.parameters():
                p.requires_grad = True

            self.optimizer_D.zero_grad()
            l_d_total = 0
            pred_d_real= self.netD(self.var_ref)
            pred_d_fake= self.netD(self.fake_H.detach())  # detach to avoid BP to G

            if self.opt['train']['gan_type'] == 'gan':
                l_d_real = self.cri_gan(pred_d_real, True)
                l_d_fake = self.cri_gan(pred_d_fake, False)
                l_d_total += l_d_real + l_d_fake
            elif self.opt['train']['gan_type'] == 'ragan':
                l_d_real = self.cri_gan(self.process_list(pred_d_real, pred_d_fake), True)
                l_d_fake = self.cri_gan(self.process_list(pred_d_fake, pred_d_real), False)
                l_d_total += (l_d_real + l_d_fake) / 2
            elif self.opt['train']['gan_type'] == 'lsgan':
                l_d_real = self.cri_gan(pred_d_real, True)
                l_d_fake = self.cri_gan(pred_d_fake, False)
                l_d_total += (l_d_real + l_d_fake) / 2

            l_d_total.backward()
            self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()/self.l_pix_w
            if self.cri_lpips:
                self.log_dict['l_g_lpips'] = l_g_lpips.item()/self.l_lpips_w
            if not self.train_opt['only_G']:
                self.log_dict['l_g_gan'] = l_g_gan.item()/self.l_gan_w
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()/self.l_fea_w
                self.log_dict['l_g_fea_trans'] = l_g_fea_trans.item()/self.l_fea_w/10

        if not self.train_opt['only_G']:
            self.log_dict['l_d_real'] = l_d_real.item()
            self.log_dict['l_d_fake'] = l_d_fake.item()
            self.log_dict['D_real'] = torch.mean(pred_d_real[0].detach())
            self.log_dict['D_fake'] = torch.mean(pred_d_fake[0].detach())

    def test(self, load_path=None, input_u=None, input_v=None):

        if load_path is not None:
            self.load_network(load_path, self.netG, self.opt['path']['strict_load'])
            print('***************************************************************')
            print('Load model successfully')
            print('***************************************************************')

        self.netG.eval()
        # self.var_H = self.var_H.squeeze(1)
        # img_to_write = self.var_L.detach()[0].float().cpu()
        # print(img_to_write.size())
        # cv2.imwrite('./test.png',img_to_write.numpy().transpose(1,2,0)*255)
        with torch.no_grad():
            if self.var_L.size()[-1] > 1280:
                width = self.var_L.size()[-1]
                height = self.var_L.size()[-2]
                fake_list = []
                for height_start in [0, int(height / 2)]:
                    for width_start in [0, int(width / 2)]:
                        self.fake_slice = self.netG(self.var_L[:,:,:,height_start: (height_start + int(height / 2)), width_start: (width_start + int(width / 2))])
                        fake_list.append(self.fake_slice)
                enhanced_frame_h1 = torch.cat([fake_list[0],fake_list[2]],2)
                enhanced_frame_h2 = torch.cat([fake_list[1],fake_list[3]],2)
                self.fake_H = torch.cat([enhanced_frame_h1,enhanced_frame_h2],3)
            else:
                self.fake_H = self.netG(self.var_L)
            if input_u is not None and input_v is not None:
                self.var_L_u = input_u.to(self.device)
                self.var_L_v = input_v.to(self.device)
                self.fake_H_u_s = self.netG(self.var_L_u.float())
                self.fake_H_v_s = self.netG(self.var_L_v.float())
                # self.fake_H_u = torch.cat((self.fake_H_u_s[0], self.fake_H_u_s[1]), 1)
                # self.fake_H_v = torch.cat((self.fake_H_v_s[0], self.fake_H_v_s[1]), 1)
                self.fake_H_u = self.fake_H_u_s
                self.fake_H_v = self.fake_H_v_s
                # self.fake_H_u = self.IWT(self.fake_H_u)
                # self.fake_H_v = self.IWT(self.fake_H_v)
            else:
                self.fake_H_u = None
                self.fake_H_v = None
            self.fake_H_all = self.fake_H
            if self.opt['network_G']['out_nc'] == 4:
                self.fake_H_all = self.IWT(self.fake_H_all)
                if input_u is not None and input_v is not None:
                    self.fake_H_u = self.IWT(self.fake_H_u)
                    self.fake_H_v = self.IWT(self.fake_H_v)
        # self.fake_H = self.var_L[:,2,:,:,:]
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0][2].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if self.fake_H_u is not None:
            out_dict['SR_U'] = self.fake_H_u.detach()[0].float().cpu()
            out_dict['SR_V'] = self.fake_H_v.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD)
            if isinstance(self.netD, nn.DataParallel) or isinstance(self.netD,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD.__class__.__name__,
                                                 self.netD.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
                    logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
            print('G loaded')
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, self.opt['path']['strict_load'])
            print('D loaded')

    def save(self, iter_step):
        if not self.train_opt['only_G']:
            self.save_network(self.netG, 'G', iter_step)
            self.save_network(self.netD, 'D', iter_step)
        else:
            self.save_network(self.netG, self.opt['network_G']['which_model_G'], iter_step, self.opt['path']['pretrain_model_G'])
