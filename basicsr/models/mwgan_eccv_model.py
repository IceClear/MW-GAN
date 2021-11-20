import torch
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel
from .sr_model import SRModel
import copy
from basicsr.archs import build_network
from basicsr.losses import build_loss

from basicsr.utils import get_root_logger, imwrite, tensor2img, tensor2img_fast, mkdirs
from basicsr.metrics import calculate_metric
from tqdm import tqdm
import os
import os.path as osp
import time
import numpy as np

@MODEL_REGISTRY.register()
class MWGANModel_ECCV(SRGANModel):
    """ESRGAN model for single image super-resolution."""

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()
        self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('motion_opt'):
            self.cri_motion = build_loss(train_opt['motion_opt']).to(self.device)
        else:
            self.cri_motion = None

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def list_detach(self, input_list):
        output_list = []
        for i in range(len(input_list)):
            output_list.append(input_list[i].detach())
        return output_list

    def list_mean(self, list_a, list_b):
        output_list = []
        for i in range(len(list_a)):
            output_list.append(list_a[i]-torch.mean(list_b[i]))
        return output_list

    def list_loss(self, list_a):
        loss = 0
        for i in range(len(list_a)):
            loss += torch.mean(list_a[i])
        return loss

    def multigan_loss(self, input_a, input_b, target_is_real, is_disc):
        if isinstance(input_a,list):
            loss_list = []
            for i in range(len(input_a)):
                temp_loss = self.cri_gan(
                    input_a[i] - torch.mean(input_b[i]),
                    target_is_real=target_is_real,
                    is_disc=is_disc)
                loss_list.append(temp_loss)
            return sum(loss_list)
        else:
            return self.cri_gan(
                input_a - torch.mean(input_b),
                target_is_real=target_is_real,
                is_disc=is_disc)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, _, _ = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, _, _ = self.net_g(self.lq)
            self.net_g.train()

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output, align_frame1, align_frame2 = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            if self.cri_motion:
                l_g_align1 = self.cri_motion(align_frame1, self.gt)
                l_g_total += l_g_align1
                l_g_align2 = self.cri_motion(align_frame2, self.gt)
                l_g_total += l_g_align2
                loss_dict['l_g_motion'] = l_g_align1 + l_g_align2

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss (relativistic gan)
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # gan loss (relativistic gan)

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.

        # real
        real_d_pred = self.net_d(self.gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = self.list_loss(self.list_detach(real_d_pred))
        loss_dict['out_d_fake'] = self.list_loss(self.list_detach(fake_d_pred))

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')
        dpsnr_dic = {}
        for idx, val_data in enumerate(dataloader):
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            img_name = val_data['folder'][0]

            if idx == 0:
                cur_name = img_name
                last_name = cur_name
                idx_begin = 0
            else:
                if last_name != cur_name:
                    idx_begin = idx
                    last_name = cur_name

            start_time = time.time()
            if val_data['lq'].size()[-1] > 1280:
                width = val_data['lq'].size()[-1]
                height = val_data['lq'].size()[-2]
                fake_list = []
                temp_data = {}
                temp_data['gt'] = val_data['gt']
                temp_data['folder'] = val_data['folder']
                for height_start in [0, int(height / 2)]:
                    for width_start in [0, int(width / 2)]:
                        temp_data['lq'] = val_data['lq'][:,:,:,height_start: (height_start + int(height / 2)), width_start: (width_start + int(width / 2))]
                        self.feed_data(temp_data)
                        self.test()
                        visuals = self.get_current_visuals()
                        fake_list.append(visuals['result'])
                enhanced_frame_h1 = torch.cat([fake_list[0],fake_list[2]],2)
                enhanced_frame_h2 = torch.cat([fake_list[1],fake_list[3]],2)
                sr_tensor = torch.cat([enhanced_frame_h1,enhanced_frame_h2],3)
            else:
                self.feed_data(val_data)
                self.test()
                visuals = self.get_current_visuals()
                sr_tensor = visuals['result']

            sr_img = tensor2img([sr_tensor])
            lq_img = tensor2img([val_data['lq'][:, val_data['lq'].size(1)//2]])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            logger = get_root_logger()
            end_time = time.time()
            # logger.info(f'\t # {img_name}: {end_time-start_time:.4f}\n')

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, img_name,
                                                 f'{img_name}_{idx-idx_begin}.png')
                imwrite(sr_img, save_img_path)

            cur_name = img_name

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name == 'lpips':
                        width = sr_img.shape[1]
                        height = sr_img.shape[0]
                        lpips_list = 0
                        for height_start in [int(height*2 / 4)]:
                            for width_start in [int(width*2 / 4)]:
                                metric_data = dict(img1=sr_img[height_start: (height_start + int(height / 4)), width_start: (width_start + int(width / 4)), :],
                                                   img2=gt_img[height_start: (height_start + int(height / 4)), width_start: (width_start + int(width / 4)), :])
                                lpips_part = calculate_metric(metric_data, opt_).data.cpu().numpy()[0][0][0][0]
                                lpips_list += lpips_part
                        lpips_list = float(lpips_list / 16)
                        self.metric_results[name] += lpips_list
                    elif name == 'dpsnr':
                        metric_data_lq = dict(img1=lq_img, img2=gt_img)
                        metric_data_sr = dict(img1=sr_img, img2=gt_img)
                        dpsnr = calculate_metric(metric_data_sr, opt_) - calculate_metric(metric_data_lq, opt_)
                        self.metric_results[name] += dpsnr
                    else:
                        metric_data = dict(img1=sr_img, img2=gt_img)
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

            if 'dpsnr' in self.metric_results:
                if img_name in dpsnr_dic:
                    dpsnr_dic[img_name].append(dpsnr)
                else:
                    dpsnr_dic[img_name] = []
                    dpsnr_dic[img_name].append(dpsnr)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        if len(dpsnr_dic)>0:
            mean_list = []
            for key_name in dpsnr_dic:
                cur_list = np.array(dpsnr_dic[key_name])
                logger.info(f'\t # {key_name}: {np.mean(cur_list):.4f}\n')
                mean_list.append(np.mean(cur_list))

            mean_list = np.array(mean_list)
            logger.info(f'\t # average: {np.mean(mean_list):.4f}\n')

@MODEL_REGISTRY.register()
class MWGANModel_ECCV_PSNR(SRModel):
    """ESRGAN model for single image super-resolution."""

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('motion_opt'):
            self.cri_motion = build_loss(train_opt['motion_opt']).to(self.device)
        else:
            self.cri_motion = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, _, _ = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, _, _ = self.net_g(self.lq)
            self.net_g.train()

    def optimize_parameters(self, current_iter):

        self.optimizer_g.zero_grad()
        self.output, align_frame1, align_frame2= self.net_g(self.lq)

        B, C, H, W = self.output.size()

        l_g_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output, self.gt)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

        if self.cri_motion:
            l_g_align1 = self.cri_motion(align_frame1, self.gt)
            if current_iter > 50000:
                l_g_total += l_g_align1*0.1
            elif current_iter > 100000:
                l_g_total += l_g_align1*0.01
            else:
                l_g_total += l_g_align1
            l_g_align2 = self.cri_motion(align_frame2, self.gt)
            if current_iter > 50000:
                l_g_total += l_g_align2*0.1
            elif current_iter > 100000:
                l_g_total += l_g_align2*0.01
            else:
                l_g_total += l_g_align2
            loss_dict['l_g_motion'] = l_g_align1 + l_g_align2

        l_g_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')
        dpsnr_dic = {}
        for idx, val_data in enumerate(dataloader):
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            img_name = val_data['folder'][0]

            if idx == 0:
                cur_name = img_name
                last_name = cur_name
                idx_begin = 0
            else:
                if last_name != cur_name:
                    idx_begin = idx
                    last_name = cur_name

            start_time = time.time()
            if val_data['lq'].size()[-1] > 1280:
                width = val_data['lq'].size()[-1]
                height = val_data['lq'].size()[-2]
                fake_list = []
                temp_data = {}
                temp_data['gt'] = val_data['gt']
                temp_data['folder'] = val_data['folder']
                for height_start in [0, int(height / 2)]:
                    for width_start in [0, int(width / 2)]:
                        temp_data['lq'] = val_data['lq'][:,:,:,height_start: (height_start + int(height / 2)), width_start: (width_start + int(width / 2))]
                        self.feed_data(temp_data)
                        self.test()
                        visuals = self.get_current_visuals()
                        fake_list.append(visuals['result'])
                enhanced_frame_h1 = torch.cat([fake_list[0],fake_list[2]],2)
                enhanced_frame_h2 = torch.cat([fake_list[1],fake_list[3]],2)
                sr_tensor = torch.cat([enhanced_frame_h1,enhanced_frame_h2],3)
            else:
                self.feed_data(val_data)
                self.test()
                visuals = self.get_current_visuals()
                sr_tensor = visuals['result']

            sr_img = tensor2img([sr_tensor])
            lq_img = tensor2img([val_data['lq'][:, val_data['lq'].size(1)//2]])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            logger = get_root_logger()
            end_time = time.time()
            # logger.info(f'\t # {img_name}: {end_time-start_time:.4f}\n')

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, img_name,
                                                 f'{img_name}_{idx-idx_begin}.png')
                imwrite(sr_img, save_img_path)

            cur_name = img_name

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name == 'lpips':
                        width = sr_img.shape[1]
                        height = sr_img.shape[0]
                        lpips_list = 0
                        for height_start in [int(height*2 / 4)]:
                            for width_start in [int(width*2 / 4)]:
                                metric_data = dict(img1=sr_img[height_start: (height_start + int(height / 4)), width_start: (width_start + int(width / 4)), :],
                                                   img2=gt_img[height_start: (height_start + int(height / 4)), width_start: (width_start + int(width / 4)), :])
                                lpips_part = calculate_metric(metric_data, opt_).data.cpu().numpy()[0][0][0][0]
                                lpips_list += lpips_part
                        lpips_list = float(lpips_list / 16)
                        self.metric_results[name] += lpips_list
                    elif name == 'dpsnr':
                        metric_data_lq = dict(img1=lq_img, img2=gt_img)
                        metric_data_sr = dict(img1=sr_img, img2=gt_img)
                        dpsnr = calculate_metric(metric_data_sr, opt_) - calculate_metric(metric_data_lq, opt_)
                        self.metric_results[name] += dpsnr
                    else:
                        metric_data = dict(img1=sr_img, img2=gt_img)
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

            if 'dpsnr' in self.metric_results:
                if img_name in dpsnr_dic:
                    dpsnr_dic[img_name].append(dpsnr)
                else:
                    dpsnr_dic[img_name] = []
                    dpsnr_dic[img_name].append(dpsnr)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        if len(dpsnr_dic)>0:
            mean_list = []
            for key_name in dpsnr_dic:
                cur_list = np.array(dpsnr_dic[key_name])
                logger.info(f'\t # {key_name}: {np.mean(cur_list):.4f}\n')
                mean_list.append(np.mean(cur_list))

            mean_list = np.array(mean_list)
            logger.info(f'\t # average: {np.mean(mean_list):.4f}\n')
