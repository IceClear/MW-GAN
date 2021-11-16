import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel

@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    # def load_network(self, net, load_path, strict=True, param_key=None):
    #     """Load network from old model.
    #     Args:
    #         load_path (str): The path of networks to be loaded.
    #         net (nn.Module): Network.
    #         strict (bool): Whether strictly loaded.
    #     """
    #     load_net = torch.load(load_path)
    #     # ======== modified areas =======================
    #     new_dict = {k:v for k,v in load_net.items() if k in net.state_dict().keys()}
    #     diff_dict = {k[:-5]:v for k,v in load_net.items() if '_orig' in k}
    #     new_dict.update(diff_dict)
    #     net.load_state_dict(new_dict, strict=strict)
    #     logger = get_root_logger()
    #     logger.info(f'Loaded {net.__class__.__name__} model from {load_path}.')

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

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            img_name = val_data['folder'][0]
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
                enhanced_frame_h1 = torch.cat([fake_list[0],fake_list[2]],3)
                enhanced_frame_h2 = torch.cat([fake_list[1],fake_list[3]],3)
                sr_tensor = torch.cat([enhanced_frame_h1,enhanced_frame_h2],4)
            else:
                self.feed_data(val_data)
                self.test()
                visuals = self.get_current_visuals()
                sr_tensor = visuals['result']

            sr_img = tensor2img([sr_tensor])
            lq_img = tensor2img([val_data['lq']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

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
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

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
                        self.metric_results[name] += calculate_metric(metric_data_sr, opt_) - calculate_metric(metric_data_lq, opt_)
                    else:
                        metric_data = dict(img1=sr_img, img2=gt_img)
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
