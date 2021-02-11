import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from data.data_process import *
import torch
import numpy as np
import re

def findfile(dir, name):
    num=re.findall(r'\d*\.?\d+',name)
    num_len = len(num[0])
    files = os.listdir(dir)
    if num_len<5:
        video_name = name[:-(num_len*2+1)]
        width = num[0]
    else:
        width = num[0][1:]
        video_name = name[:-(num_len*2-2)]
    for f in files:
        if video_name in f and width in f:
            return f
    return None

def switch_data(opt, filename ,logger):
    for phase, dataset_opt in opt['datasets'].items():
        dataset_opt['dataroot_GT'] = dataset_opt['datarootGT']+filename+'_GT.lmdb'
        dataset_opt['dataroot_LQ'] = dataset_opt['dataroot']+filename+'_LQ.lmdb'
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        # test_loaders.append(test_loader)

    assert test_loader is not None
    return test_loader

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
# test_loaders = []
# for phase, dataset_opt in sorted(opt['datasets'].items()):
#     test_set = create_dataset(dataset_opt)
#     test_loader = create_dataloader(test_set, dataset_opt)
#     logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
#     test_loaders.append(test_loader)

model = create_model(opt)

# f = open('./testvidyo_info.txt','r')
f = open('/media/iceclear/yangren/test_18/list.txt','r')

for c in f.readlines():
    c_array = c.split()
    test_set_name = c_array[0]
    frame_total = int(c_array[3])
    vid_width = int(c_array[1])
    vid_height = int(c_array[2])
    width_crop = 0
    height_crop = 0
    crop_size = 64
    filename = test_set_name+str(vid_width)+'x'+str(vid_height)
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    dataset_dir = osp.join(opt['path']['results_root'], filename)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['test_time'] = []
    test_results['dpsnr'] = []
    test_results['dssim'] = []
    test_results['dpsnr_rgb'] = []
    test_results['dssim_rgb'] = []

    test_loader = switch_data(opt, filename ,logger)

    yuv_path = '/media/iceclear/yuhang/RA_Rec/'
    readPath_GT = '/media/iceclear/yuhang/YUV_All/'

    # yuv_path = '/media/iceclear/yangren/test_18/vidyo/'
    # readPath_GT = '/media/iceclear/yangren/test_18/vidyo_JM32/'

    if frame_total>100:
        frame_total=100

    cur_video_lq = findfile(yuv_path,filename)
    cur_video_gt = findfile(readPath_GT,filename)

    [y_gt,u_gt,v_gt] = read_YUV420(readPath_GT + cur_video_gt,vid_height,vid_width,frame_total)
    [y_lq,u_lq,v_lq] = read_YUV420(yuv_path + cur_video_lq,vid_height,vid_width,frame_total)

    frame_height = y_lq[0].shape[0]
    frame_width = y_lq[0].shape[1]

    if frame_width%crop_size != 0 or frame_height%crop_size != 0:
        if frame_width%crop_size != 0:
            width_crop = int(frame_width%crop_size/2)//2
        if frame_height%crop_size != 0:
            height_crop = int(frame_height%crop_size/2)//2

    data_count = 0
    for data in test_loader:
        data_count += 1

        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        model.feed_data(data, need_GT=need_GT)
        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        if opt['enhance_uv']:
            data_u = []
            data_v = []
            for frame_i in range(5):

                if height_crop>0:
                    temp_u_lq = u_lq[data_count-1+frame_i][height_crop:-height_crop, :]
                    temp_v_lq = v_lq[data_count-1+frame_i][height_crop:-height_crop, :]
                else:
                    temp_u_lq = u_lq[data_count-1+frame_i]
                    temp_v_lq = v_lq[data_count-1+frame_i]
                if width_crop>0:
                    temp_u_lq = temp_u_lq[:, width_crop:-width_crop]
                    temp_v_lq = temp_v_lq[:, width_crop:-width_crop]

                data_u += [temp_u_lq/255.0]
                data_v += [temp_v_lq/255.0]
            data_u = torch.from_numpy(np.array(data_u))
            data_u = data_u.unsqueeze(0)
            data_u = data_u.unsqueeze(2)
            data_u = data_u.float()
            data_v = torch.from_numpy(np.array(data_v))
            data_v = data_v.unsqueeze(0)
            data_v = data_v.unsqueeze(2)
            data_v = data_v.float()
            test_start_time = time.time()
            model.test(load_path=osp.join(opt['path']['models']), input_u=data_u, input_v=data_v)
            test_end_time = time.time()
        else:
            test_start_time = time.time()
            model.test(load_path=osp.join(opt['path']['models']))
            test_end_time = time.time()
        visuals = model.get_current_visuals(need_GT=need_GT)

        test_time = test_end_time - test_start_time
        test_results['test_time'].append(test_time)
        sr_y = util.tensor2img(visuals['SR'])  # uint8
        lr_y = util.tensor2img(visuals['LQ'])

        if sr_y.ndim == 2:
            if opt['enhance_uv']:
                sr_u = util.tensor2img(visuals['SR_U'])
                sr_v = util.tensor2img(visuals['SR_V'])
            else:
                if height_crop>0:
                    temp_u_lq = u_lq[data_count+1][height_crop:-height_crop, :]
                    temp_v_lq = v_lq[data_count+1][height_crop:-height_crop, :]
                else:
                    temp_u_lq = u_lq[data_count+1]
                    temp_v_lq = v_lq[data_count+1]
                if width_crop>0:
                    temp_u_lq = temp_u_lq[:, width_crop:-width_crop]
                    temp_v_lq = temp_v_lq[:, width_crop:-width_crop]
                sr_u = temp_u_lq
                sr_v = temp_v_lq

            sr_img = yuv2rgb(sr_y, sr_u, sr_v)
            lr_img = yuv2rgb(lr_y, sr_u, sr_v)
        else:
            sr_img = sr_y
            lr_img = lr_y

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')

        util.save_img(sr_img, save_img_path)
        # util.saveimg_yuv(dataset_dir,sr_y.shape[1],sr_y.shape[0],sr_y, sr_u, sr_v, img_name)
        lq_img_path = osp.join(dataset_dir +'/lq/', img_name + '.png')
        util.mkdir(dataset_dir +'/lq/')
        util.save_img(lr_img, lq_img_path)
        # util.saveimg_yuv(dataset_dir +'/lq/',lr_y.shape[1],lr_y.shape[0],lr_y, sr_u, sr_v, img_name)

        # calculate PSNR and SSIM
        if need_GT:
            gt_y = util.tensor2img(visuals['GT'])

            if gt_y.ndim == 2:

                if height_crop>0:
                    temp_u_gt = u_gt[data_count+1][height_crop:-height_crop, :]
                    temp_v_gt = v_gt[data_count+1][height_crop:-height_crop, :]
                else:
                    temp_u_gt = u_gt[data_count+1]
                    temp_v_gt = v_gt[data_count+1]
                if width_crop>0:
                    temp_u_gt = temp_u_gt[:, width_crop:-width_crop]
                    temp_v_gt = temp_v_gt[:, width_crop:-width_crop]

                gt_u = temp_u_gt
                gt_v = temp_v_gt

                gt_img = yuv2rgb(gt_y, gt_u, gt_v)
            else:
                gt_img = gt_y

            gt_img_path = osp.join(dataset_dir +'/gt/', img_name + '.png')
            util.mkdir(dataset_dir +'/gt/')
            util.save_img(gt_img, gt_img_path)
            # util.saveimg_yuv(dataset_dir +'/gt/',gt_y.shape[1],gt_y.shape[0],gt_y, gt_u, gt_v, img_name)

            gt_img = gt_img / 255.
            sr_img = sr_img / 255.
            lr_img = lr_img / 255.

            crop_border = opt['crop_border']

            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

            psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            psnr_ori_rgb = util.calculate_psnr(lr_img * 255, cropped_gt_img * 255)
            dpsnr_rgb = psnr - psnr_ori_rgb
            ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
            ssim_ori_rgb = util.calculate_ssim(lr_img * 255, cropped_gt_img * 255)
            dssim_rgb = ssim - ssim_ori_rgb
            test_results['psnr'].append(psnr)
            test_results['dpsnr_rgb'].append(dpsnr_rgb)
            test_results['ssim'].append(ssim)
            test_results['dssim_rgb'].append(dssim_rgb)

            if gt_img.ndim>2:
                if gt_img.shape[2] == 3:  # RGB image
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    lq_img_y = bgr2ycbcr(lr_img, only_y=True)

                    if opt['enhance_uv']:
                        sr_img_y = sr_y
                        gt_img_y = gt_y
                        lq_img_y = lr_y

                    if crop_border == 0:
                        cropped_sr_img_y = sr_img_y
                        cropped_gt_img_y = gt_img_y
                        cropped_lq_img_y = lq_img_y
                    else:
                        cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                        cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                        cropped_lq_img_y = lq_img_y[crop_border:-crop_border, crop_border:-crop_border]

                    # print(np.mean(cropped_sr_img_y))
                    # print(np.mean(cropped_gt_img_y))
                    # print(np.mean(cropped_lq_img_y))
                    # print(s)
                    psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    psnr_ori = util.calculate_psnr(cropped_lq_img_y * 255, cropped_gt_img_y * 255)

                    ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    ssim_ori = util.calculate_ssim(cropped_lq_img_y * 255, cropped_gt_img_y * 255)
                    test_results['psnr_y'].append(psnr_y)
                    test_results['ssim_y'].append(ssim_y)
                    test_results['dpsnr'].append(psnr_y-psnr_ori)
                    test_results['dssim'].append(ssim_y-ssim_ori)
                    logger.info(
                        '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; DPSNR_RGB: {:.6f} dB; DSSIM_RGB: {:.6f}; PSNR_Y: {:.6f} dB; DPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}; DSSIM_Y: {:.6f}; test_time: {:.6f}.'.
                        format(img_name, psnr, ssim, dpsnr_rgb, dssim_rgb, psnr_y, psnr_y-psnr_ori, ssim_y, ssim_y-ssim_ori, test_time))
            else:
                logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; DSSIM_Y: {:.6f};  DPSNR_RGB: {:.6f} dB; DSSIM_RGB: {:.6f}; test_time: {:.6f}.'.format(img_name, psnr, ssim, dpsnr_rgb, dssim_rgb, test_time))
        else:
            logger.info(img_name)

    if need_GT:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_dssim_rgb = sum(test_results['dssim_rgb']) / len(test_results['dssim_rgb'])
        ave_dpsnr_rgb = sum(test_results['dpsnr_rgb']) / len(test_results['dpsnr_rgb'])
        ave_time = sum(test_results['test_time']) / len(test_results['test_time'])
        logger.info(
            '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}; DPSNR_RGB: {:.6f}; DSSIM_RGB: {:.6f}; test time: {:.6f}\n'.format(
                test_set_name, ave_psnr, ave_ssim, ave_dpsnr_rgb, ave_dssim_rgb, ave_time))
        if test_results['psnr_y'] and test_results['ssim_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            ave_dssim_rgb = sum(test_results['dssim_rgb']) / len(test_results['dssim_rgb'])
            ave_dpsnr_rgb = sum(test_results['dpsnr_rgb']) / len(test_results['dpsnr_rgb'])
            ave_dpsnr = sum(test_results['dpsnr']) / len(test_results['dpsnr'])
            ave_dssim = sum(test_results['dssim']) / len(test_results['dssim'])
            logger.info(
                '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; DPSNR_RGB: {:.6f}; dPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}; DSSIM_RGB: {:.6f}; dSSIM_Y: {:.6f}\n'.
                format(ave_psnr_y, ave_dpsnr_rgb, ave_dpsnr, ave_ssim_y, ave_dssim_rgb, ave_dssim))