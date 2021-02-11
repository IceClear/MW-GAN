import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

import cv2


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    os.environ['RANK'] = str(0)
    os.environ['WORLD_SIZE'] = str(4)
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def switch_data(opt, filename,rank,logger,total_epochs,total_iters, train):
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            if train:
                print(filename)
                dataset_opt['dataroot_GT'] = dataset_opt['datarootGT']+filename+'_GT.lmdb'
                dataset_opt['dataroot_LQ'] = dataset_opt['dataroot']+filename+'_LQ.lmdb'
                train_set = create_dataset(dataset_opt)
                train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
                # total_epochs = int(math.ceil(total_iters / train_size))

                if opt['dist']:
                    train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                    total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
                else:
                    train_sampler = None
                train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
                if rank <= 0:
                    logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                        len(train_set), train_size))
                    logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                        total_epochs, total_iters))
            else:
                train_loader=None
        elif phase == 'val':
            if not train:
                dataset_opt['dataroot_GT'] = dataset_opt['datarootGT']+filename+'_GT.lmdb'
                dataset_opt['dataroot_LQ'] = dataset_opt['dataroot']+filename+'_LQ.lmdb'
                val_set = create_dataset(dataset_opt)
                val_loader = create_dataloader(val_set, dataset_opt, opt, None)
                if rank <= 0:
                    logger.info('Number of val images in [{:s}]: {:d}'.format(
                        dataset_opt['name'], len(val_set)))
            else:
                val_loader=None
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    # assert train_loader is not None
    return train_loader, val_loader


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))
            util.mkdirs('../experiments/pretrained_models/')

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    total_epochs = 100000
    total_iters = int(opt['train']['niter'])
    # total_epochs = total_iters
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        # if epoch % opt['train']['redata_freq'] == 0:
        #     filename = random.choice(trainlist)
        # try:
        #     [train_loader, _] = switch_data(opt,filename,rank,logger,total_epochs,total_iters,True)
        # except:
        #     print('Data error')
        #     filename = random.choice(trainlist)
        #     continue
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])
            # #### test input data
            # print(train_data['GT'].shape)
            # print(train_data['LQ'].shape)
            # LQ_data = train_data['LQ'].numpy()
            # GT_data = train_data['GT'].numpy()
            # LQ_data = LQ_data.transpose((0,1,3,4,2))
            # GT_data = GT_data.transpose((0,1,3,4,2))
            # cv2.imwrite('../01.jpg',LQ_data[0][0]*255)
            # cv2.imwrite('../02.jpg',LQ_data[0][1]*255)
            # cv2.imwrite('../03.jpg',LQ_data[0][2]*255)
            # cv2.imwrite('../04.jpg',LQ_data[0][3]*255)
            # cv2.imwrite('../05.jpg',LQ_data[0][4]*255)
            # cv2.imwrite('../GT.jpg',GT_data[0][0]*255)
            # print(s)

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                avg_psnr = 0.0
                # lq_psnr = 0.0
                dpsnr = 0.0
                idx = 0
                # filename = random.choice(vallist)
                # filename = vallist[0]
                # [_, val_loader] = switch_data(opt,filename,rank,logger,total_epochs,total_iters, False)
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'])
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8
                    lq_img = util.tensor2img(visuals['LQ'])  # uint8
                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir,
                                                 '{:s}_{:d}.png'.format(img_name, current_step))

                    if opt['train']['save_test_img']:
                        if opt['train']['save_test_img_num']>0:
                            if idx<opt['train']['save_test_img_num']:
                                util.save_img(sr_img, save_img_path)
                        else:
                            util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    crop_size = opt['scale']
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    lq_img = lq_img / 255.
                    lq_psnr = util.calculate_psnr(lq_img * 255., gt_img * 255.)
                    cur_psnr = util.calculate_psnr(sr_img * 255., gt_img * 255.)
                    avg_psnr += cur_psnr
                    dpsnr += cur_psnr-lq_psnr

                avg_psnr = avg_psnr / idx
                lq_psnr = lq_psnr / idx
                dpsnr = dpsnr / idx
                # print(avg_psnr)
                # print(lq_psnr)
                # print(s)
                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger.info('# Validation # DPSNR: {:.4e}'.format(dpsnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, dpsnr: {:.4e}'.format(
                    epoch, current_step, avg_psnr, dpsnr))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('dpsnr', dpsnr, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
