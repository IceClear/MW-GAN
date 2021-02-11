import os
import os.path as osp
import sys
from multiprocessing import Pool
import numpy as np
import cv2
import shutil
import glob

try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from utils.util import ProgressBar
except ImportError:
    pass


def main():
    train_or_test = 'train'
    qp = 37
    channal_num=3
    if train_or_test == 'train':
        if channal_num == 3:
            read_lq_file = '/media/iceclear/iceking/YUV_f_img_crop_'+str(qp)+'_rgb/'
            read_gt_file = '/media/iceclear/iceking/YUV_GT_img_crop_rgb/'
        elif channal_num == 1:
            read_lq_file = '/media/iceclear/iceking/YUV_f_img_crop_'+str(qp)+'_yuv/'
            read_gt_file = '/media/iceclear/iceking/YUV_GT_img_crop_yuv/'
    elif train_or_test == 'test':
        read_lq_file = '/media/iceclear/iceking/YUV_f_img_crop_'+str(qp)+'_test/'
        read_gt_file = '/media/iceclear/iceking/YUV_GT_img_crop_test/'
    else:
        print(s)

    info_mode = 'GT' # GT or LQ

    if info_mode == 'LQ':
        read_list = glob.glob(read_lq_file+'/*')
    elif info_mode == 'GT':
        read_list = glob.glob(read_gt_file+'/*')

    # f = open('../yuvInfo.txt','r')
    n_thread = 5
    crop_sz = 224
    step = 448
    thres_sz = 22
    compression_level = 0  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.
    print('*********** current mode is: ' + info_mode + ' **************')

    """A multi-thread tool to crop sub imags."""
    for c in read_list:
        filename = os.path.basename(c)
        print('>>>>>>>>>>>>>> '+filename+' is starting')
        input_folder = c
        # if info_mode == 'LQ':
        #     input_folder = '/media/iceclear/iceking/YUV_nof_img_crop_37_Yonly/'+filename
        # elif info_mode == 'GT':
        #     input_folder = '/media/iceclear/iceking/YUV_GT_img_crop_37_Yonly/'+filename
        if train_or_test == 'train':
            if channal_num==3:
                save_folder = '/media/iceclear/iceking/YUV_'+info_mode+'_imgrgb_sub'+ str(crop_sz) +'_'+str(qp)+'/'+filename
            elif channal_num==1:
                save_folder = '/media/iceclear/iceking/YUV_'+info_mode+'_imgyuv_sub'+ str(crop_sz) +'_'+str(qp)+'/'+filename
        elif train_or_test == 'test':
            save_folder = '/media/iceclear/iceking/YUV_'+info_mode+'_imgrgb_sub'+ str(crop_sz) +'_'+str(qp)+'_test/'+filename

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            print('mkdir [{:s}] ...'.format(save_folder))
        else:
            print('Folder [{:s}] already exists. Exit...'.format(save_folder))
            # continue
            # shutil.rmtree(save_folder,True)
            # os.makedirs(save_folder)
            # sys.exit(1)

        img_list = []
        for root, _, file_list in sorted(os.walk(input_folder)):
            path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
            img_list.extend(path)

        def update(arg):
            pbar.update(arg)

        pbar = ProgressBar(len(img_list))

        pool = Pool(n_thread)
        for path in img_list:
            pool.apply_async(worker,
                             args=(path, save_folder, crop_sz, step, thres_sz, compression_level),
                             callback=update)
        pool.close()
        pool.join()
        print(filename + 'is done.')
    print('All subprocesses done.')


def worker(path, save_folder, crop_sz, step, thres_sz, compression_level):
    img_name = os.path.basename(path)
    img = cv2.imread(path)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            # var = np.var(crop_img / 255)
            # if var > 0.008:
            #     print(img_name, index_str, var)
            if not os.path.exists(os.path.join(save_folder, img_name.replace('.npy', '_s{:03d}.npy'.format(index)))):
                cv2.imwrite(
                    os.path.join(save_folder, img_name.replace('.npy', '_s{:03d}.npy'.format(index))),
                    crop_img)
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
