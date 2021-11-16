import cv2
import numpy as np
import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

read_lq_path = '/media/minglang/iceking/YUV_265_img_vidyo_32_rgb/'
read_gt_path = '/media/minglang/iceking/YUV_GT_img_vidyo_rgb_265/'

save_lq_path = '/media/minglang/iceking/YUV_265_img_vidyo_32_rgb_png/'
save_gt_path = '/media/minglang/iceking/YUV_GT_img_vidyo_rgb_265_png/'
mkdirs(save_lq_path)
mkdirs(save_gt_path)

lq_list = os.listdir(read_lq_path)
for file in lq_list:
    print(file)
    npy_list = os.listdir(os.path.join(read_lq_path, file))
    for npy_file in npy_list:
        lq_img = np.load(os.path.join(read_lq_path, file, npy_file))
        gt_img = np.load(os.path.join(read_gt_path, file, npy_file))
        img_name = npy_file.split('.')[0]+".png"
        mkdirs(os.path.join(save_lq_path, file))
        mkdirs(os.path.join(save_gt_path, file))
        cv2.imwrite(os.path.join(save_lq_path, file, img_name), lq_img)
        cv2.imwrite(os.path.join(save_gt_path, file, img_name), gt_img)
