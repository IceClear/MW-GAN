import os
import os.path as osp
import sys
from multiprocessing import Pool
import numpy as np
import cv2
import shutil
from glob import glob
import math

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def main():

    gt_path = '/media/iceclear/IceKing2/AAAI2020/YUV_GT_img_crop/ParkScene_1920x1080_24/*.png'
    test_path = '/media/iceclear/IceKing2/AAAI2020/MFQEv2.0-master/out_37/NP/ParkScene_1920x1080_24/*.png'
    gt_list = glob(gt_path)
    test_list = glob(test_path)

    psnr_dict = {}
    ssim_dict = {}
    resi_dict = {}

    for index in range(len(gt_list)):
        print(index)
        gt_img = cv2.imread(gt_list[index], cv2.IMREAD_UNCHANGED)
        test_img = cv2.imread(test_list[index], cv2.IMREAD_UNCHANGED)
        cur_psnr = calculate_psnr(gt_img,test_img)
        # cur_ssim = calculate_ssim(gt_img,test_img)
        cur_resi = np.mean((gt_img-test_img)*(gt_img-test_img))
        psnr_dict[str(index)] = cur_psnr
        # ssim_dict[str(index)] = cur_ssim
        resi_dict[str(index)] = cur_resi
        # print(cur_resi)
    resi_dict = dict(sorted(resi_dict.items(), key=lambda e: e[1], reverse=True))
    for key, value in resi_dict.items():
        print(key)
        print(value)
        print(test_list[int(key)])
    # print(resi_dict)
    # for
    # gt_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)



if __name__ == '__main__':
    main()
