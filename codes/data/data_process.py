# -*- coding: utf-8 -*-
import os, sys
sys.path.append('./')
import subprocess
import imageio
import numpy as np
import subprocess
import cv2
import glob
import math
"""
Created on Thu Jan 10 10:48:00 2013
@author: Chen Ming
"""


def read_YUV420(image_path, rows, cols, numfrm):
    """
    读取YUV文件，解析为Y, U, V图像
    :param image_path: YUV图像路径
    :param rows: 给定高
    :param cols: 给定宽
    :return: 列表，[Y, U, V]
    """
    # create Y
    gray = np.zeros((rows, cols), np.uint8)
    # print(type(gray))
    # print(gray.shape)

    # create U,V
    img_U = np.zeros((int(rows / 2), int(cols / 2)), np.uint8)
    # print(type(img_U))
    # print(img_U.shape)

    img_V = np.zeros((int(rows / 2), int(cols / 2)), np.uint8)
    # print(type(img_V))
    # print(img_V.shape)
    Y = []
    U = []
    V = []
    reader=open(image_path,'rb')

    # with open(image_path, 'rb') as reader:
    for num in range(numfrm-1):
        Y_buf = reader.read(cols * rows)
        gray = np.reshape(np.frombuffer(Y_buf, dtype=np.uint8), [rows, cols])

        U_buf = reader.read(cols//2 * rows//2)
        img_U = np.reshape(np.frombuffer(U_buf, dtype=np.uint8), [rows//2, cols//2])

        V_buf = reader.read(cols//2 * rows//2)
        img_V = np.reshape(np.frombuffer(V_buf, dtype=np.uint8), [rows//2, cols//2])

        Y = Y+[gray]
        U = U+[img_U]
        V = V+[img_V]

    return [Y, U, V]

def nomalize(input):
    temp = np.maximum(0, input)
    temp = np.minimum(255, temp)
    return temp

def yuv2rgb(Y,U,V):

    enlarge_U = cv2.resize(U, (0, 0), fx=2.0, fy=2.0)
    enlarge_V = cv2.resize(V, (0, 0), fx=2.0, fy=2.0)
    # 合并YUV3通道
    img_YUV = cv2.merge([Y, enlarge_U, enlarge_V])

    dst = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)

    return dst

def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)

def createpath(path):
    while not os.path.exists(path):
        os.makedirs(path)



if __name__ == '__main__':
    fixerror = False
    qp_value = 37
    frame_maxnum = 100
    info_mode = 'x265'
    crop_size = 64
    channal_num = 3
    # add_name='_qp37_rec'
    add_name='_qp37'

    if info_mode == 'train':
        readPath_GT = '/media/iceclear/yangren/train_108/raw/'
        readPath_RAnof = '/media/iceclear/yangren/train_108/HM16.5_LDP/QP'+str(qp_value)+'/'

        if channal_num==3:
            savePath_GT = '/media/iceclear/iceking/YUV_GT_img_crop_rgb/'
            # readPath_RAnof = '/media/iceclear/yuhang/RA_Rec/'
            savePath_RAnof = '/media/iceclear/iceking/YUV_f_img_crop_'+str(qp_value)+'_rgb/'
        elif channal_num==1:
            savePath_GT = '/media/iceclear/iceking/YUV_GT_img_crop_yuv/'
            # readPath_RAnof = '/media/iceclear/yuhang/RA_Rec/'
            savePath_RAnof = '/media/iceclear/iceking/YUV_f_img_crop_'+str(qp_value)+'_yuv/'

    elif info_mode == 'test':
        readPath_GT = '/media/iceclear/yangren/test_18/raw/'
        readPath_RAnof = '/media/iceclear/yangren/test_18/HM16.5_LDP/QP'+str(qp_value)+'/'
        if channal_num==3:
            savePath_GT = '/media/iceclear/iceking/YUV_GT_img_crop_test_rgb/'
            # readPath_RAnof = '/media/iceclear/yuhang/RA_Rec/'
            savePath_RAnof = '/media/iceclear/iceking/YUV_f_img_crop_'+str(qp_value)+'_test_rgb/'
        elif channal_num==1:
            savePath_GT = '/media/iceclear/iceking/YUV_GT_img_crop_test_yuv/'
            # readPath_RAnof = '/media/iceclear/yuhang/RA_Rec/'
            savePath_RAnof = '/media/iceclear/iceking/YUV_f_img_crop_'+str(qp_value)+'_test_yuv/'

    elif info_mode == 'mos':
        readPath_GT = '/media/iceclear/yangren/train_108/raw/'
        readPath_RAnof = '/media/iceclear/yangren/train_108/HM16.5_LDP/QP'+str(qp_value)+'/'
        if channal_num==3:
            savePath_GT = '/media/iceclear/iceking/YUV_GT_img_crop_rgb/'
            # readPath_RAnof = '/media/iceclear/yuhang/RA_Rec/'
            savePath_RAnof = '/media/iceclear/iceking/YUV_f_img_crop_'+str(qp_value)+'_rgb/'

    elif info_mode == 'jm':
        readPath_GT = '/media/iceclear/yangren/test_18/raw/'
        readPath_RAnof = '/media/iceclear/yangren/test_18/JM_qp'+str(qp_value)+'/'
        if channal_num==3:
            savePath_GT = '/media/iceclear/iceking/YUV_GT_img_crop_rgb_JM/'
            # readPath_RAnof = '/media/iceclear/yuhang/RA_Rec/'
            savePath_RAnof = '/media/iceclear/iceking/YUV_JM_img_crop_'+str(qp_value)+'_rgb/'

    elif info_mode == 'vidyo':
        readPath_GT = '/media/iceclear/yangren/test_18/vidyo/'
        readPath_RAnof = '/media/iceclear/yangren/test_18/vidyo_JM'+str(qp_value)+'/'
        if channal_num==3:
            savePath_GT = '/media/iceclear/iceking/YUV_GT_img_vidyo_rgb_265/'
            # readPath_RAnof = '/media/iceclear/yuhang/RA_Rec/'
            savePath_RAnof = '/media/iceclear/iceking/YUV_265_img_vidyo_'+str(qp_value)+'_rgb/'

    elif info_mode == 'x265':
        readPath_GT = '/media/iceclear/yangren/test_18/raw/'
        readPath_RAnof = '/media/iceclear/yangren/test_18/x265_qp'+str(qp_value)+'/'
        if channal_num==3:
            savePath_GT = '/media/iceclear/iceking/YUV_GT_img_x265_rgb/'
            # readPath_RAnof = '/media/iceclear/yuhang/RA_Rec/'
            savePath_RAnof = '/media/iceclear/iceking/YUV_img_x265_'+str(qp_value)+'_rgb/'

    errorfile = []
    errorwith = []
    errorheight = []
    errorframe = []

    video_list = glob.glob(readPath_GT+"*.yuv")
    # video_list = ['/media/iceclear/yangren/test_18/raw/KristenAndSara_1280x720_600.yuv']
    for c in video_list:
        c_array = os.path.basename(c).split('_')
        filename = c_array[0]+c_array[1]
        height_x_width = c_array[1]
        print('>>>>>>>>>>>>>> '+filename+' is starting')
        psnr_list = []
        oripsnr_list = []

        frame_per_second = 30
        frame_total_name = int(c_array[2][:-4])
        frame_total = int(c_array[2][:-4])
        vid_width = int(height_x_width[0:len(height_x_width)//2])
        vid_height = int(height_x_width[len(height_x_width)//2+1:])

        # if vid_width != 1920:
        #     continue

        width_crop = 0
        height_crop = 0

        if info_mode != 'train':
            if vid_width%crop_size != 0 or vid_height%crop_size != 0:
                if vid_width%crop_size != 0:
                    width_crop = int(vid_width%crop_size/2)
                if vid_height%crop_size != 0:
                    height_crop = int(vid_height%crop_size/2)

        createpath(savePath_GT+filename+'/')
        createpath(savePath_RAnof+filename+'/')

        try:
            [y_gt,u_gt,v_gt] = read_YUV420(c,vid_height,vid_width,frame_total)

            # [y_lq,u_lq,v_lq] = read_YUV420(readPath_RAnof + 'rec_RA_' + filename + '_qp'+str(qp_value)+'_nf'+ str(frame_total) +'.yuv',
            # vid_height,vid_width,frame_total)
            gt_name = os.path.basename(c).split('.')[0]
            [y_lq,u_lq,v_lq] = read_YUV420(readPath_RAnof + gt_name + add_name + '.yuv',vid_height,vid_width,frame_total)


            if frame_total>frame_maxnum and info_mode!='test':
                frame_total = frame_maxnum+1
            # [Y,U,V] = read_YUV420(readPath + filename+'_resi.yuv',vid_height,vid_width,frame_total)
            # print((Y[0]==Y[5]).all())
            # subprocess.call(["./360tools_conv", "-i", '/media/s/YuhangSong_1/env/ff/vr_new/'+dataset[i]+'.yuv', "-w",str(W), "-h", str(H), "-x", str(1), "-o", '/media/s/Iceclear/CMP/'+dataset[i]+'.yuv', "-l", str(3840), "-m", str(2880), "-y", str(1), "-f", str(3)])
            for index in range(frame_total-1):
                # psnr_frame = computePSNR(y_gt[index],y_lq[index])
                # psnr_list.append(psnr_frame)
                rgb_lq = yuv2rgb(y_lq[index],u_lq[index],v_lq[index])
                rgb_gt = yuv2rgb(y_gt[index],u_gt[index],v_gt[index])
                if height_crop>0:
                    temp_y_lq = y_lq[index][height_crop:-height_crop, :]
                    temp_y_gt = y_gt[index][height_crop:-height_crop, :]
                    rgb_lq = rgb_lq[height_crop:-height_crop, :, :]
                    rgb_gt = rgb_gt[height_crop:-height_crop, :, :]
                else:
                    temp_y_lq = y_lq[index]
                    temp_y_gt = y_gt[index]
                    rgb_lq = rgb_lq
                    rgb_gt = rgb_gt
                if width_crop>0:
                    temp_y_lq = temp_y_lq[:, width_crop:-width_crop]
                    temp_y_gt = temp_y_gt[:, width_crop:-width_crop]
                    rgb_lq = rgb_lq[:, width_crop:-width_crop, :]
                    rgb_gt = rgb_gt[:, width_crop:-width_crop, :]

                psnr_frame = computePSNR(rgb_gt,rgb_lq)
                print(psnr_frame)

                if not os.path.exists(savePath_GT+filename+'/'+filename+'_'+str(index)+'.png'):
                    if channal_num==3:
                        cv2.imwrite(savePath_GT+filename+'/'+filename+'_'+str(index)+'.png',rgb_gt)
                        # np.save(savePath_GT+filename+'/'+filename+'_'+str(index)+'.npy',rgb_gt)
                    elif channal_num==1:
                        cv2.imwrite(savePath_GT+filename+'/'+filename+'_'+str(index)+'.png',temp_y_gt)

                if not os.path.exists(savePath_RAnof+filename+'/'+filename+'_'+str(index)+'.png'):
                    if channal_num==3:
                        cv2.imwrite(savePath_RAnof+filename+'/'+filename+'_'+str(index)+'.png',rgb_lq)
                        # np.save(savePath_RAnof+filename+'/'+filename+'_'+str(index)+'.npy',rgb_lq)
                    elif channal_num==1:
                        cv2.imwrite(savePath_RAnof+filename+'/'+filename+'_'+str(index)+'.png',temp_y_lq)


            # psnr_list = np.array(psnr_list)
            print('>>>>>>>>>>>>>> '+filename+' is finished')
            # print('Average PSNR is: '+str(np.mean(psnr_list)))

        except Exception as e:
            errorfile.append(filename)
            errorwith.append(vid_width)
            errorheight.append(vid_height)
            errorframe.append(frame_total_name)
            print('ERROR: '+filename)

        print('All finish, error list is: ')
        print(errorfile)
