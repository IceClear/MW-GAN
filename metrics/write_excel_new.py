#!/usr/bin/env python
# coding=utf-8
from xlwt import *
from glob import glob
import os
import random
#需要xlwt库的支持
#import xlwt

#指定file以utf-8的格式打开
#指定打开的文件名

video_file = glob('/home/iceclear/Video-compression/video-PI42/*')
file_path = 'C:/Users/IRC207/Desktop/video-PI42'

method_list=['LQ']
file = Workbook(encoding = 'utf-8')
table = file.add_sheet('data')
data = ['File name']
for exp in video_file:
    video_list= glob(exp + '/*.mp4')
    for video_item in video_list:
        if os.path.basename(video_item)[-11:-4]=='416x240':
            addition_method=['MW-GAN','CVRGAN','MFQE2.0','SEVGAN']
            random.shuffle(addition_method)
            method_list+=addition_method
            for i in range(len(method_list)):
                data.append(os.path.join(file_path,method_list[i],os.path.basename(video_item)))
            method_list=['LQ']
    break
for i,p in enumerate(data):
    table.write(i,0,p)
for i,p in enumerate(data):
    if i>0:
        table.write(i,1,os.path.join(file_path,'LQ',os.path.basename(p)))
    else:
        table.write(i,1,os.path.basename(p))
file.save('TEST_1.xls')

method_list=['LQ']
file = Workbook(encoding = 'utf-8')
table = file.add_sheet('data')
data = ['File name']
for exp in video_file:
    video_list= glob(exp + '/*.mp4')
    for video_item in video_list:
        if os.path.basename(video_item)[-11:-4]=='832x480':
            addition_method=['MW-GAN','CVRGAN','MFQE2.0','SEVGAN']
            random.shuffle(addition_method)
            method_list+=addition_method
            for i in range(len(method_list)):
                data.append(os.path.join(file_path,method_list[i],os.path.basename(video_item)))
            method_list=['LQ']
    break
for i,p in enumerate(data):
    table.write(i,0,p)
for i,p in enumerate(data):
    if i>0:
        table.write(i,1,os.path.join(file_path,'LQ',os.path.basename(p)))
    else:
        table.write(i,1,os.path.basename(p))
file.save('TEST_2.xls')

method_list=['LQ']
file = Workbook(encoding = 'utf-8')
table = file.add_sheet('data')
data = ['File name']
for exp in video_file:
    video_list= glob(exp + '/*.mp4')
    for video_item in video_list:
        if os.path.basename(video_item)[-12:-4]=='1280x720':
            addition_method=['MW-GAN','CVRGAN','MFQE2.0','SEVGAN']
            random.shuffle(addition_method)
            method_list+=addition_method
            for i in range(len(method_list)):
                data.append(os.path.join(file_path,method_list[i],os.path.basename(video_item)))
            method_list=['LQ']
    break
for i,p in enumerate(data):
    table.write(i,0,p)
for i,p in enumerate(data):
    if i>0:
        table.write(i,1,os.path.join(file_path,'LQ',os.path.basename(p)))
    else:
        table.write(i,1,os.path.basename(p))
file.save('TEST_3.xls')

method_list=['LQ']
file = Workbook(encoding = 'utf-8')
table = file.add_sheet('data')
data = ['File name']
for exp in video_file:
    video_list= glob(exp + '/*.mp4')
    for video_item in video_list:
        if os.path.basename(video_item)[-13:-4]=='1920x1080':
            addition_method=['MW-GAN','CVRGAN','MFQE2.0','SEVGAN']
            random.shuffle(addition_method)
            method_list+=addition_method
            for i in range(len(method_list)):
                data.append(os.path.join(file_path,method_list[i],os.path.basename(video_item)))
            method_list=['LQ']
    break
for i,p in enumerate(data):
    table.write(i,0,p)
for i,p in enumerate(data):
    if i>0:
        table.write(i,1,os.path.join(file_path,'LQ',os.path.basename(p)))
    else:
        table.write(i,1,os.path.basename(p))
file.save('TEST_4.xls')

method_list=['LQ']
file = Workbook(encoding = 'utf-8')
table = file.add_sheet('data')
data = ['File name']
for exp in video_file:
    video_list= glob(exp + '/*.mp4')
    for video_item in video_list:
        if os.path.basename(video_item)[-13:-4]=='2560x1600':
            addition_method=['MW-GAN','CVRGAN','MFQE2.0','SEVGAN']
            random.shuffle(addition_method)
            method_list+=addition_method
            for i in range(len(method_list)):
                data.append(os.path.join(file_path,method_list[i],os.path.basename(video_item)))
            method_list=['LQ']
    break
for i,p in enumerate(data):
    table.write(i,0,p)
for i,p in enumerate(data):
    if i>0:
        table.write(i,1,os.path.join(file_path,'LQ',os.path.basename(p)))
    else:
        table.write(i,1,os.path.basename(p))
file.save('TEST_5.xls')
# data = {}
# #字典数据
#
# ldata = []
# num = [a for a in data]
# #for循环指定取出key值存入num��?
# num.sort()
# #字典数据取出后无需，需要先排序
#
# for x in num:
# #for循环将data字典中的键和值分批的保存在ldata��?
#   t = [int(x)]
#   for a in data[x]:
#     t.append(a)
#   ldata.append(t)
#
# for i,p in enumerate(ldata):
# #将数据写入文��?i是enumerate()函数返回的序号数
#   for j,q in enumerate(p):
#     # print i,j,q
#     table.write(i,j,q)
