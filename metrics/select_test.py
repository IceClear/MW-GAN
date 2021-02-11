from operator import add
import xlrd
import os
from glob import glob
import numpy as np

qp_name = 'QP37'
base_path = '/home/iceclear/Video-compression/PIRM2018/test_new/'
exps = glob(base_path + '*')
# num_pic = 320

method_name = ['li', 'cvpr', 'dcad', 'dncnn', 'dscnn', 'MFQE2.0', 'ori','SVEGAN']

frame_dic = {}
index_dic = {}
f = open('../codes/testInfo.txt','r')
for c in f.readlines():
    c_array = c.split()
    frame_total = int(c_array[3])
    vid_width = int(c_array[1])
    vid_height = int(c_array[2])
    test_set_name = c_array[0]+'_'+str(vid_width)+'x'+str(vid_height)
    frame_dic[test_set_name] = frame_total
    index_dic[test_set_name] = []

for i_m in range(len(method_name)):
    bk = xlrd.open_workbook(base_path+qp_name+'_'+method_name[i_m]+'.xlsx')  # 打开文件
    sh = bk.sheet_by_name("Sheet1")  # 打开sheet1
    col_num = sh.ncols
    row_num = sh.nrows
    col_list = []
    for i in range(0, col_num):
        # 获取第i行的正行的数据
        col_data = sh.col_values(i)
        col_list.append(col_data)
    name_list = col_list[0]
    mse_list = col_list[1]
    ma_list = col_list[2]  # 如1080
    niqe_list = col_list[3]  # 如1920
    last_name = None
    current_name = None
    ma_result = []
    niqe_result = []
    pi_result = []
    ma_memory = []
    niqe_memory = []
    pi_memory = []
    for i_row in range(1,row_num):
        info_temp = name_list[i_row].split('_')
        row_name = info_temp[0]+'_'+info_temp[1]
        if last_name is None:
            last_name = row_name
        frame_index = int(info_temp[-1][:-5])
        if last_name == row_name:
            if frame_index>2 and frame_index<frame_dic[row_name]-2:
                ma_memory.append(ma_list[i_row])
                niqe_memory.append(niqe_list[i_row])
                pi_memory.append(0.5*(10-ma_list[i_row]+niqe_list[i_row]))
        else:
            ma_result.append(np.mean(np.array(ma_memory)))
            niqe_result.append(np.mean(np.array(niqe_memory)))
            pi_result.append(np.mean(np.array(pi_memory)))
            ma_memory = []
            niqe_memory = []
            pi_memory = []
            ma_memory.append(ma_list[i_row])
            niqe_memory.append(niqe_list[i_row])
            pi_memory.append(0.5*(10-ma_list[i_row]+niqe_list[i_row]))
        # if i_m == 0:
        #     name = row_name
        #     for split_i in range(2,len(info_temp)-2):
        #         name += '_'
        #         name += info_temp[split_i]
        #     if frame_index>2 and frame_index<frame_dic[name]-2:
        #         index_dic[name].append(frame_index)
        last_name = row_name
    if i_m == 0:
        np.save(base_path+'test_frame.npy', index_dic)
    ma_result.append(np.mean(np.array(ma_memory)))
    niqe_result.append(np.mean(np.array(niqe_memory)))
    pi_result.append(np.mean(np.array(pi_memory)))
    ma_memory = []
    niqe_memory = []
    pi_memory = []
    print('\n****************************************************\n')
    print(method_name[i_m])
    print('Ma Result:')
    print(ma_result)
    print('NIQE Result:')
    print(niqe_result)
    print('PI Result:')
    print(pi_result)
    print('\n****************************************************\n')
