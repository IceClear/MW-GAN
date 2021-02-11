import os
import os.path as osp
import sys
import glob
import pickle
import lmdb
import cv2
import numpy as np
import pywt
try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from utils.util import ProgressBar
except ImportError:
    pass

frame_maxnum = 300
memory_len = 2

# configurations
mode = 1  # 1 for reading all the images to memory and then writing to lmdb (more memory);
# 2 for reading several images and then writing to lmdb, loop over (less memory)
batch = 1000  # Used in mode 2. After batch images, lmdb commits.
qp = 37
###########################################
# if not lmdb_save_path.endswith('.lmdb'):
#     raise ValueError("lmdb_save_path must end with \'lmdb\'.")
# #### whether the lmdb file exist
# if osp.exists(lmdb_save_path):
#     print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
    # sys.exit(1)
info_mode = 'LQ' # GT or LQ
print('*********** current mode is: ' + info_mode + ' **************')

print('*********** current size is: ' + 'test' + ' **************')
meta_info = {'name': 'YUV_all_sub'+'_test'}

if info_mode=='LQ':
    # readpath = '/media/iceclear/yuhang/YUV_f_img_crop_'+str(qp)+'_test_rgb/*'
    # readpath = '/media/iceclear/iceking/YUV_JM_img_crop_'+str(qp)+'_rgb/*'
    readpath = '/media/iceclear/iceking/YUV_img_x265_'+str(qp)+'_rgb/*'
elif info_mode=='GT':
    # readpath = '/media/iceclear/yuhang/YUV_GT_img_crop_test_rgb/*'
    # readpath = '/media/iceclear/iceking/YUV_GT_img_crop_rgb_JM/*'
    readpath = '/media/iceclear/iceking/YUV_GT_img_x265_rgb/*'
read_list = glob.glob(readpath)

save_path='/media/iceclear/iceking/YUV_lmdb_X265_qp'+str(qp)+'/'
# save_path='/media/iceclear/iceking/YUV_lmdb_X265_qp'+str(qp)+'/'

# video_list = ['BasketballDrive1920x1080','Kimono1920x1080','ParkScene1920x1080','BQTerrace1920x1080','Cactus1920x1080']

for c in read_list:
    filename = os.path.basename(c)
    print('>>>>>>>>>>>>>> '+filename+' is starting')
    img_folder = c+'/*'
    lmdb_save_path = save_path+filename+'_'+info_mode+'.lmdb'
    # if os.path.exists(lmdb_save_path):
    #     continue

    if not os.path.exists(save_path):
        os.makedirs(save_path)



    img_list = sorted(glob.glob(img_folder),key=lambda x:(int(x.split('_')[-1][:-4])))
    frame_total = len(img_list)
    # num_sub = int(len(img_list)/frame_total)
    if frame_total>frame_maxnum:
        frame_total = frame_maxnum

    num_sub = 0
    if num_sub==0:
        num_sub = 1
        # frame_total = min(len(img_list), frame_total)
    # frame_i = 2
    # block_i = 1
    # print(img_list)
    # print(img_list[frame_i*num_sub+block_i])
    # for cur_frame_in_batch in range(2*memory_len+1):
    #     print(img_list[(frame_i+cur_frame_in_batch-memory_len)*num_sub+block_i])
    # print(s)

    if mode == 1:
        print('Read images...')
        # dataset = [np.load(v) for v in img_list]
        dataset = [cv2.imread(v) for v in img_list]
        data_size = sum([img.nbytes for img in dataset])

    elif mode == 2:
        print('Calculating the total size of images...')
        data_size = sum(os.stat(v).st_size for v in img_list)*(memory_len*2+1)
    else:
        raise ValueError('mode should be 1 or 2')

    env = lmdb.open(lmdb_save_path, map_size=data_size * 20)
    txn = env.begin(write=True)  # txn is a Transaction object
    key_l = []
    resolution_l = []
    pbar = ProgressBar(frame_total)

    for frame_i in range(memory_len,frame_total-memory_len,1):
        pbar.update('Write frame {}'.format(frame_i))
        for block_i in range(num_sub):
            base_name = 'frame_'+str(frame_i)+'_'+str(block_i)
            key = base_name.encode('ascii')
            data = []
            if info_mode == 'LQ':
                for cur_frame_in_batch in range(2*memory_len+1):
                    cur_data = dataset[(frame_i+cur_frame_in_batch-memory_len)*num_sub+block_i]
                    data.append(cur_data)

            elif info_mode == 'GT':
                cur_data = dataset[frame_i*num_sub+block_i]
                data.append(cur_data)

            data = np.array(data)

            if data.ndim == 3:
                D, H, W = data.shape
                C = 1
            else:
                D, H, W, C = data.shape
            txn.put(key, data)
            key_l.append(base_name)
            resolution_l.append('{:d}_{:d}_{:d}_{:d}'.format(D, C, H, W))
            # commit in mode 2
            if mode == 2 and i % batch == 1:
                txn.commit()
                txn = env.begin(write=True)

    ### create meta information
    # check whether all the images are the same size
    same_resolution = (len(set(resolution_l)) <= 1)
    if same_resolution:
        meta_info['resolution'] = [resolution_l[0]]
        meta_info['keys'] = key_l
        print('All images have the same resolution. Simplify the meta info...')
    else:
        meta_info['resolution'] = resolution_l
        meta_info['keys'] = key_l
        print('Not all images have the same resolution. Save meta info for each image...')

    #### pickle dump
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

    txn.commit()
    env.close()

print('Finish writing lmdb.')
