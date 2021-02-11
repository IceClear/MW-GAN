import imageio
import numpy as np
import cv2
import os
import time

def createpath(path):
    while not os.path.exists(path):
        os.makedirs(path)


readPath = '/media/iceclear/Ice/video_interpolation/YUV_all_pred/'
savePath = '/media/iceclear/Ice/video_interpolation/YUV_all_img/'

frame_per_video = 50
errorlist = []
# get_data
f = open('../yuvInfo.txt','r')
for c in f.readlines():
    time_b = time.time()
    c_array = c.split()
    filename = c_array[0]
    print('>>>>>>>>>>>>>> '+filename+' is starting')
    createpath(savePath+filename)

    try:
        vid_all = imageio.get_reader(readPath+filename+'_out.avi',  'ffmpeg')
        frame_per_second_all = vid_all.get_meta_data()['fps']
        frame_total_all = vid_all.get_meta_data()['nframes']
        vid_width_all = vid_all.get_meta_data()['source_size'][0]
        vid_height_all = vid_all.get_meta_data()['source_size'][1]

        vid_gt = imageio.get_reader(readPath+filename+'_ori.avi',  'ffmpeg')
        frame_per_second_gt = vid_all.get_meta_data()['fps']
        frame_total_gt = vid_all.get_meta_data()['nframes']
        vid_width_gt = vid_all.get_meta_data()['source_size'][0]
        vid_height_gt = vid_all.get_meta_data()['source_size'][1]

        vid_pred = imageio.get_reader(readPath+filename+'_pred.avi',  'ffmpeg')
        frame_per_second_pred = vid_pred.get_meta_data()['fps']
        frame_total_pred = vid_pred.get_meta_data()['nframes']
        vid_width_pred = vid_pred.get_meta_data()['source_size'][0]
        vid_height_pred = vid_pred.get_meta_data()['source_size'][1]

        if frame_total_all != frame_total_pred:
            print('error: total frame not match.')
            print(s)

        sample_interval = int((frame_total_all-1)/frame_per_video)
        for frame_i in range(frame_per_video):
            frame_former = vid_all.get_data(frame_i)
            frame_pred = vid_pred.get_data(frame_i+1)
            frame_gt = vid_gt.get_data(frame_i+1)
            frame_later = vid_all.get_data(frame_i+2)
            cv2.imwrite(savePath+filename+'/frame_'+str(frame_i)+'_01.png',frame_former)
            cv2.imwrite(savePath+filename+'/frame_'+str(frame_i)+'_02.png',frame_later)
            cv2.imwrite(savePath+filename+'/frame_'+str(frame_i)+'_GT.png',frame_gt)
            cv2.imwrite(savePath+filename+'/frame_'+str(frame_i)+'_p.png',frame_pred)
    except Exception as e:
        errorlist.append(filename)
        print('error: '+filename)

    print(time.time()-time_b)
    print('>>>>>>>>>>>>>> '+filename+' is finished')

print('All is finished')
print(errorlist)
np.save('./errorlist.npy',np.array(errorlist))
