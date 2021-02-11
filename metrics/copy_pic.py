import shutil
import glob
import tqdm
import os
import numpy as np
import re

# index_path = '/home/iceclear/Video-compression/PIRM2018/test/test_frame.npy'
# index_dic = np.load(index_path,allow_pickle=True).item()

def findfile(dir, name):
    num=re.findall(r'\d*\.?\d+',name)
    num_len = len(num[0])
    files = os.listdir(dir)
    video_name = name[:-(num_len*2+1)]
    for f in files:
        if video_name in f and num[0] in f:
            return f
    return None

exp = glob.glob("/home/iceclear/JJP/Wavelet/results/MSRGANx432-PI/*")
savepath = "/home/iceclear/JJP/Wavelet/results/test-32/"
os.makedirs(savepath,exist_ok=True)

exp=['/home/iceclear/JJP/Wavelet/results/MSRGANx432-PI/PeopleOnStreet2560x1600',
'/home/iceclear/JJP/Wavelet/results/MSRGANx432-PI/Kimono1920x1080',
'/home/iceclear/JJP/Wavelet/results/MSRGANx432-PI/RaceHorses832x480',
'/home/iceclear/JJP/Wavelet/results/MSRGANx432-PI/BasketballPass416x240']

for video in exp:
    files = glob.glob(video + "/*.png")
    # files=files[0:95]
    if len(files)>0:
        files_list = []
        interval = (len(files)-2+5)//20
        index = 2
        while index < len(files)+5:
            if os.path.exists(video + "/frame_"+str(index)+"_0.png"):
                files_list.append(video + "/frame_"+str(index)+"_0.png")
            else:
                print(video + "/frame_"+str(index)+"_0.png")
            index += interval
        # files = glob.glob(video + "/*.png")
        # # files = sorted(files)

        # files = [files[i] for i in range(len(files)) if i % 25 == 0]

        #print(files)
        print(len(files_list))
        for f in tqdm.tqdm(files_list):
            # video_name = findfile('/media/iceclear/iceking/TEST-QP42/cvpr/',video.split('/')[-1])
            shutil.copy2(f, savepath + '/'+ video.split('/')[-1]+'_'+os.path.basename(f))

    files = glob.glob(video + "/lq/*.png")
    # files=files[0:95]
    os.makedirs(savepath+"_lq",exist_ok=True)
    if len(files)>0:
        files_list = []
        interval = (len(files)-2+5)//20
        index = 2
        while index < len(files)+5:
            if os.path.exists(video + "/lq/frame_"+str(index)+"_0.png"):
                files_list.append(video + "/lq/frame_"+str(index)+"_0.png")
            else:
                print(video + "/lq/frame_"+str(index)+"_0.png")
            index += interval
        # files = glob.glob(video + "/*.png")
        # # files = sorted(files)
        #
        # files = [files[i] for i in range(len(files)) if i % 25 == 0]

        #print(files)
        print(len(files_list))
        for f in tqdm.tqdm(files_list):
            # video_name = findfile('/media/iceclear/iceking/TEST-QP42/cvpr/',video.split('/')[-1])
            shutil.copy2(f, savepath + '_lq/' + video.split('/')[-1]+'_'+os.path.basename(f))


    files = glob.glob(video + "/gt/*.png")
    # files=files[0:95]
    os.makedirs(savepath+"_gt",exist_ok=True)
    if len(files)>0:
     files_list = []
     interval = (len(files)-2+5)//20
     index = 2
     while index < len(files)+5:
         if os.path.exists(video + "/gt/frame_"+str(index)+"_0.png"):
             files_list.append(video + "/gt/frame_"+str(index)+"_0.png")
         else:
             print(video + "/gt/frame_"+str(index)+"_0.png")
         index += interval
     # files = glob.glob(video + "/*.png")
     # # files = sorted(files)
     #
     # files = [files[i] for i in range(len(files)) if i % 25 == 0]

     #print(files)
     print(len(files_list))
     for f in tqdm.tqdm(files_list):
         # video_name = findfile('/media/iceclear/iceking/TEST-QP42/cvpr/',video.split('/')[-1])
         shutil.copy2(f, savepath + '_gt/' + video.split('/')[-1]+'_'+os.path.basename(f))

# for videos in exps:
#     if not os.path.isfile(videos):
#         video_name = videos.split('/')[-1]
#         for i in range(len(index_dic[video_name])):
#             ori_name = videos+'/frame_'+str(index_dic[video_name][i])+'_0.png'
#             shutil.copy2(ori_name, savepath + os.path.basename(videos)+'_frame_'+str(index_dic[video_name][i])+'.png')
