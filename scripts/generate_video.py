import imageio
import subprocess
import glob
import os
import numpy as np
import cv2

fixerror = False

def remove(filename):
    try:
        os.remove(filename)
    except Exception as e:
        print('file not exist')

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

readPath = '/media/minglang/iceking/results_others/MWGAN_37_formal_abla_Dnowpt_100000/visualization/YUV_37/'
ref_path = '/media/minglang/iceking/results_new/MWGAN_37_formal/visualization/YUV_37/'
savePath = '/media/minglang/iceking/mos_37/GT/'

# readPath = '/media/minglang/iceking/compare_results_MFQE/QP_37/MFQE2.0/'
# savePath = '/media/minglang/iceking/mos_37/MFQE/'

mkdirs(savePath)
# originpath = '/media/iceclear/Ice/video_interpolation/original_high_fps_videos/'
# format_list = ['*.mp4','*.MOV','*.m4v','*.mov','*.MP4']

# file_list = os.listdir(readPath)
# for c in file_list:
#     if os.path.isdir(os.path.join(readPath, c)):
#         print(c+' is starting')
#         frame_rate = 20
#         c_split = c.split('x')
#         height = int(c_split[-1])
#         if height <= 480:
#             p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, c+'_'+'%d.png'),
#             '-c:v', 'libx265', '-crf', '0', '-vframes', str(150), os.path.join(savePath, c+'.mkv')])
#         else:
#             p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, c+'_'+'%d.png'),'-filter:v','crop=832:480',
#             '-c:v', 'libx265', '-crf', '0', '-vframes', str(150), os.path.join(savePath, c+'.mkv')])
#         p.communicate()
#         print(c+' is finished')

# file_list = os.listdir(readPath)
# for c in file_list:
#     if os.path.isdir(os.path.join(readPath, c)):
#         print(c+' is starting')
#         frame_rate = 15
#         c_split = c.split('x')
#         height = int(c_split[-1])
#         frame_num = str(len(glob.glob(os.path.join(ref_path, c, '*.png'))))
#         if height < 480:
#             current_img = cv2.imread(os.path.join(readPath, c, c+'_'+'0.png'))
#             ref_img = cv2.imread(os.path.join(ref_path, 'BasketballPass416x240', 'BasketballPass416x240_0.png'))
#             if current_img.shape[0] != ref_img.shape[0] or current_img.shape[1] != ref_img.shape[1]:
#                 resize_width = min(ref_img.shape[1], current_img.shape[1])
#                 resize_height = min(ref_img.shape[0], current_img.shape[0])
#                 p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, c+'_'+'%d.png'),'-filter:v','crop='+str(resize_width)+':'+str(resize_height),
#                 '-c:v', 'libx265', '-crf', '0', '-vframes', str(frame_num), os.path.join(savePath, c+'.mkv')])
#             else:
#                 p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, c+'_'+'%d.png'),
#                 '-c:v', 'libx265', '-crf', '0', '-vframes', str(frame_num), os.path.join(savePath, c+'.mkv')])
#         else:
#             current_img = cv2.imread(os.path.join(readPath, c, c+'_'+'0.png'))
#             ref_img = cv2.imread(os.path.join(ref_path, 'BasketballDrill832x480', 'BasketballDrill832x480_0.png'))
#             resize_width = min(ref_img.shape[1], current_img.shape[1])
#             resize_height = min(ref_img.shape[0], current_img.shape[0])
#             p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, c+'_'+'%d.png'),'-filter:v','crop='+str(resize_width)+':'+str(resize_height),
#             '-c:v', 'libx265', '-crf', '0', '-vframes', str(frame_num), os.path.join(savePath, c+'.mkv')])
#         p.communicate()
#         print(c+' is finished')


file_list = os.listdir(readPath)
for c in file_list:
    if os.path.isdir(os.path.join(readPath, c)):
        print(c+' is starting')
        frame_rate = 15
        c_split = c.split('x')
        height = int(c_split[-1])
        frame_num = str(len(glob.glob(os.path.join(ref_path, c, '*.png'))))
        if height < 480:
            current_img = cv2.imread(os.path.join(readPath, c, 'gt', c+'_'+'0.png'))
            ref_img = cv2.imread(os.path.join(ref_path, 'BasketballPass416x240', 'BasketballPass416x240_0.png'))
            if current_img.shape[0] != ref_img.shape[0] or current_img.shape[1] != ref_img.shape[1]:
                resize_width = min(ref_img.shape[1], current_img.shape[1])
                resize_height = min(ref_img.shape[0], current_img.shape[0])
                p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, 'gt', c+'_'+'%d.png'),'-filter:v','crop='+str(resize_width)+':'+str(resize_height),
                '-c:v', 'libx265', '-crf', '0', '-vframes', str(frame_num), os.path.join(savePath, c+'.mkv')])
            else:
                p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, 'gt', c+'_'+'%d.png'),
                '-c:v', 'libx265', '-crf', '0', '-vframes', str(frame_num), os.path.join(savePath, c+'.mkv')])
        else:
            current_img = cv2.imread(os.path.join(readPath, c, 'gt', c+'_'+'0.png'))
            ref_img = cv2.imread(os.path.join(ref_path, 'BasketballDrill832x480', 'BasketballDrill832x480_0.png'))
            resize_width = min(ref_img.shape[1], current_img.shape[1])
            resize_height = min(ref_img.shape[0], current_img.shape[0])
            p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, 'gt', c+'_'+'%d.png'),'-filter:v','crop='+str(resize_width)+':'+str(resize_height),
            '-c:v', 'libx265', '-crf', '0', '-vframes', str(frame_num), os.path.join(savePath, c+'.mkv')])
        p.communicate()
        print(c+' is finished')

# file_list = os.listdir(readPath)
# for c in file_list:
#     if os.path.isdir(os.path.join(readPath, c)):
#         print(c+' is starting')
#         frame_rate = 15
#         c_split = c.split('x')
#         height = int(c_split[-1])
#         frame_num = str(len(glob.glob(os.path.join(ref_path, c, '*.png'))))
#         if height < 480:
#             current_img = cv2.imread(os.path.join(readPath, c, 'frame_'+'2_0.png'))
#             ref_img = cv2.imread(os.path.join(ref_path, 'BasketballPass416x240', 'BasketballPass416x240_0.png'))
#             if current_img.shape[0] != ref_img.shape[0] or current_img.shape[1] != ref_img.shape[1]:
#                 resize_width = min(ref_img.shape[1], current_img.shape[1])
#                 resize_height = min(ref_img.shape[0], current_img.shape[0])
#                 p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, 'frame_'+'%d_0.png'),'-filter:v','crop='+str(resize_width)+':'+str(resize_height),
#                 '-c:v', 'libx265', '-crf', '0', '-vframes', str(frame_num), os.path.join(savePath, c+'.mkv')])
#             else:
#                 p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, 'frame_'+'%d_0.png'),
#                 '-c:v', 'libx265', '-crf', '0', '-vframes', str(frame_num), os.path.join(savePath, c+'.mkv')])
#         else:
#             current_img = cv2.imread(os.path.join(readPath, c, 'frame_'+'2_0.png'))
#             ref_img = cv2.imread(os.path.join(ref_path, 'BasketballDrill832x480', 'BasketballDrill832x480_0.png'))
#             resize_width = min(ref_img.shape[1], current_img.shape[1])
#             resize_height = min(ref_img.shape[0], current_img.shape[0])
#             p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, 'frame_'+'%d_0.png'),'-filter:v','crop='+str(resize_width)+':'+str(resize_height),
#             '-c:v', 'libx265', '-crf', '0', '-vframes', str(frame_num), os.path.join(savePath, c+'.mkv')])
#         p.communicate()
#         print(c+' is finished')

# file_list = os.listdir(readPath)
# for c in file_list:
#     if os.path.isdir(os.path.join(readPath, c)):
#         print(c+' is starting')
#         frame_rate = 15
#         c_split = c.split('_')[1].split('x')
#         height = int(c_split[-1])
#         save_name = c.split('_')[0]+c.split('_')[1]
#         frame_num = str(len(glob.glob(os.path.join(ref_path, save_name, '*.png'))))
#
#         if height < 480:
#             current_img = cv2.imread(os.path.join(readPath, c, 'frame_'+'2.png'))
#             ref_img = cv2.imread(os.path.join(ref_path, 'BasketballPass416x240', 'BasketballPass416x240_0.png'))
#             if current_img.shape[0] != ref_img.shape[0] or current_img.shape[1] != ref_img.shape[1]:
#                 resize_width = min(ref_img.shape[1], current_img.shape[1])
#                 resize_height = min(ref_img.shape[0], current_img.shape[0])
#                 p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, 'frame_'+'%d.png'),'-filter:v','crop='+str(resize_width)+':'+str(resize_height),
#                 '-c:v', 'libx265', '-crf', '0', '-vframes', str(frame_num), os.path.join(savePath, save_name+'.mkv')])
#             else:
#                 p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, 'frame_'+'%d.png'),
#                 '-c:v', 'libx265', '-crf', '0', '-vframes', str(frame_num), os.path.join(savePath, save_name+'.mkv')])
#         else:
#             current_img = cv2.imread(os.path.join(readPath, c, 'frame_'+'2.png'))
#             ref_img = cv2.imread(os.path.join(ref_path, 'BasketballDrill832x480', 'BasketballDrill832x480_0.png'))
#             resize_width = min(ref_img.shape[1], current_img.shape[1])
#             resize_height = min(ref_img.shape[0], current_img.shape[0])
#             p = subprocess.Popen(['ffmpeg', '-r', str(frame_rate), '-start_number', str(0), '-i', os.path.join(readPath, c, 'frame_'+'%d.png'),'-filter:v','crop='+str(resize_width)+':'+str(resize_height),
#             '-c:v', 'libx265', '-crf', '0', '-vframes', str(frame_num), os.path.join(savePath, save_name+'.mkv')])
#         p.communicate()
#         print(c+' is finished')
