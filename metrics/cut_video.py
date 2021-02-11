#ffmpeg -framerate 30 -i BasketballDrill_832x480_50_%d.png Project.mp4
from ffmpy import FFmpeg
from tqdm import tqdm
from glob import glob
import os

exps = glob('/home/iceclear/Video-compression/video-PI42/*')

save_path = "/home/iceclear/Video-compression/video-PI42_refine/"

os.makedirs(save_path,
            exist_ok=True)

for exp in exps:
    os.makedirs(os.path.join(save_path,os.path.basename(exp)),exist_ok=True)
    # for folder in tqdm(folders):
    #     print(type(os.path.splitext(folder)[1])
    # print(s)
    os.makedirs(os.path.join(save_path,os.path.basename(exp)),exist_ok=True)
    video_list= glob(exp + '/*.mp4')
    for video_item in tqdm(video_list):
        ff = FFmpeg(
            inputs={video_item: None},
            outputs={
                os.path.join(save_path,os.path.basename(exp),os.path.basename(video_item)):
                '-vcodec copy -acodec copy -ss 00:00:00 -to 00:00:15 -y'
            })
        print(ff.cmd)
        ff.run()
