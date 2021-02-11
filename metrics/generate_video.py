#ffmpeg -framerate 30 -i BasketballDrill_832x480_50_%d.png Project.mp4
from ffmpy import FFmpeg
from tqdm import tqdm
from glob import glob
import os

exps = glob('/home/iceclear/Video-compression/SVEGAN/results/SEVGAN/')

save_path = "/home/iceclear/Video-compression/video-PI42/SEVGAN/"

os.makedirs(save_path,
            exist_ok=True)

for exp in exps:
    os.makedirs(save_path +
                os.path.basename(exp) + "/",
                exist_ok=True)

    folders = glob(exp + '/*')
    # for folder in tqdm(folders):
    #     print(type(os.path.splitext(folder)[1])
    # print(s)

    for folder in tqdm(folders):
        if os.path.splitext(folder)[1] != '.log' and os.path.splitext(folder)[1] != '.txt':
            if not os.path.exists(save_path + os.path.basename(exp) + "/" + os.path.basename(folder) + ".mp4"):
                ff = FFmpeg(
                    inputs={folder + '/' + 'frame' + '_%d'+'_0.png': None},
                    outputs={
                        save_path + os.path.basename(exp) + "/" + os.path.basename(folder) + ".mp4":
                        '-framerate 15, -b 200.0M'
                    })
                print(ff.cmd)
                ff.run()
