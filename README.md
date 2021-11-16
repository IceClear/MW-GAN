# MW-GAN

This repo is the official code for the following papers:

* [**MW-GAN+ for Perceptual Quality Enhancement on Compressed Video.**]()
[*Jianyi Wang*](https://iceclear.github.io/resume/2021/04/06/Resume.html),
[*Mai Xu (Corresponding)*](http://shi.buaa.edu.cn/MaiXu/zh_CN/index.htm),
[*Xin Deng*](http://shi.buaa.edu.cn/XinDeng/zh_CN/index/49459/list/index.htm),
[*Liquan Shen*](https://scholar.google.com/citations?user=EUEEtlYAAAAJ&hl=zh-CN),
[*Yuhang Song*](http://www.cs.ox.ac.uk/people/yuhang.song/).

Published on [**IEEE Transactions on Circuits and Systems for Video Technology**](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=76) in 2021.
By [MC2 Lab](http://buaamc2.net/) @ [Beihang University](http://ev.buaa.edu.cn/).

* [**Multi-level Wavelet-based Generative Adversarial Network for Perceptual Quality Enhancement of Compressed Video.**](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_24)
[*Jianyi Wang*](https://iceclear.github.io/resume/2021/04/06/Resume.html),
[*Xin Deng*](http://shi.buaa.edu.cn/XinDeng/zh_CN/index/49459/list/index.htm),
[*Mai Xu*](http://shi.buaa.edu.cn/MaiXu/zh_CN/index.htm),
[*Congyong Chen*](),
[*Yuhang Song*](http://www.cs.ox.ac.uk/people/yuhang.song/).

Published on [**16TH EUROPEAN CONFERENCE ON COMPUTER VISION**](https://eccv2020.eu/) in 2020.
By [MC2 Lab](http://buaamc2.net/) @ [Beihang University](http://ev.buaa.edu.cn/).

## Visual results on JCT-VC

Compressed video (QP=42)      |  Ours
:-------------------------:|:-------------------------:
![](https://github.com/IceClear/MW-GAN/blob/master/figure/basketball-lq.gif)  |  ![](https://github.com/IceClear/MW-GAN/blob/master/figure/basketball-our.gif)
:-------------------------:|:-------------------------:
![](https://github.com/IceClear/MW-GAN/blob/master/figure/racehorse-lq.gif)  |  ![](https://github.com/IceClear/MW-GAN/blob/master/figure/racehorse-our.gif)

## Dependencies and Installation
- This repo is completely built based on [BasicSR](https://github.com/xinntao/BasicSR). You need to install follow [Install from a local clone](https://github.com/xinntao/BasicSR/blob/master/INSTALL.md)

## Dataset Preparation
Generally, we directly read cropped images from folders.
- Run [data_process.py](https://github.com/IceClear/MW-GAN/blob/master/scripts/data_preparation/data_process.py) to extract frames from videos.
- This repo should also support LMDB format for faster IO speed as [BasicSR](https://github.com/xinntao/BasicSR). Not tested yet.

## Get Started
The same as [BasicSR](https://github.com/xinntao/BasicSR), you can see [here](https://github.com/xinntao/BasicSR/blob/master/docs/TrainTest.md) for details.

:star: *MWGAN+ Train:*

- **MWGAN+ PSNR Model:** `CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/MWGAN/train_MWGAN_PSNR.yml`
- **MWGAN+ GAN Model:** `CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/MWGAN/train_MWGAN_Percep.yml`
- **Tradeoff Model:** `CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/MWGAN/train_MWGAN_Tradeoff.yml`

:star: *MWGAN Train:*

- **MWGAN PSNR Model:** `CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/MWGAN/train_MWGAN_ECCV_PSNR.yml`
- **MWGAN GAN Model:** `CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/MWGAN/train_MWGAN_ECCV_Percep.yml`

:star: *Test:*

- **Test example:** `CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/MWGAN/test_MWGAN_Tradeoff.yml`

## Pre-train model
Here the models we provide are trained on QP37 in RGB space.

:star: *MWGAN+ Model:*

- [MWGAN+ PSNR Model](https://drive.google.com/file/d/172drsGyZoRFZdSGOfvGsRg9ALTatrbaK/view?usp=sharing)
- [MWGAN+ GAN Model]()
- [Tradeoff Model](https://drive.google.com/file/d/19LMZI4HwwqEGrYyGoEtN9JMEAthkZZV_/view?usp=sharing)

:star: *MWGAN Model:*

- [MWGAN PSNR Model]()
- [MWGAN GAN Model]()

## Acknowledgement
This repo is built mainly based on [BasicSR](https://github.com/xinntao/BasicSR). Also borrowing codes from [pacnet](https://github.com/NVlabs/pacnet) and [MWCNN_PyTorch](https://github.com/lpj0/MWCNN_PyTorch). We thank a lot for their contributions to the community.

## Citation
If you find our paper or code useful for your research, please cite:
```
@inproceedings{wang2020multi,
  title={Multi-level Wavelet-Based Generative Adversarial Network for Perceptual Quality Enhancement of Compressed Video},
  author={Wang, Jianyi and Deng, Xin and Xu, Mai and Chen, Congyong and Song, Yuhang},
  booktitle={European Conference on Computer Vision},
  pages={405--421},
  year={2020},
  organization={Springer}
}
```
