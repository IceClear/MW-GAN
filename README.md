# MW-GAN+
This repo is the official code for [*MW-GAN+ for Perceptual Quality Enhancement on Compressed Video (In submission)*](), the improved version of our conference paper:

* [**Multi-level Wavelet-based Generative Adversarial Network for Perceptual Quality Enhancement of Compressed Video.**](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_24)
[*Jianyi Wang*](http://buaamc2.net/html/Members/jianyiwang.html),
[*Xin Deng*](http://www.commsp.ee.ic.ac.uk/~xindeng/),
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

**The code will be released in two months.**

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)).
- [0.4.0 <= PyTorch <= 1.1.0](https://pytorch.org/).
- See [requirements.txt](https://github.com/IceClear/MW-GAN/blob/master/metrics/requirements.txt) for other dependencies. You can just run: `pip install -r requirements.txt`.

## Dataset Preparation
Following [BasicSR](https://github.com/xinntao/BasicSR), we use datasets in LDMB format for faster IO speed.
- First run [data_process.py](https://github.com/IceClear/MW-GAN/blob/master/codes/data/data_process.py) to extract frames from videos.
- Then run [extract_subimgs_single.py](https://github.com/IceClear/MW-GAN/blob/master/codes/scripts/extract_subimgs_single.py) to cut frames into small pieces for training.
- Finally run [create_lmdb_one.py](https://github.com/IceClear/MW-GAN/blob/master/codes/scripts/create_lmdb_one.py) to generate lmdb for training.
- Run [create_lmdb_test.py](https://github.com/IceClear/MW-GAN/blob/master/codes/scripts/create_lmdb_test.py) to generate lmdb for testing (You need to first run [data_process.py](https://github.com/IceClear/MW-GAN/blob/master/codes/data/data_process.py) to obtain test frames from videos).

## Get Started
- Run `python train.py -opt options/train/train_MWGAN_rgb.yml` for training.
- Run `python test.py -opt options/test/test_MWGAN_rgb.yml` for testing.

## Tips
- [train_MWGAN_yuv.yml](https://github.com/IceClear/MW-GAN/blob/master/codes/options/train/train_MWGAN_yuv.yml) and [test_MWGAN_yuv.yml](https://github.com/IceClear/MW-GAN/blob/master/codes/options/test/test_MWGAN_yuv.yml) are for YUV training and testing. You can also use these files to reproduce [MW-GAN](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_24) in our conference paper. You just need to change some settings and make some modifications in [DenseMWNet_arch](https://github.com/IceClear/MW-GAN/blob/master/codes/models/modules/DenseMWNet_arch.py).

## Pre-train model
Here we provide a [model](https://drive.google.com/file/d/1F2NxoH3ynYWdbvQBopy5K5M8hrlYkJBC/view?usp=sharing) trained for QP42. For other models you can just finetune on this model.

## Acknowledgement
This repo is built mainly based on [BasicSR](https://github.com/xinntao/BasicSR). Also borrowing codes from [pacnet](https://github.com/NVlabs/pacnet), [MWCNN_PyTorch](https://github.com/lpj0/MWCNN_PyTorch) and [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity). We thank a lot for their contributions to the community.

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
