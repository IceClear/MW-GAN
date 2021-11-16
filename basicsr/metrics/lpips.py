import cv2
import numpy as np

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.utils import img2tensor
from torchvision.transforms.functional import normalize
try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


@METRIC_REGISTRY.register()
def calculate_lpips(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float32) / 255.
    img2 = img2.astype(np.float32) / 255.

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    # -------------------------------------------------------------------------
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]

    img1, img2 = img2tensor([img1, img2], bgr2rgb=True, float32=True)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    # norm to [-1, 1]
    normalize(img1, mean, std, inplace=True)
    normalize(img2, mean, std, inplace=True)

    # calculate lpips
    lpips_val = loss_fn_vgg(img1.unsqueeze(0).cuda(), img2.unsqueeze(0).cuda())
    del img1
    del img2
    del loss_fn_vgg

    return lpips_val
