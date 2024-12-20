import numpy as np
import torch
import cv2
import torch.nn.functional as F

from utils import reorder_image, to_y_channel, rgb2ycbcr_pt, img2tensor

def calculate_psnr(img, img2, crop_border=0, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)

def calculate_psnr_pt(img, img2, crop_border=0, test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).
    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: SSIM result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # valid mode for window size 11
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()

def _ssim_pth(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).
    It is called by func:`calculate_ssim_pt`.
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])


def calculate_ssim(img, img2, crop_border=0, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).
    ``Paper: Image quality assessment: From error visibility to structural similarity``
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()


def calculate_ssim_pt(img, img2, crop_border=0, test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity) (PyTorch version).
    ``Paper: Image quality assessment: From error visibility to structural similarity``
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: SSIM result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img * 255., img2 * 255.)
    return ssim


def img_scissors(img, origin_size, dest_size):  # 将img先裁剪为origin_size*origin_size，再resize为dest_size*dest_size
    from skimage import io, transform
    from skimage.util import img_as_ubyte

    height, width = img.shape[:2]
    center_y, center_x = height // 2, width // 2
    crop_size = origin_size
    half_crop = crop_size // 2
    start_x = max(center_x - half_crop, 0)
    start_y = max(center_y - half_crop, 0)
    end_x = min(center_x + half_crop, width)
    end_y = min(center_y + half_crop, height)
    cropped_img = img[start_y:end_y, start_x:end_x]
    resized_img = transform.resize(cropped_img, (dest_size, dest_size), anti_aliasing=True)
    resized_img_ubyte = img_as_ubyte(resized_img)
    return resized_img_ubyte


if __name__ == '__main__':
    params_path = 'pre-train-models/'
    '''
    PSNR 是用来衡量图像之间的差异程度的指标，基于均方误差（MSE）。PSNR值越高,图像质量越好。
    通常情况下，30 dB 以上被认为是较高质量。20-30 dB 之间为可接受的质量。低于 20 dB 则质量较差。
    
    SSIM 是一种衡量两幅图像在亮度、对比度和结构信息上的相似性的方法，SSIM值在 [0, 1] 之间，1 表示完全相同。
    0.8 以上表示较好的结构相似度。
    '''
    video_origin = cv2.VideoCapture("MP4/Jae-in.mp4")   # 在这里修改路径，改为想要计算的视频路径，video_origin即原视频
    video_result = cv2.VideoCapture("MP4/Jae-inRes.mp4")    # 在这里修改路径，改为想要计算的视频路径，video_result即hallo生成的视频
    index = 0
    PSNR = 0.0
    SSIM = 0.0
    if video_origin.isOpened() and video_result.isOpened():
        rval_origin, frame_origin = video_origin.read()  # 读取视频帧
        rval_result, frame_result = video_result.read()
    else:
        rval_origin = False
        rval_result = False

    while rval_origin and rval_result:
        print(index)
        rval_origin, frame_origin = video_origin.read()
        img_origin = img_scissors(frame_origin, 720, 512)
        rval_result, frame_result = video_result.read()
        img_result = frame_result
        if img_origin is None or img_result is None:
            print("Loop End.")
            break
        else:
            PSNR_temp = calculate_psnr(img_origin, img_result, crop_border=0)
            PSNR += PSNR_temp
            SSIM_temp = calculate_ssim(img_origin, img_result, crop_border=0)
            SSIM += SSIM_temp
            print(PSNR_temp, SSIM_temp)
            index += 1
    PSNR /= index
    SSIM /= index
    print(PSNR, SSIM) 
