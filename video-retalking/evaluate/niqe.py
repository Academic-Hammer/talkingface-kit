import sys
import math
import logging
import os
from os.path import dirname
from os.path import join

import tempfile
import cv2
from skimage.transform import resize
from PIL import Image, ImageFilter
import scipy
import scipy.ndimage
import scipy.special
import scipy.io
import numpy
import numpy as np
from numpy import (amin, amax, ravel, asarray, arange, ones, newaxis,
                   transpose, iscomplexobj, uint8, issubdtype, array)

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_paths():
    reference_path = None

    # 检查参数数量
    if len(sys.argv) <= 2:
        logging.error("Usage: python niqe.py --reference <reference_file>")
        sys.exit(1)

    # 遍历参数
    for i, arg in enumerate(sys.argv):
        if arg == "--reference":
            # 检查是否有下一个参数，即文件路径
            if i + 1 < len(sys.argv):
                reference_path = sys.argv[i + 1]
                # 检查路径是否是.mp4文件
                if not reference_path.endswith((".mp4", ".avi", ".mov", ".mkv")) or not os.path.isfile(reference_path):
                    logging.error(f"The provided reference path is not a valid .mp4 file: {reference_path}")
                    sys.exit(1)
            else:
                logging.error("No file path provided after --reference argument.")
                sys.exit(1)

    if reference_path is None:
        logging.error("--reference arguments must be provided.")
        sys.exit(1)

    logging.info(f"Reference path: {reference_path}")
    return reference_path

def extract_frames(video_path, output_dir):
    """提取视频帧并保存到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    return frame_count, output_dir

def compute_niqe_on_video(video_path, niqe_func):
    """计算单个视频的 NIQE 平均值"""
    temp_dir = "temp_frames"
    frame_count, frame_dir = extract_frames(video_path, temp_dir)
    niqe_scores = []

    for frame_name in sorted(os.listdir(frame_dir)):
        frame_path = os.path.join(frame_dir, frame_name)
        frame = Image.open(frame_path).convert('LA')
        frame_gray = np.array(frame)[:, :, 0]  # 取灰度图像
        score = niqe_func(frame_gray)
        niqe_scores.append(score)

    # 清理临时目录
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

    # 返回该视频的平均 NIQE
    return np.mean(niqe_scores)

def batch_niqe_evaluation(video_folder, niqe_func):
    """计算文件夹中所有视频的 NIQE 并计算总体平均值"""
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]
    video_niqe_scores = []

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        logging.info(f"Processing {video_file}...")
        video_niqe = compute_niqe_on_video(video_path, niqe_func)
        video_niqe_scores.append(video_niqe)
        logging.info(f"NIQE for {video_file}: {video_niqe}")

    # 计算所有视频的总体平均 NIQE
    overall_avg_niqe = np.mean(video_niqe_scores)
    logging.info(f"Overall Average NIQE: {overall_avg_niqe}")

    return video_niqe_scores, overall_avg_niqe

#需要python3.6版本

# from __future__ import division, print_function, absolute_import

# Functions which need the PIL


if not hasattr(Image, 'frombytes'):
    Image.frombytes = Image.fromstring

__all__ = ['fromimage', 'toimage', 'imsave', 'imread', 'bytescale',
           'imrotate', 'imresize', 'imshow', 'imfilter']


@numpy.deprecate(message="`bytescale` is deprecated in SciPy 1.0.0, "
                         "and will be removed in 1.2.0.")
def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    
    if data.dtype == uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(uint8)


@numpy.deprecate(message="`imread` is deprecated in SciPy 1.0.0, "
                         "and will be removed in 1.2.0.\n"
                         "Use ``imageio.imread`` instead.")
def imread(name, flatten=False, mode=None):
  

    im = Image.open(name)
    return fromimage(im, flatten=flatten, mode=mode)


@numpy.deprecate(message="`imsave` is deprecated in SciPy 1.0.0, "
                         "and will be removed in 1.2.0.\n"
                         "Use ``imageio.imwrite`` instead.")
def imsave(name, arr, format=None):
   
    im = toimage(arr, channel_axis=2)
    if format is None:
        im.save(name)
    else:
        im.save(name, format)
    return


@numpy.deprecate(message="`fromimage` is deprecated in SciPy 1.0.0. "
                         "and will be removed in 1.2.0.\n"
                         "Use ``np.asarray(im)`` instead.")
def fromimage(im, flatten=False, mode=None):

    if not Image.isImageType(im):
        raise TypeError("Input is not a PIL image.")

    if mode is not None:
        if mode != im.mode:
            im = im.convert(mode)
    elif im.mode == 'P':
        # Mode 'P' means there is an indexed "palette".  If we leave the mode
        # as 'P', then when we do `a = array(im)` below, `a` will be a 2-D
        # containing the indices into the palette, and not a 3-D array
        # containing the RGB or RGBA values.
        if 'transparency' in im.info:
            im = im.convert('RGBA')
        else:
            im = im.convert('RGB')

    if flatten:
        im = im.convert('F')
    elif im.mode == '1':
        # Workaround for crash in PIL. When im is 1-bit, the call array(im)
        # can cause a seg. fault, or generate garbage. See
        # https://github.com/scipy/scipy/issues/2138 and
        # https://github.com/python-pillow/Pillow/issues/350.
        #
        # This converts im from a 1-bit image to an 8-bit image.
        im = im.convert('L')

    a = array(im)
    return a


_errstr = "Mode is unknown or incompatible with input array shape."


@numpy.deprecate(message="`toimage` is deprecated in SciPy 1.0.0, "
                         "and will be removed in 1.2.0.\n"
                         "Use Pillow's ``Image.fromarray`` directly instead.")
def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
   
    data = asarray(arr)
    if iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(numpy.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(asarray(pal, dtype=uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (arange(0, 256, 1, dtype=uint8)[:, newaxis] *
                       ones((3,), dtype=uint8)[newaxis, :])
                image.putpalette(asarray(pal, dtype=uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = amin(ravel(data))
        if cmax is None:
            cmax = amax(ravel(data))
        data = (data * 1.0 - cmin) * (high - low) / (cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(numpy.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = numpy.flatnonzero(asarray(shape) == 3)[0]
        else:
            ca = numpy.flatnonzero(asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


@numpy.deprecate(message="`imrotate` is deprecated in SciPy 1.0.0, "
                         "and will be removed in 1.2.0.\n"
                         "Use ``skimage.transform.rotate`` instead.")
def imrotate(arr, angle, interp='bilinear'):
  
    arr = asarray(arr)
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    im = toimage(arr)
    im = im.rotate(angle, resample=func[interp])
    return fromimage(im)


@numpy.deprecate(message="`imshow` is deprecated in SciPy 1.0.0, "
                         "and will be removed in 1.2.0.\n"
                         "Use ``matplotlib.pyplot.imshow`` instead.")
def imshow(arr):
  
    im = toimage(arr)
    fnum, fname = tempfile.mkstemp('.png')
    try:
        im.save(fname)
    except Exception:
        raise RuntimeError("Error saving temporary image data.")

    import os
    os.close(fnum)

    cmd = os.environ.get('SCIPY_PIL_IMAGE_VIEWER', 'see')
    status = os.system("%s %s" % (cmd, fname))

    os.unlink(fname)
    if status != 0:
        raise RuntimeError('Could not execute image viewer.')


@numpy.deprecate(message="`imresize` is deprecated in SciPy 1.0.0, "
                         "and will be removed in 1.3.0.\n"
                         "Use Pillow instead: ``numpy.array(Image.fromarray(arr).resize())``.")
def imresize(arr, size, interp='bilinear', mode=None):
  
    im = toimage(arr, mode=mode)
    ts = type(size)
    if issubdtype(ts, numpy.signedinteger):
        percent = size / 100.0
        size = tuple((array(im.size) * percent).astype(int))
    elif issubdtype(type(size), numpy.floating):
        size = tuple((array(im.size) * size).astype(int))
    else:
        size = (size[1], size[0])
    func = {'nearest': 0, 'lanczos': 1, 'bilinear': 2, 'bicubic': 3, 'cubic': 3}
    imnew = im.resize(size, resample=func[interp])
    return fromimage(imnew)


@numpy.deprecate(message="`imfilter` is deprecated in SciPy 1.0.0, "
                         "and will be removed in 1.2.0.\n"
                         "Use Pillow filtering functionality directly.")
def imfilter(arr, ftype):
   
    _tdict = {'blur': ImageFilter.BLUR,
              'contour': ImageFilter.CONTOUR,
              'detail': ImageFilter.DETAIL,
              'edge_enhance': ImageFilter.EDGE_ENHANCE,
              'edge_enhance_more': ImageFilter.EDGE_ENHANCE_MORE,
              'emboss': ImageFilter.EMBOSS,
              'find_edges': ImageFilter.FIND_EDGES,
              'smooth': ImageFilter.SMOOTH,
              'smooth_more': ImageFilter.SMOOTH_MORE,
              'sharpen': ImageFilter.SHARPEN
              }

    im = toimage(arr)
    if ftype not in _tdict:
        raise ValueError("Unknown filter type.")
    return fromimage(im.filter(_tdict[ftype]))


# 下面加上自己需要的文件，也可以放在另一个文件，导入该函数
gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)

def aggd_features(imdata):
    #flatten imdata
    imdata.shape = (len(imdata.flat),)
    imdata2 = imdata*imdata
    left_data = imdata2[imdata<0]
    right_data = imdata2[imdata>=0]
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data))
    if len(right_data) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data))

    if right_mean_sqrt != 0:
      gamma_hat = left_mean_sqrt/right_mean_sqrt
    else:
      gamma_hat = np.inf
    #solve r-hat norm

    imdata2_mean = np.mean(imdata2)
    if imdata2_mean != 0:
      r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata2))
    else:
      r_hat = np.inf
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1)*(gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    #solve alpha by guessing values that minimize ro
    pos = np.argmin((prec_gammas - rhat_norm)**2);
    alpha = gamma_range[pos]

    gam1 = scipy.special.gamma(1.0/alpha)
    gam2 = scipy.special.gamma(2.0/alpha)
    gam3 = scipy.special.gamma(3.0/alpha)

    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    #mean parameter
    N = (br - bl)*(gam2 / gam1)#*aggdratio
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

def ggd_features(imdata):
    nr_gam = 1/prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq/E**2
    pos = np.argmin(np.abs(nr_gam - rho));
    return gamma_range[pos], sigma_sq

def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)


def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights

def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
      avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image)/(var_image + C), var_image, mu_image


def _niqe_extract_subband_feats(mscncoefs):
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl+br)/2.0,
            alpha1, N1, bl1, br1,  # (V)
            alpha2, N2, bl2, br2,  # (H)
            alpha3, N3, bl3, bl3,  # (D1)
            alpha4, N4, bl4, bl4,  # (D2)
    ])

def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)

def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = int(patch_size)
    patches = []
    for j in range(0, h-patch_size+1, patch_size):
        for i in range(0, w-patch_size+1, patch_size):
            patch = img[j:j+patch_size, i:i+patch_size]
            patches.append(patch)

    patches = np.array(patches)
    
    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features

def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        logging.warning("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0: 
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]


    img = img.astype(np.float32)
    new_height = int(img.shape[0] * 0.5)
    new_width = int(img.shape[1] * 0.5)
    img2 = resize(img, (new_height, new_width), order=3, mode='reflect', anti_aliasing=True)

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)


    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size/2)

    feats = np.hstack((feats_lvl1, feats_lvl2))# feats_lvl3))

    return feats

def niqe(inputImgData):

    patch_size = 96
    module_path = dirname(__file__)

    # TODO: memoize
    params = scipy.io.loadmat(join(module_path, 'data', 'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]


    M, N = inputImgData.shape

    # assert C == 1, "niqe called with videos containing %d channels. Please supply only the luminance channel" % (C,)
    assert M > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"


    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov+sample_cov)/2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score


if __name__ == "__main__":
    # 文件路径
    video_folder = get_paths() 

    logging.info('Computing NIQE calculation...')
    # 批量计算视频的 NIQE
    overall_avg = compute_niqe_on_video(video_folder, niqe)
    logging.info(f"Average NIQE: {overall_avg}")  

# python niqe.py --reference ..\reference\May.mp4