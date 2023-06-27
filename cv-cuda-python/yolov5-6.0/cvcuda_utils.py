import glob
import os

import torch
import cvcuda  # CV-CUDA library
import torchnvjpeg  # encode-decode library
from cvcuda import Border 

from multiprocessing.pool import ThreadPool, Pool
from pathlib import Path

import cv2
import numpy as np
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        
        # Read image
        self.count += 1

        # ======================================
        image_data = open(path, 'rb').read()
        decoder = torchnvjpeg.Decoder()
        img0 = decoder.decode(image_data)  # [H, W, C]  [R, G, B] is not used in classification 
        # ======================================

        assert img0 is not None, 'Image Not Found ' + path
        print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize 
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]   # [im, ratio, (dw, dh)][0] = im   ***

        # Convert, scale and expand batch dim   # ***
        # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def __len__(self):
        return self.nf  # number of files
    

def letterbox(im, new_shape=(640, 640), color=[114.0, 114.0, 114.0], auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]  im.shape:[height, width, channels]

    # =======================compute pad and others from original code ===========================
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    # ==============================================================================

    # ==========================================================================
    im = torch.unsqueeze(im, dim=0).permute(0, 3, 1, 2)   # from NWHC to NCHW

    im = im.permute(0, 2, 3, 1).contiguous()  # from NCHW to NHWC
    im = cvcuda.as_tensor(im, "NHWC")   # 加载张量到GPU，通道排序设定
    # tag: Preprocess
    """
    Preprocessing includes the following sequence of operations.
    Resize -> DataType Convert(U8->F32) -> Normalize
    (Apply mean and std deviation) -> Interleaved to Planar
    """
    
    if shape[::-1] != new_unpad:  # resize
        # Resize to the input network dimensions
        im = cvcuda.resize(
            im,
            (
                1,              # batch size
                new_unpad[1],   # target height   # attention!!
                new_unpad[0],   # target width
                3,              # target channels
            ),
            cvcuda.Interp.LINEAR,
        )
        # im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)    # ***

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # width corresponding to four sides
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border rectangle

    # top, bottom, left, right = torch.as_tensor(top).cuda(), torch.as_tensor(bottom).cuda(), torch.as_tensor(left).cuda(), torch.as_tensor(right).cuda()

    im = cvcuda.copymakeborder(im, border_mode=Border.CONSTANT, border_value=color, top=top, bottom=bottom, left=left, right=right)  # add border rectangle

    # Convert to the data type and range of values needed by the input layer
    # i.e uint8->float. A Scale is applied to normalize the values in the
    # range 0-1
    im = cvcuda.convertto(im, np.float32, scale=1 / 255)
    
    # The final stage in the preprocess pipeline includes converting the RGB
    # buffer into a planar buffer
    im = cvcuda.reformat(im, "NCHW")

    return im, ratio, (dw, dh)

