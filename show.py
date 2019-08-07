import numpy as np
from matplotlib import pylab as plt
import cv2
import time
import glob
from utils import *
import os

RAW_IMAGE_DIRECTORY = '../imx390_hdr24/raw_images_day/*raw'
TARGET_DIRECTORY = "day"

def downsample(A):
    h, w = A.shape
    new_h = h // 2
    new_w = w // 2
    
    R = A[::2, ::2]
    B = A[1::2, 1::2]
    G = (A[1::2, ::2] + A[::2, 1::2]) / 2

    return np.dstack((R, G, B))


def show(filename, fileID):

    # Get Raw Data
    img = merge_raw(filename, 540, 960)
    scale = np.max(img)

    s = downsample(img)
    
    # Tone-Mapping
    tonemap_night = cv2.createTonemapReinhard(1, -4, 0.7, 0.7)
    ldrh = tonemap_night.process((np.float32(s / scale)))

    # Denoising
    ldrh = cv2.fastNlMeansDenoisingColored((ldrh * 255).astype(np.uint8), searchWindowSize=15, templateWindowSize=3, h=1) / 255

    # Adjust White Balance
    ldrh = grey_world(ldrh)
    output = ldrh

    # Bi-linear Demosaicing
    sd = debayer(img)

    # Tone-Mapping
    tonemap_night = cv2.createTonemapReinhard(1, -4, 0.7, 0.7)
    ldrh = tonemap_night.process((np.float32(sd / scale)))

    # Adjust White Balance
    ldrh = grey_world(ldrh)

    output = np.concatenate((cv2.resize(output,(output.shape[1]*2,output.shape[0]*2)),ldrh),axis=0)

    cv2.imwrite('{}/output_{}.png'.format(TARGET_DIRECTORY, fileID), output[...,::-1]*255)
    

if __name__ == '__main__':
    if not os.path.isdir(TARGET_DIRECTORY):
        e = os.system("mkdir {}".format(TARGET_DIRECTORY))

    filelist = sorted(glob.glob(RAW_IMAGE_DIRECTORY))
    for fileID, filename in enumerate(filelist):
        print (filename)
        try:
            show(filename, fileID)
        except Exception as e:
            print (e)
            continue
