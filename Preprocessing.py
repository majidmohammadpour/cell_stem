

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


import Util as util


#-------------------------------------------------------------------------
DATA_PATH = '../dataset/moved_data/'
SAVE_PATH = '../dataset/prep_data2/'
util.mkdir(SAVE_PATH)
IMG_SIZE = 512




# -------------------------------------------------------------------------------------
for dirname, subdirs, files in os.walk(DATA_PATH):
    for filename in files:
        if filename[-4:] == '.jpg':
            dirname = dirname.replace('\\', '/')
            # -----------------------------------------------------
            # read src
            src_img = cv2.imread(dirname + '/' + filename)[:, :, 0]
            #
            # -----------------------------------------------------
            # mkdir dst path
            group = dirname.split('/')[-2]
            util.mkdir(SAVE_PATH + group)
            cell_class = dirname.split('/')[-1]
            COMPARE_PATH = SAVE_PATH + '{}/{}_compare/'.format(group, cell_class)
            util.mkdir(COMPARE_PATH)
            #
            # -----------------------------------------------------
            # [preprocessing]
            # - (IMG_SIZE, IMG_SIZE) Box Scanning => Minimum box
            # - masking (find circle -> expand(ditate) -> merge)
            # - equalize histogram
            min_box = src_img[:IMG_SIZE, :IMG_SIZE]
            interval = 10
            cnt_rows = int((src_img.shape[0] - IMG_SIZE) / interval)
            cnt_cols = int((src_img.shape[1] - IMG_SIZE) / interval)
            for row in range(cnt_rows):
                for col in range(cnt_cols):
                    cur_box = src_img[
                              row * interval: row * interval + IMG_SIZE,
                              col * interval: col * interval + IMG_SIZE
                              ]
                    if np.sum(cur_box) < np.sum(min_box):
                        min_box = cur_box
            plt.subplot(1, 2, 1)
            plt.title('Max:{}, min:{}'.format(np.max(min_box), np.min(min_box)))
            plt.imshow(min_box, cmap='gray')
            #
            ret, min_box = cv2.threshold(min_box, 150, 255, cv2.THRESH_TRUNC)
            HE_img = (min_box - np.min(min_box)) / (np.max(min_box) - np.min(min_box)) * 255
            HE_img = HE_img.astype(np.uint8)
            plt.subplot(1, 2, 2)
            plt.title('HE_Max:{}, HE_min:{}'.format(np.max(HE_img), np.min(HE_img)))
            plt.imshow(HE_img, cmap='gray')
            plt.axis('off')
            plt.savefig(COMPARE_PATH + filename[:-4] + '.png')
            plt.close()
            # ------------------------------------------------------------------------------
            DST_PATH = SAVE_PATH + '{}/{}/'.format(group, cell_class)
            util.mkdir(DST_PATH)
            img_pil = Image.fromarray(HE_img)
            img_pil.save(DST_PATH + filename)








# nohup python Preprocessing.py   output.log 2&1 &




