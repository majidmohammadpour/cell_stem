
######################## -*- coding: utf-8 -*-


import os
import sys
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import cv2
import shutil


import Models               # StemCell Model, Dataset
import Util as util         # Utility functions



os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



# =============================================================================================
# Variables
SHOW = True                                     # 이미지 새로 저장 여부
IMG_SIZE = 512                                  # IMG_SIZE
TEST_PATH = '../dataset2/'                      # Test대상 이미지 폴더
MODEL_PATH = '../models/'                       # Detection Model 저장 폴더
SAVE_PATH = '../results/test_20200807/'         # Inference 결과 저장할 폴더
util.mkdir(SAVE_PATH)



D_LAYERS = 2                                    # Number of Dense Layers
FILTER = 8
KERNEL = 3
batch_size = 20
learn_rate = 1e-4
Device = ''                                     # GPU 번호, 구분하지 않으면 ''



LABEL = ['1', '2', '3']                         # 3classes model
LABEL_C1 = ['1', '9']                           # 2classes model - for Class1
LABEL_C2 = ['2', '9']                           # 2classes model - for Class2
LABEL_C3 = ['3', '9']                           # 2classes model - for Class3
DAYS_DIC = {
    'day0': 0, 'day0.5': 1, 'day1': 2, 'day1.5': 3, 'day2': 4,
    'day2.5': 5, 'day3': 6, 'day3.5': 7, 'day4': 8, 'day4.5': 9,
    'day5': 10, 'day5.5': 11, 'day6': 12, 'day6.5': 13, 'day7': 14,
    'day7.5': 15, 'day8': 16, 'day8.5': 17, 'day9': 18, 'day9.5': 19, 'day10': 19
}



# =============================================================================================
# Load Models
model_dic = {}

PROJECT = 'StemCell_3classes-Days'      # 3classes 구별 모델(5CL: accuracy 높음)
g = tf.Graph()
with g.as_default():
    model = Models.Model_StemCell_Days(g, Device, MODEL_PATH + '{}_CL{}.ckpt'.format(PROJECT, 5),
        IMG_SIZE, 3, 5, D_LAYERS, learn_rate, cost_wgt=[], reverse=True, filter=FILTER, kernel=KERNEL)
    model_dic.update({'3classes-5CL': model})


PROJECT = 'StemCell_3classes-Days'      # 3classes 구별 모델(4CL: CAM 이미지가 더 보기 좋음)
g = tf.Graph()
with g.as_default():
    model = Models.Model_StemCell_Days(g, Device, MODEL_PATH + '{}_CL{}.ckpt'.format(PROJECT, 4),
        IMG_SIZE, 3, 4, D_LAYERS, learn_rate, cost_wgt=[], reverse=True, filter=FILTER, kernel=KERNEL)
    model_dic.update({'3classes-4CL': model})


PROJECT = 'StemCell_2classes-C1-Days'   # 2classes 구별 모델(C1 or NOT)
g = tf.Graph()
with g.as_default():
    model = Models.Model_StemCell_Days(g, Device, MODEL_PATH + '{}_CL{}.ckpt'.format(PROJECT, 5),
        IMG_SIZE, 2, 5, D_LAYERS, learn_rate, cost_wgt=[], reverse=True, filter=FILTER, kernel=KERNEL)
    model_dic.update({'2classes-C1': model})


PROJECT = 'StemCell_2classes-C2-Days'   # 2classes 구별 모델(C2 or NOT)
g = tf.Graph()
with g.as_default():
    model = Models.Model_StemCell_Days(g, Device, MODEL_PATH + '{}_CL{}.ckpt'.format(PROJECT, 5),
        IMG_SIZE, 2, 5, D_LAYERS, learn_rate, cost_wgt=[], reverse=True, filter=FILTER, kernel=KERNEL)
    model_dic.update({'2classes-C2': model})


PROJECT = 'StemCell_2classes-C3-Days'   # 2classes 구별 모델(C3 or NOT)
g = tf.Graph()
with g.as_default():
    model = Models.Model_StemCell_Days(g, Device, MODEL_PATH + '{}_CL{}.ckpt'.format(PROJECT, 5),
        IMG_SIZE, 2, 5, D_LAYERS, learn_rate, cost_wgt=[], reverse=True, filter=FILTER, kernel=KERNEL)
    model_dic.update({'2classes-C3': model})



# =============================================================================================
# Test
pred_results = []
for dirname, subdirs, files in os.walk(TEST_PATH):
    # -------------------------------------------------------------
    # 예외처리
    if '96well' in dirname:
        continue
    dirname = dirname.replace('\\', '/')
    if 'TIMELINE' in dirname:
        continue
    for filename in files:
        if '.jpg' not in filename.lower():
            continue
        # ------------------------------------------------------------
        try:
            # Day One-hot
            day_start_index = dirname.index('day')
            day = dirname[day_start_index:].split('/')[0]
            print('day:', day)
            day_one_hot = [0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0]
            day_index = DAYS_DIC.get(day)
            if day_index is None:       # Day Info가 없는 데이터
                continue
            day_one_hot[day_index] = 1
            day_one_hot = np.array(day_one_hot).reshape(-1, 20)
            #
            # ------------------------------------------------------------
            # Preprocessing : image value sum값이 가장 작은(가장 어두운: Cell) crop 저장
            img_src = cv2.imread(dirname + '/' + filename)[:, :, 0]
            img_crop = img_src[:IMG_SIZE, :IMG_SIZE]
            interval = 10
            cnt_rows = int((img_src.shape[0] - IMG_SIZE) / interval)
            cnt_cols = int((img_src.shape[1] - IMG_SIZE) / interval)
            for row in range(cnt_rows):
                for col in range(cnt_cols):
                    cur_box = img_src[
                              row * interval: row * interval + IMG_SIZE,
                              col * interval: col * interval + IMG_SIZE
                              ]
                    if np.sum(cur_box) < np.sum(img_crop):
                        img_crop = cur_box
            #
            # -------------------------------------------------------------
            # Inference
            predictions = []
            cam_patches = []
            for key, model in model_dic.items():
                l_val, p_val = model.test(img_crop.reshape(1, IMG_SIZE, IMG_SIZE, 1), day_one_hot)
                if '3classes' in key:
                    pred = LABEL[p_val[0]]
                elif '2classes-C1' in key:
                    pred = LABEL_C1[p_val[0]]
                elif '2classes-C2' in key:
                    pred = LABEL_C2[p_val[0]]
                elif '2classes-C3' in key:
                    pred = LABEL_C3[p_val[0]]
                if pred in LABEL and pred not in predictions:   # '1' or '2', '3'
                    predictions.append(pred)
                # --------------------------------------------------------------
                # CAM : SHOW is True인 경우만 저장
                if SHOW:
                    if key == '3classes-5CL':   # 3classes: 4CL만 이용
                        continue
                    cam_val, cam_p = model.get_cam(img_crop.reshape(1, IMG_SIZE, IMG_SIZE, 1), day_one_hot)
                    cam_vis = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())), cam_val))
                    cam_vis = np.array(cam_vis).reshape(IMG_SIZE, IMG_SIZE)
                    cam_patch = np.zeros([IMG_SIZE, IMG_SIZE])
                    for dx in range(IMG_SIZE):
                        for dy in range(IMG_SIZE):
                            if cam_vis[dx][dy] > 0:
                                cam_patch[dx][dy] = cam_vis[dx][dy]
                    cam_patches.append(cam_patch)
            # ---------------------------------------------------------
            # Save Prediction Value
            pred_str = ''
            for pred in predictions:
                if pred_str == '':
                    pred_str = pred
                else:
                    pred_str = '{} or {}'.format(pred_str, pred)
            #
            print(dirname + '_' + filename, day, pred_str)
            pred_results.append([dirname, filename, day, pred_str])
            # ---------------------------------------------------------
            if SHOW:
                FIG_SIZE = (16, 8)
                pt = plt.figure(figsize=FIG_SIZE)
                #
                # Original
                pt = plt.subplot(1, 2, 1)
                pt = plt.title('Origin', fontsize=9)
                pt = plt.imshow(img_src, cmap='gray')
                pt = plt.axis('off')
                #
                # Prediction
                cam_patch = np.max(cam_patches, axis=0)
                pt = plt.subplot(1, 2, 2)
                pt = plt.title('Pred:{}'.format(pred_str), fontsize=9)
                pt = plt.imshow(img_crop, cmap='gray')
                pt = plt.imshow(cam_patch, cmap=plt.cm.jet, alpha=0.4, interpolation='nearest', vmin=0, vmax=1)
                pt = plt.axis('off')
                #
                save_file = SAVE_PATH + 'P{}_{}_{}'.format(
                    pred_str, dirname.replace('/', '_').replace('../dataset/testset/', ''), filename)
                plt.savefig(save_file)
                plt.close()
        except Exception as ex:
            print(str(ex))




# =============================================================================================
# Save Results
pd_results = pd.DataFrame(pred_results)
pd_results.columns = ['dirname', 'filename', 'day', 'prediction']
pd_results.to_csv('../results/test_20200807.csv', index=False)






# nohup python Inference_Ensemble_ADD.py   output.log 2&1 &

