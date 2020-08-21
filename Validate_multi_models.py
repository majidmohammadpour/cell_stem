
######################## -*- coding: utf-8 -*-


import os
import sys
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import cv2
import skimage
#from imgaug import augmenters as iaa

import Models
import Util as util



# ========================================================================================================
# Variables
IMG_SIZE = 512                                  # IMG_SIZE
DATA_TYPE = 'prep_data2'                         # 데이터 속성 대표하는 임의의 값
FILE_PATH = '../dataset/{}/'.format(DATA_TYPE)  # Data 폴더


REFINE_TYPE = 'refine'
IMG_PATH_1 = FILE_PATH + 'train/1_{}/'.format(REFINE_TYPE)             # class1 이미지
IMG_PATH_2 = FILE_PATH + 'train/2_{}/'.format(REFINE_TYPE)             # class2 이미지
IMG_PATH_3 = FILE_PATH + 'train/3_{}/'.format(REFINE_TYPE)             # class3 이미지



MODEL_PATH = '../models/'                       # Model을 저장할 폴더
util.mkdir(MODEL_PATH)
SAVE_PATH = '../results/'
util.mkdir(SAVE_PATH)


C_LAYERS = 5                                    # Number of Convolution Layers
#C_LAYERS = 4                                    # Number of Convolution Layers
#C_LAYERS = 3                                    # Number of Convolution Layers
D_LAYERS = 2                                    # Number of Dense Layers
FILTER = 8
KERNEL = 3
batch_size = 20
Device = '3'                                     # GPU 번호, 구분하지 않으면 ''


docs = pd.read_csv(FILE_PATH + 'cell_info_days.csv')




# ========================================================================================================
# [Class_1]
test_1 = []
for file in os.listdir(IMG_PATH_1):
    file_name = file[:-4]
    test_1.append(file_name)


# [Class_2]
test_2 = []
for file in os.listdir(IMG_PATH_2):
    file_name = file[:-4]
    test_2.append(file_name)


# [Class_3]
test_3 = []
for file in os.listdir(IMG_PATH_3):
    file_name = file[:-4]
    test_3.append(file_name)




# test_data
test_data = Models.Dataset(
    #[], 0, 0, IMG_PATH_1,
    test_1, 0, len(test_1), IMG_PATH_1,
    #[], 0, 0, IMG_PATH_2,
    test_2, 0, len(test_2), IMG_PATH_2,
    #[], 0, 0, IMG_PATH_3,
    test_3, 0, len(test_3), IMG_PATH_3,
    img_size=IMG_SIZE, balance=False, balance_size=0, augment=0
)
print(len(test_data.images), sum(test_data.labels), test_data.images.shape)
print(sys.getsizeof(test_data.images))  # about 1.7GB





# ========================================================================================================
PROJECT_KEYS = [
    'Refine-2class-C1-Days',
    'Refine-2class-C2-Days',
    'Refine-2class-C3-Days',
    'Refine-3class-Days',
    'Refine-3class-ImgOnly',
]


learn_rate = 1e-4
model_dic = {}
for p_key in PROJECT_KEYS:
    PROJECT = 'StemCell_Reverse_{}_{}'.format(DATA_TYPE, p_key)
    print()
    print(PROJECT)
    g = tf.Graph()
    with g.as_default():
        if '2class' in p_key:
            model = Models.Model_StemCell_Days(
                g, Device, MODEL_PATH + '{}_CL{}_DL{}_b{}.ckpt'.format(PROJECT, C_LAYERS, D_LAYERS, batch_size),
                IMG_SIZE, 2, C_LAYERS, D_LAYERS, learn_rate, cost_wgt=[],
                reverse=True, filter=FILTER, kernel=KERNEL
            )
        elif 'Refine-3class-Days' in p_key:
            model = Models.Model_StemCell_Days(
                g, Device, MODEL_PATH + '{}_CL{}_DL{}_b{}.ckpt'.format(PROJECT, C_LAYERS, D_LAYERS, batch_size),
                IMG_SIZE, 3, C_LAYERS, D_LAYERS, learn_rate, cost_wgt=[],
                reverse=True, filter=FILTER, kernel=KERNEL
            )
        elif 'Refine-3class-ImgOnly' in p_key:
            model = Models.Model_StemCell(
                g, Device, MODEL_PATH + '{}_CL{}_DL{}_b{}.ckpt'.format(PROJECT, 4, D_LAYERS, batch_size),
                IMG_SIZE, 3, C_LAYERS, D_LAYERS, learn_rate, cost_wgt=[],
                reverse=True, filter=FILTER, kernel=KERNEL
            )
        model_dic.update({p_key: model})





# ===================================================================================
# Multi-model Predict

del_list  = [
    '1_Plate3_day4.5_C05_00000.jpg', '2_day0_C07_00000.jpg', '2_day0_C10_00000.jpg', '2_day0_D09_00000.jpg',
    '2_day0_D12_00000.jpg', '2_day0_E08_00000.jpg', '2_day0_F08_00000.jpg', '2_day0_F11_00000.jpg',
    '2_day0_G05_00000.jpg', '2_day0_G12_00000.jpg', '2_day0_H11_00000.jpg', '3_CBZ_Day2_H01_00000.jpg',
    '3_plate2_day1.5_E02_00000.jpg', '3_plate2_day1.5_F02_00000.jpg', '3_plate2_day1.5_G02_00000.jpg',
    '3_plate2_day1.5_H03_00000.jpg', '3_plate2_day1_A01_00000.jpg', '3_plate2_day1_B01_00000.jpg',
    '3_plate2_day1_C01_00000.jpg', '3_plate2_day1_F01_00000.jpg', '3_plate2_day1_F02_00000.jpg',
    '3_plate2_day1_G02_00000.jpg', '3_plate2_day1_H01_00000.jpg', '3_plate2_day1_H02_00000.jpg',
    '3_plate2_day2_E02_00000.jpg', '3_plate2_day2_F02_00000.jpg', '3_plate2_day2_G02_00000.jpg'
]



pred_results = []
for i in range(len(test_data.images)):
    name = test_data.names[i]
    img = test_data.images[i]
    label = np.argmax(test_data.labels[i]) + 1
    res = [name, label]
    if name + '.jpg' in del_list:
        continue
    # -------------------------------------------------------------
    # Test model
    day_one_hot = []
    day_info = docs[docs.filename == name + '.jpg']
    #
    if len(day_info) >= 1:
        day_one_hot.append([
            day_info.get('day0').values[0], day_info.get('day0.5').values[0],
            day_info.get('day1').values[0], day_info.get('day1.5').values[0],
            day_info.get('day2').values[0], day_info.get('day2.5').values[0],
            day_info.get('day3').values[0], day_info.get('day3.5').values[0],
            day_info.get('day4').values[0], day_info.get('day4.5').values[0],
            day_info.get('day5').values[0], day_info.get('day5.5').values[0],
            day_info.get('day6').values[0], day_info.get('day6.5').values[0],
            day_info.get('day7').values[0], day_info.get('day7.5').values[0],
            day_info.get('day8').values[0], day_info.get('day8.5').values[0],
            day_info.get('day9').values[0], day_info.get('day9.5').values[0]
        ])
        #
        for key in PROJECT_KEYS:
            model = model_dic.get(key)
            if key not in 'Refine-3class-ImgOnly':
                l_val, p_val = model.test(img.reshape(1, model.IMG_SIZE, model.IMG_SIZE, 1), day_one_hot)
            else:
                l_val, p_val = model.test(img.reshape(1, model.IMG_SIZE, model.IMG_SIZE, 1))
            res.append(list(p_val)[0])
            for L in range(len(l_val[0])):
                res.append(list(l_val[0])[L])
    print(i, res)
    pred_results.append(res)




pd_results = pd.DataFrame(pred_results)
pd_results.to_csv('../results/predict_multi_model_{}_Total.csv'.format(REFINE_TYPE), index=False)





# ===================================================================================================
learn_rate = 1e-4
for p_key in PROJECT_KEYS:
    PROJECT = 'StemCell_Reverse_{}_{}'.format(DATA_TYPE, p_key)
    model = model_dic.get(p_key)
    # --------------------------------------------------------------------------------------------
    if p_key not in 'Refine-3class-ImgOnly':
        CAM_PATH = SAVE_PATH + '{}/{}_CL{}_DL{}/'.format(PROJECT, REFINE_TYPE, C_LAYERS, D_LAYERS)
        util.mkdir(CAM_PATH)
        Models.validate_cam_with_days(model, test_data, CAM_PATH, 4000, 0, ['C1', 'C2', 'C3'], docs)
    else:
        CAM_PATH = SAVE_PATH + '{}/{}_CL{}_DL{}/'.format(PROJECT, REFINE_TYPE, C_LAYERS, D_LAYERS)
        util.mkdir(CAM_PATH)
        Models.validate_cam(model, test_data, CAM_PATH, 4000, 0, ['C1', 'C2', 'C3'])


