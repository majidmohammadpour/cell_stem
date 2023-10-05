
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



# ===============================================================================
# Variables
IMG_SIZE = 512                                  # IMG_SIZE
#DATA_TYPE = 'prep_data1'                         # 데이터 속성 대표하는 임의의 값
DATA_TYPE = 'prep_data2'                         # 데이터 속성 대표하는 임의의 값
FILE_PATH = '../dataset/{}/'.format(DATA_TYPE)  # Data 폴더
IMG_PATH_1 = FILE_PATH + 'train/1/'             # class1 이미지
IMG_PATH_2 = FILE_PATH + 'train/2/'             # class2 이미지
IMG_PATH_3 = FILE_PATH + 'train/3/'             # class3 이미지
MODEL_PATH = '../models/'                       # Model을 저장할 폴더
util.mkdir(MODEL_PATH)
SAVE_PATH = '../results/'
util.mkdir(SAVE_PATH)


NUM_CLASSES = 2                                 # class 1, 2, 3
C_LAYERS = 4                                    # Number of Convolution Layers
D_LAYERS = 2                                    # Number of Dense Layers
FILTER = 8
KERNEL = 3
batch_size = 20
Device = '2'                                     # GPU 번호, 구분하지 않으면 ''





# ===============================================================================================
# Loading Data
# train_valid list를 저장해 놓아야 끊어서 train/valdation 할 때 동일한 데이터 적용 가능 (random shuffle 할 경우)
filename = '../dataset/train_valid_dic.data'
train_valid_dic = util.load_data(filename)


train_1 = train_valid_dic.get('train_1')
train_2 = train_valid_dic.get('train_2')
train_3 = train_valid_dic.get('train_3')
valid_1 = train_valid_dic.get('valid_1')
valid_2 = train_valid_dic.get('valid_2')
valid_3 = train_valid_dic.get('valid_3')
print(len(train_1), len(train_2), len(train_3), len(valid_1), len(valid_2), len(valid_3))



# ===============================================================================================
# Save Validation data
filename = '../dataset/valid_{}_{}.data'.format(DATA_TYPE, '2class-C2')
try:
    valid_data_2class = util.load_data(filename)
except:
    # Valid Data : 아직 seq_all 적용 안함
    valid_data_2class = Models.Dataset_2class(
        valid_2, 0, len(valid_2), IMG_PATH_2,   # Positive
        #
        valid_1, 0, len(valid_1), IMG_PATH_1,   # Negative-1
        valid_3, 0, len(valid_3), IMG_PATH_3,   # Nagative-2
        img_size=IMG_SIZE, balance=False, balance_size=0, augment=0
    )
    print(sys.getsizeof(valid_data_2class.images))  # about 1.7GB
    util.save_data(valid_data_2class, filename)

print(len(valid_data_2class.images), sum(valid_data_2class.labels), valid_data_2class.images.shape)



# ========================================================================================================
# Make Network
# 512 => 256 => 128 => 64 => 32
# Model
learn_rate = 1e-4
PROJECT = 'StemCell_{}-{}'.format(DATA_TYPE, '2class-C2-Reverse-Days')
g = tf.Graph()
with g.as_default():
    model = Models.Model_StemCell_Days(
        g, Device, MODEL_PATH + '{}_CL{}_DL{}_b{}.ckpt'.format(PROJECT, C_LAYERS, D_LAYERS, batch_size),
        IMG_SIZE, NUM_CLASSES, C_LAYERS, D_LAYERS, learn_rate, cost_wgt=[],
        reverse=True, filter=FILTER, kernel=KERNEL
    )




# ===============================================================================================================
# Training
num_epochs = 10000
START_ACC = 0.6
START_COST = 3
best_acc = START_ACC
best_cost = START_COST



docs = pd.read_csv(FILE_PATH + 'cell_info_days.csv')



LOAD_SIZE_TRAIN = 500
LOAD_SIZE_VALID = 60
for e in range(num_epochs):
    print()
    print(e)
    # -----------------------------------------------------------------------
    # Train Data
    start_1 = random.randint(0, len(train_1) - LOAD_SIZE_TRAIN - 1)
    start_2 = random.randint(0, len(train_2) - LOAD_SIZE_TRAIN - 1)
    start_3 = random.randint(0, len(train_3) - LOAD_SIZE_TRAIN - 1)
    train_data = Models.Dataset_2class(
        train_2, start_2, start_2 + LOAD_SIZE_TRAIN, IMG_PATH_2,
        #
        train_1, start_1, start_1 + LOAD_SIZE_TRAIN, IMG_PATH_1,
        train_3, start_3, start_3 + LOAD_SIZE_TRAIN, IMG_PATH_3,
        img_size=IMG_SIZE, balance=True, balance_size=LOAD_SIZE_TRAIN, augment=5
    )
    print(len(train_data.images), sum(train_data.labels), train_data.images.shape)
    # -----------------------------------------------------------------------
    # Valid Data
    if best_acc <= 0.8:
        start_1 = random.randint(0, len(valid_1) - LOAD_SIZE_VALID - 1)
        start_2 = random.randint(0, len(valid_2) - LOAD_SIZE_VALID - 1)
        start_3 = random.randint(0, len(valid_3) - LOAD_SIZE_VALID - 1)
        valid_data = Models.Dataset_2class(
            valid_2, start_2, start_2 + LOAD_SIZE_VALID, IMG_PATH_2,
            #
            valid_1, start_1, start_1 + LOAD_SIZE_VALID, IMG_PATH_1,
            valid_3, start_3, start_3 + LOAD_SIZE_VALID, IMG_PATH_3,
            img_size=IMG_SIZE, balance=True, balance_size=LOAD_SIZE_VALID, augment=0
        )
        print(len(valid_data.images), sum(valid_data.labels), valid_data.images.shape)
    else:
        valid_data = valid_data_2class
    my_dataset = {'train': train_data, 'val': valid_data}
    model, best_acc, best_cost = Models.train_model_with_days(model, best_acc, best_cost, my_dataset, batch_size, docs)



# nohup python Train_2classes_with_Day.py   output.log 2&1 &




'''

# ===============================================================================================================
# Results with CAM
CAM_PATH = SAVE_PATH + '{}/{}/'.format(PROJECT, 'CAM')
util.mkdir(CAM_PATH)


Models.validate_model_cam(model, valid_data_total, CAM_PATH, 1000, 0, ['C2', 'Others'])



'''
