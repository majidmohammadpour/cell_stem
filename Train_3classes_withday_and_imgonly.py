
######################## -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import os
import sys
import random
import copy
import matplotlib.pyplot as plt


import tensorflow as tf
import cv2
import skimage
#from imgaug import augmenters as iaa

import Models
import Util as util



# ===============================================================================
# Variables
IMG_SIZE = 512                                  # IMG_SIZE
DATA_TYPE = 'prep_data2'                         # 데이터 속성 대표하는 임의의 값
FILE_PATH = '../dataset/{}/'.format(DATA_TYPE)  # Data 폴더
IMG_PATH_1 = FILE_PATH + 'train/1_refine/'             # class1 이미지
IMG_PATH_2 = FILE_PATH + 'train/2_refine/'             # class2 이미지
IMG_PATH_3 = FILE_PATH + 'train/3_refine/'             # class3 이미지
MODEL_PATH = '../models/'                       # Model을 저장할 폴더
util.mkdir(MODEL_PATH)
SAVE_PATH = '../results/'
util.mkdir(SAVE_PATH)



NUM_CLASSES = 3                                 # class 1, 2, 3
#C_LAYERS = 5                                    # Number of Convolution Layers (Img + Day Info)
C_LAYERS = 4                                    # Number of Convolution Layers (Img Only)
D_LAYERS = 2                                    # Number of Dense Layers
FILTER = 8
KERNEL = 3
batch_size = 20
Device = ''



# ===============================================================================================
# Loading Data
# train_valid list를 저장해 놓아야 끊어서 train/valdation 할 때 동일한 데이터 적용 가능 (random shuffle 할 경우)
SHUFFLE = True
VALID_PER = 0.1



filename = '../dataset/train_valid_dic_Refine.data'
try:
    train_valid_dic = util.load_data(filename)
except:
    train_valid_dic = {}
    # ------------------------------------------
    # [Class_1]
    files_1 = []
    for file in os.listdir(IMG_PATH_1):
        file_name = file[:-4]
        files_1.append(file_name)
    #
    # [Class_2]
    files_2 = []
    for file in os.listdir(IMG_PATH_2):
        file_name = file[:-4]
        files_2.append(file_name)
    #
    # [Class_3]
    files_3 = []
    for file in os.listdir(IMG_PATH_3):
        file_name = file[:-4]
        files_3.append(file_name)
    #
    print(len(files_1), len(files_2), len(files_3))
    # ------------------------------------------
    if SHUFFLE:
        random.shuffle(files_1)
        random.shuffle(files_2)
        random.shuffle(files_3)
    #
    CNT_TRAIN = int(len(files_1) * (1 - VALID_PER))
    train_1 = files_1[: CNT_TRAIN]
    valid_1 = files_1[CNT_TRAIN: ]
    #
    CNT_TRAIN = int(len(files_2) * (1 - VALID_PER))
    train_2 = files_2[: CNT_TRAIN]
    valid_2 = files_2[CNT_TRAIN: ]
    #
    CNT_TRAIN = int(len(files_3) * (1 - VALID_PER))
    train_3 = files_3[: CNT_TRAIN]
    valid_3 = files_3[CNT_TRAIN: ]
    #
    train_valid_dic.update({
        'train_1': train_1, 'train_2': train_2, 'train_3': train_3,
        'valid_1': valid_1, 'valid_2': valid_2, 'valid_3': valid_3
    })
    util.save_data(train_valid_dic, filename)




train_1 = train_valid_dic.get('train_1')
train_2 = train_valid_dic.get('train_2')
train_3 = train_valid_dic.get('train_3')
valid_1 = train_valid_dic.get('valid_1')
valid_2 = train_valid_dic.get('valid_2')
valid_3 = train_valid_dic.get('valid_3')
print(len(train_1), len(train_2), len(train_3), len(valid_1), len(valid_2), len(valid_3))





# ===============================================================================================
# Save Validation data
valid_data_dic = {}
PROJECT_KEYS = [
    '3classes-Days',       # C_LAYERS=4
    '3classes-ImgOnly'
]
for p_key in PROJECT_KEYS:
    filename = '../dataset/valid_{}_{}.data'.format(DATA_TYPE, p_key)
    try:
        valid_data = util.load_data(filename)
        valid_data_dic.update({p_key: valid_data})
    except:
        # Valid Data
        if p_key in ['3classes-Days', '3classes-ImgOnly']:
            valid_data = Models.Dataset(
                valid_1, 0, len(valid_1), IMG_PATH_1,   # Negative-1
                valid_2, 0, len(valid_2), IMG_PATH_2,   # Nagative-2
                valid_3, 0, len(valid_3), IMG_PATH_3,  # Positive
                img_size=IMG_SIZE, balance=False, balance_size=0, augment=0
            )
        #
        print(len(valid_data.images), sum(valid_data.labels), valid_data.images.shape)
        print(sys.getsizeof(valid_data.images))  # about 1.7GB
        util.save_data(valid_data, filename)
        valid_data_dic.update({p_key: valid_data})



# ========================================================================================================
# Make Network
# 512 => 256 => 128 => 64 => 32 => 16
learn_rate = 1e-4
docs = pd.read_csv(FILE_PATH + 'cell_info_days.csv')



# Model
model_dic = {}
for p_key in PROJECT_KEYS:
    print()
    g = tf.Graph()
    with g.as_default():
        if p_key == '3classes-Days':
            PROJECT = 'StemCell_{}'.format(p_key)
            model = Models.Model_StemCell_Days(
                g, Device, MODEL_PATH + '{}_CL{}.ckpt'.format(PROJECT, C_LAYERS),
                IMG_SIZE, NUM_CLASSES, C_LAYERS, D_LAYERS, learn_rate, cost_wgt=[],
                reverse=True, filter=FILTER, kernel=KERNEL
            )
        if p_key == '3classes-ImgOnly':
            PROJECT = 'StemCell_{}'.format(p_key)
            model = Models.Model_StemCell(
                g, Device, MODEL_PATH + '{}_CL{}.ckpt'.format(PROJECT, C_LAYERS),
                IMG_SIZE, NUM_CLASSES, C_LAYERS, D_LAYERS, learn_rate, cost_wgt=[],
                reverse=True, filter=FILTER, kernel=KERNEL
            )
        model_dic.update({p_key: model})




# ===============================================================================================================
# Training
num_epochs = 10000
START_ACC = 0.5
START_COST = 2
best_accs = {}
best_costs = {}
for p_key in PROJECT_KEYS:
    best_accs.update({p_key: START_ACC})
    best_costs.update({p_key: START_COST})



LOAD_SIZE_TRAIN = 500
LOAD_SIZE_VALID = 60
for e in range(num_epochs):
    for p_key, model in model_dic.items():
        print()
        print(e, p_key)
        best_acc = best_accs.get(p_key)
        best_cost = best_costs.get(p_key)
        # -----------------------------------------------------------------------
        train_start_1 = random.randint(0, len(train_1) - LOAD_SIZE_TRAIN - 1)
        train_start_2 = random.randint(0, len(train_2) - LOAD_SIZE_TRAIN - 1)
        train_start_3 = random.randint(0, len(train_3) - LOAD_SIZE_TRAIN - 1)
        #
        valid_start_1 = random.randint(0, len(valid_1) - LOAD_SIZE_VALID - 1)
        valid_start_2 = random.randint(0, len(valid_2) - LOAD_SIZE_VALID - 1)
        valid_start_3 = random.randint(0, len(valid_3) - LOAD_SIZE_VALID - 1)
        # -------------------------------------------------------------------------------------
        # Train Data
        if p_key == '3classes-Days':
            # Train Data
            train_data = Models.Dataset(
                train_1, train_start_1, train_start_1 + LOAD_SIZE_TRAIN, IMG_PATH_1,
                train_2, train_start_2, train_start_2 + LOAD_SIZE_TRAIN, IMG_PATH_2,
                train_3, train_start_3, train_start_3 + LOAD_SIZE_TRAIN, IMG_PATH_3,
                img_size=IMG_SIZE, balance=True, balance_size=LOAD_SIZE_TRAIN, augment=10
            )
            print(train_start_1, train_start_2, train_start_3,
                  len(train_data.images), sum(train_data.labels), train_data.images.shape)
            # Valid Data
            if best_acc <= 0.8:
                valid_data = Models.Dataset(
                    valid_1, valid_start_1, valid_start_1 + LOAD_SIZE_VALID, IMG_PATH_1,
                    valid_2, valid_start_2, valid_start_2 + LOAD_SIZE_VALID, IMG_PATH_2,
                    valid_3, valid_start_3, valid_start_3 + LOAD_SIZE_VALID, IMG_PATH_3,
                    img_size=IMG_SIZE, balance=True, balance_size=LOAD_SIZE_VALID, augment=0
                )
                print(valid_start_1, valid_start_2, valid_start_3,
                      len(valid_data.images), sum(valid_data.labels), valid_data.images.shape)
            else:
                valid_data = valid_data_dic.get(p_key)
            #
            my_dataset = {'train': train_data, 'val': valid_data}
            model, best_acc, best_cost = Models.train_model_with_days(model, best_acc, best_cost,
                                                                      my_dataset, batch_size, docs)
        # ------------------------------------------------------------------------------------------
        if p_key == '3classes-ImgOnly':
            # Train Data
            train_data = Models.Dataset(
                train_1, train_start_1, train_start_1 + LOAD_SIZE_TRAIN, IMG_PATH_1,
                train_2, train_start_2, train_start_2 + LOAD_SIZE_TRAIN, IMG_PATH_2,
                train_3, train_start_3, train_start_3 + LOAD_SIZE_TRAIN, IMG_PATH_3,
                img_size=IMG_SIZE, balance=True, balance_size=LOAD_SIZE_TRAIN, augment=10
            )
            print(train_start_1, train_start_2, train_start_3,
                  len(train_data.images), sum(train_data.labels), train_data.images.shape)
            # Valid Data
            if best_acc <= 0.8:
                valid_data = Models.Dataset(
                    valid_1, valid_start_1, valid_start_1 + LOAD_SIZE_VALID, IMG_PATH_1,
                    valid_2, valid_start_2, valid_start_2 + LOAD_SIZE_VALID, IMG_PATH_2,
                    valid_3, valid_start_3, valid_start_3 + LOAD_SIZE_VALID, IMG_PATH_3,
                    img_size=IMG_SIZE, balance=True, balance_size=LOAD_SIZE_VALID, augment=0
                )
                print(valid_start_1, valid_start_2, valid_start_3,
                      len(valid_data.images), sum(valid_data.labels), valid_data.images.shape)
            else:
                valid_data = valid_data_dic.get(p_key)
            #
            my_dataset = {'train': train_data, 'val': valid_data}
            model, best_acc, best_cost = Models.train_model(model, best_acc, best_cost, my_dataset, batch_size)
        #
        # ----------------------------------------------------------------------------------------------
        model_dic.update({p_key: model})
        best_accs.update({p_key: best_acc})
        best_costs.update({p_key: best_cost})




# nohup python Train_Multi_models_3classes.py   output.log 2&1 &

