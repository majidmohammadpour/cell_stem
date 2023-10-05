

import os
import sys, gc
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt


import cv2
import skimage
import skimage.transform
import tensorflow as tf


import Util as util




# ==============================================================================================
def augment_images(img, move_range=128, rotate_range=180, scale_range=0.1):
    dst_img = copy.deepcopy(img)
    # -------------------------------------------------------
    # Flip
    dst_img = cv2.flip(dst_img, random.randint(0, 1))   # 0: 상하반전, 1: 좌우반전
    #
    # -------------------------------------------------------
    # Moving: 상하좌우 128픽셀 랜덤
    rows, cols = dst_img.shape[:2]
    move_x = random.randint(-move_range, move_range)
    move_y = random.randint(-move_range, move_range)
    M = np.float32([[1, 0, move_x], [0, 1, move_y]])    # Transform Matrix
    dst_img = cv2.warpAffine(dst_img, M, (cols, rows))
    #
    # -------------------------------------------------------
    # Rotation: -180~180도 랜덤 / Scale: 0.9 ~ 1.1배 랜덤
    rotate_val = random.randint(-rotate_range, rotate_range)
    scale = (1 - scale_range) + random.random() * 2 * scale_range     # (0.9 ~ 1.1)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_val, scale)
    dst_img = cv2.warpAffine(dst_img, M, (cols, rows))
    #
    return dst_img





# ==============================================================================================
class Dataset():
    def __init__(self,
                 files_c1, start_c1, end_c1, file_path_c1,  # [1, 0, 0]
                 files_c2, start_c2, end_c2, file_path_c2,  # [0, 1, 0]
                 files_c3, start_c3, end_c3, file_path_c3,  # [0, 0, 1]
                 img_size=512, balance=True, balance_size=100, augment=0):
        self.images = []
        self.labels = []
        self.names = []
        # ----------------------------------------------------------------------------------
        # Class1
        images_c1 = []
        labels_c1 = []
        names_c1 = []
        #
        for i in range(start_c1, end_c1):
            file_name = files_c1[i]
            img_file = file_path_c1 + '{}.jpg'.format(file_name)
            img = cv2.imread(img_file)
            if img is None:
                print(file_name, img, img_file)
                continue
            img = img[:, :, 0]
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            # -------------------------------------------------------
            # Basic Augmentation : Random Noise
            img = skimage.util.random_noise(img, mode='gaussian', var=0.001)  # ==> 0 ~ 1 scale로 변환됨
            # -------------------------------------------------------
            #
            # Augmentation
            if augment > 0:
                for aug in range(augment):
                    img_aug = augment_images(img)
                    images_c1.append(img_aug)
                    labels_c1.append([1, 0, 0])
                    names_c1.append(file_name)
            elif augment == 0:
                images_c1.append(img)
                labels_c1.append([1, 0, 0])
                names_c1.append(file_name)
        # ----------------------------------------------------------------------------------------
        # Class2
        images_c2 = []
        labels_c2 = []
        names_c2 = []
        #
        for i in range(start_c2, end_c2):
            file_name = files_c2[i]
            img_file = file_path_c2 + '{}.jpg'.format(file_name)
            img = cv2.imread(img_file)
            if img is None:
                print(file_name, img, img_file)
                continue
            img = img[:, :, 0]
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            # -------------------------------------------------------
            # Basic Augmentation : Random Noise
            img = skimage.util.random_noise(img, mode='gaussian', var=0.001)  # ==> 0 ~ 1 scale로 변환됨
            # -------------------------------------------------------
            #
            # Augmentation
            if augment > 0:
                for aug in range(augment):
                    img_aug = augment_images(img)
                    images_c2.append(img_aug)
                    labels_c2.append([0, 1, 0])
                    names_c2.append(file_name)
            elif augment == 0:
                images_c2.append(img)
                labels_c2.append([0, 1, 0])
                names_c2.append(file_name)
        # ----------------------------------------------------------------------------------------
        # Class3
        images_c3 = []
        labels_c3 = []
        names_c3 = []
        #
        for i in range(start_c3, end_c3):
            file_name = files_c3[i]
            img_file = file_path_c3 + '{}.jpg'.format(file_name)
            img = cv2.imread(img_file)
            if img is None:
                print(file_name, img, img_file)
                continue
            img = img[:, :, 0]
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            # -------------------------------------------------------
            # Basic Augmentation : Random Noise
            img = skimage.util.random_noise(img, mode='gaussian', var=0.001)  # ==> 0 ~ 1 scale로 변환됨
            # -------------------------------------------------------
            #
            # Augmentation
            if augment > 0:
                for aug in range(augment):
                    img_aug = augment_images(img)
                    images_c3.append(img_aug)
                    labels_c3.append([0, 0, 1])
                    names_c3.append(file_name)
            elif augment == 0:
                images_c3.append(img)
                labels_c3.append([0, 0, 1])
                names_c3.append(file_name)
        print('\nclass1:', len(images_c1), ', class2:', len(images_c2), ', class3:', len(images_c3))
        # ----------------------------------------------------------------------------------
        if balance == True:
            for i in range(balance_size):
                index = random.randint(0, len(images_c1) - 1)
                self.images.append(images_c1[index])
                self.labels.append(labels_c1[index])
                self.names.append(names_c1[index])
                #
                index = random.randint(0, len(images_c2) - 1)
                self.images.append(images_c2[index])
                self.labels.append(labels_c2[index])
                self.names.append(names_c2[index])
                #
                index = random.randint(0, len(images_c3) - 1)
                self.images.append(images_c3[index])
                self.labels.append(labels_c3[index])
                self.names.append(names_c3[index])
        else:
            for i in range(len(images_c1)):
                self.images.append(images_c1[i])
                self.labels.append(labels_c1[i])
                self.names.append(names_c1[i])
            for i in range(len(images_c2)):
                self.images.append(images_c2[i])
                self.labels.append(labels_c2[i])
                self.names.append(names_c2[i])
            for i in range(len(images_c3)):
                self.images.append(images_c3[i])
                self.labels.append(labels_c3[i])
                self.names.append(names_c3[i])
        #
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.names = np.array(self.names)





# ==============================================================================================
# 2-class classfication용 Dataset
class Dataset_2class():
    def __init__(self,
                 files_c1, start_c1, end_c1, file_path_c1,  # [1, 0]
                 files_c2, start_c2, end_c2, file_path_c2,  # [0, 1]
                 files_c3, start_c3, end_c3, file_path_c3,  # [0, 1]
                 img_size=512, balance=True, balance_size=100, augment=0):
        self.images = []
        self.labels = []
        self.names = []
        #
        # ----------------------------------------------------------------------------------
        # Positive
        images_c1 = []
        labels_c1 = []
        names_c1 = []
        for i in range(start_c1, end_c1):
            file_name = files_c1[i]
            img_file = file_path_c1 + '{}.jpg'.format(file_name)
            img = cv2.imread(img_file)
            if img is None:
                print(file_name, img, img_file)
                continue
            img = img[:, :, 0]
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            # -------------------------------------------------------
            # Basic Augmentation : Random Noise
            img = skimage.util.random_noise(img, mode='gaussian', var=0.001)    # ==> 0 ~ 1 scale로 변환됨
            # -------------------------------------------------------
            #
            # Augmentation
            if augment > 0:
                for aug in range(augment):
                    img_aug = augment_images(img)
                    images_c1.append(img_aug)
                    labels_c1.append([1, 0])
                    names_c1.append(file_name)
            elif augment == 0:
                images_c1.append(img)
                labels_c1.append([1, 0])
                names_c1.append(file_name)
        #
        # ----------------------------------------------------------------------------------------
        # Class_Neg
        images_c2 = []
        labels_c2 = []
        names_c2 = []
        for i in range(start_c2, end_c2):
            file_name = files_c2[i]
            img_file = file_path_c2 + '{}.jpg'.format(file_name)
            img = cv2.imread(img_file)
            if img is None:
                print(file_name, img, img_file)
                continue
            img = img[:, :, 0]
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            # -------------------------------------------------------
            # Basic Augmentation : Random Noise
            img = skimage.util.random_noise(img, mode='gaussian', var=0.001)  # ==> 0 ~ 1 scale로 변환됨
            # -------------------------------------------------------
            #
            # Augmentation
            if augment > 0:
                for aug in range(augment):
                    img_aug = augment_images(img)
                    images_c2.append(img_aug)
                    labels_c2.append([0, 1])
                    names_c2.append(file_name)
            elif augment == 0:
                images_c2.append(img)
                labels_c2.append([0, 1])
                names_c2.append(file_name)
        #
        # Class_Neg2
        for i in range(start_c3, end_c3):
            file_name = files_c3[i]
            img_file = file_path_c3 + '{}.jpg'.format(file_name)
            img = cv2.imread(img_file)
            if img is None:
                print(file_name, img, img_file)
                continue
            img = img[:, :, 0]
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
            # -------------------------------------------------------
            # Basic Augmentation : Random Noise
            img = skimage.util.random_noise(img, mode='gaussian', var=0.001)  # ==> 0 ~ 1 scale로 변환됨
            # -------------------------------------------------------
            #
            # Augmentation
            if augment > 0:
                for aug in range(augment):
                    img_aug = augment_images(img)
                    images_c2.append(img_aug)
                    labels_c2.append([0, 1])
                    names_c2.append(file_name)
            elif augment == 0:
                images_c2.append(img)
                labels_c2.append([0, 1])
                names_c2.append(file_name)
        print('\nclass1:', len(images_c1), ', class2:', len(images_c2))
        # ----------------------------------------------------------------------------------
        if balance == True: # class별로 동일한 숫자로 랜덤하게 Dataset구성 (Train용)
            for i in range(balance_size):
                index = random.randint(0, len(images_c1) - 1)
                self.images.append(images_c1[index])
                self.labels.append(labels_c1[index])
                self.names.append(names_c1[index])
                #
                index = random.randint(0, len(images_c2) - 1)
                self.images.append(images_c2[index])
                self.labels.append(labels_c2[index])
                self.names.append(names_c2[index])
        else:
            for i in range(len(images_c1)):
                self.images.append(images_c1[i])
                self.labels.append(labels_c1[i])
                self.names.append(names_c1[i])
            for i in range(len(images_c2)):
                self.images.append(images_c2[i])
                self.labels.append(labels_c2[i])
                self.names.append(names_c2[i])
        #
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.names = np.array(self.names)





# =================================================================================================
# CAM
def get_class_map(label, net_cam, im_width, weight):
    output_channels = int(net_cam.get_shape()[-1])   # channel 갯수
    w_transpose = tf.transpose(weight)
    w_label = tf.gather(w_transpose, label)
    w_label = tf.reshape(w_label, [-1, output_channels, 1])
    net_cam_resized = tf.image.resize_bilinear(net_cam, [im_width, im_width])
    net_cam_resized_reshape = tf.reshape(net_cam_resized, [-1, im_width * im_width, output_channels])
    classmap = tf.matmul(net_cam_resized_reshape, w_label)
    classmap = tf.reshape(classmap, [-1, im_width, im_width])
    return classmap



# =================================================================================================
# Model
class Model_StemCell:
    def __init__(self, graph, device, model_name, img_size=512,
                 n_classes=3, c_layers=3, d_layers=2, learn_rate=1e-5, cost_wgt=[],
                 reverse=False, filter=8, kernel=4, pool=2, drop_out=0.7):
        self.Graph = graph
        self.IMG_SIZE = img_size
        self.Device = device
        self.ModelName = model_name
        print(model_name)
        #
        NUM_CLASSES = n_classes
        FILTER = filter
        KERNEL = kernel
        POOL = pool
        DROP_OUT = drop_out
        self.CONV_LAYERS = c_layers
        self.DENSE_LAYERS = d_layers
        self.DENSE_NODE = 8
        STRIDE = 2
        tf.set_random_seed(777)
        self.learning_rate = learn_rate
        #
        # ----------------------------------------------------------------------
        # Make Network
        self.X = tf.placeholder(tf.float32, (None, self.IMG_SIZE, self.IMG_SIZE, 1))
        self.Y = tf.placeholder(tf.int64, [None, NUM_CLASSES])
        self.IS_TRAIN = tf.placeholder(tf.bool)
        #
        net_cell = self.X
        if reverse:
            net_cell = -self.X  # REVERSE
        ffwd = net_cell
        ker_2x = net_cell
        ker_4x = net_cell
        for n in range(self.CONV_LAYERS):
            print('net_cell:', net_cell.shape)
            FILTER = FILTER * (n % 2 + 1)
            net_cell = tf.layers.conv2d(inputs=net_cell, filters=FILTER, kernel_size=KERNEL, padding='same', activation=tf.nn.relu)
            net_cell = tf.layers.max_pooling2d(inputs=net_cell, pool_size=POOL, strides=STRIDE, padding='same')
            # --------------------------------------------------------------------------------
            ker_2x = tf.layers.conv2d(inputs=ker_2x, filters=FILTER, kernel_size=KERNEL * 2, padding='same', activation=tf.nn.relu)
            ker_2x = tf.layers.max_pooling2d(inputs=ker_2x, pool_size=POOL, strides=STRIDE, padding='same')
            ker_4x = tf.layers.conv2d(inputs=ker_4x, filters=FILTER, kernel_size=KERNEL * 4, padding='same', activation=tf.nn.relu)
            ker_4x = tf.layers.max_pooling2d(inputs=ker_4x, pool_size=POOL, strides=STRIDE, padding='same')
            # --------------------------------------------------------------------------------
            net_cell = tf.layers.batch_normalization(inputs=net_cell, center=True, scale=True, training=self.IS_TRAIN)
            net_cell = tf.layers.dropout(inputs=net_cell, rate=DROP_OUT, training=self.IS_TRAIN)
            ffwd = tf.layers.max_pooling2d(inputs=ffwd, pool_size=POOL, strides=STRIDE, padding='same')
            net_cell = tf.concat([net_cell, ker_2x, ker_4x, ffwd], axis=3)
        #
        # Flatten and Concatenation
        self.net_cam = net_cell
        self.net_flat = tf.reshape(net_cell, [-1, net_cell.shape[1]._value * net_cell.shape[2]._value * net_cell.shape[3]._value])
        #
        for i in range(self.DENSE_LAYERS):
            self.net_flat = tf.layers.dense(self.net_flat, self.DENSE_NODE, activation=tf.nn.relu)
            self.net_flat = tf.layers.dropout(self.net_flat, DROP_OUT)
            self.DENSE_NODE = self.DENSE_NODE * 2
        #
        self.logits = tf.layers.dense(self.net_flat, NUM_CLASSES, kernel_initializer=tf.contrib.layers.xavier_initializer())
        if cost_wgt != [] and len(cost_wgt) == n_classes:
            # weighted logits
            self.logits = tf.multiply(self.logits, tf.constant(cost_wgt))
            self.logits = tf.nn.softmax(self.logits)
        else:
            self.logits = tf.nn.softmax(self.logits)
        print('net_flat:', self.net_flat, ', net_cam=', self.net_cam)
        #
        # ----------------------------------------------------------------------
        # for CAM
        NET_DEPTH = self.net_cam.shape[3]._value
        self.gap = tf.reduce_mean(self.net_cam, (1, 2))
        self.gap_w = tf.get_variable('cam_w1', shape=[NET_DEPTH, NUM_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        self.cam = get_class_map(0, self.net_cam, img_size, self.gap_w)
        #
        # for Batch Normalization
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        #
        # ----------------------------------------------------------------------
        # Prediction and Accuracy
        self.predict = tf.argmax(self.logits, 1)
        self.correct_prediction = tf.equal(self.predict, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        #
        # ----------------------------------------------------------------------
        # Make Session
        if (self.Device != ''):
            self.config = tf.ConfigProto()
            self.config.gpu_options.visible_device_list = self.Device
            self.sess = tf.Session(graph=self.Graph, config=self.config)
        else:
            self.sess = tf.Session(graph=self.Graph)
        self.sess.run(tf.global_variables_initializer())
        if (model_name != ''):
            try:
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, model_name)
            except:
                print("Exception when loading:", model_name)
    #
    # ----------------------------------------------------------------------
    # Functions
    def train(self, x, y, is_train):
        if is_train:
            _ = self.sess.run(self.optimizer, feed_dict={self.X: x, self.Y: y, self.IS_TRAIN: is_train})
        l, p, c, a = self.sess.run([self.logits, self.predict, self.cost, self.accuracy],
                                   feed_dict={self.X: x, self.Y: y, self.IS_TRAIN: is_train})
        return c, l, p, a
    # ----------------------------------------------------------------------
    def test(self, x):
        l, p = self.sess.run([self.logits, self.predict], feed_dict={self.X: x, self.IS_TRAIN: False})
        return l, p
    # ----------------------------------------------------------------------
    def get_cam(self, x):
        cam_val, p_val = self.sess.run([self.cam, self.predict], feed_dict={self.X: x, self.IS_TRAIN: False})
        return cam_val, p_val
    # ----------------------------------------------------------------------
    def save(self):
        self.saver.save(self.sess, self.ModelName)
        return
    #
    def save(self, model_name):
        self.saver.save(self.sess, model_name)
        return



# ==============================================================================================
# Training model
def train_model(model, acc, cost, dataset, batch_size=10):
    best_acc = acc
    best_cost = cost
    print('-' * 25)
    # ----------------------------------------------------------------------
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            IS_TRAIN = True
        else:
            IS_TRAIN = False
        train_data = dataset.get(phase)
        #
        # Training / Evaluation
        cost_sum = 0
        accuracy_sum = 0
        total_cnt = 0
        BATCH_CNT = int(len(train_data.images) / batch_size)
        img_size = model.IMG_SIZE
        for b in range(BATCH_CNT):
            batch_X = train_data.images[batch_size * b: batch_size * (b + 1)].reshape(-1, img_size, img_size, 1)
            batch_Y = train_data.labels[batch_size * b: batch_size * (b + 1)]
            #
            c, l, p, a = model.train(batch_X, batch_Y, IS_TRAIN)
            cost_sum += c
            accuracy_sum += a
            total_cnt += 1
        #
        # ----------------------------------------------------------------------
        # Statistics
        if total_cnt == 0:
            continue
        epoch_cost = cost_sum / total_cnt
        epoch_acc = accuracy_sum / total_cnt
        print(phase, ', cost={0:0.4f}'.format(epoch_cost), ', accuracy={0:0.4f}'.format(epoch_acc))
        #
        # ----------------------------------------------------------------------
        # Save Model
        if phase == 'val' and epoch_cost < best_cost:
            best_acc = epoch_acc
            best_cost = epoch_cost
            print('change best_model: {0:0.4f}'.format(epoch_cost))
            model.save(model.ModelName)
    # --------------------------------------------------------------------------
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Cost: {:4f}'.format(best_cost))
    return model, best_acc, best_cost



# ==============================================================================================
# Validation with CAM
def validate_cam(model, valid_data, CAM_PATH, NUM_VALIDATE=1000, PATCH_THRESH=0, LABEL=[1, 2, 3]):
    cnt = 0
    cnt_correct = 0
    for i in range(len(valid_data.images)):
        name = valid_data.names[i]
        img = valid_data.images[i]
        label = valid_data.labels[i]
        label_str = LABEL[np.argmax(label)]
        cnt += 1
        if (cnt > NUM_VALIDATE):
           break
        # -------------------------------------------------------------
        # Test model
        l_val, p_val = model.test(img.reshape(1, model.IMG_SIZE, model.IMG_SIZE, 1))
        if p_val[0] == np.argmax(label):
            cnt_correct += 1
        pred_str = LABEL[p_val[0]]
        print("name=", name, ', label=', label_str, ', pred=', pred_str)
        #
        # -------------------------------------------------------------
        # Draw CAM
        cam_val, cam_p = model.get_cam(img.reshape(1, model.IMG_SIZE, model.IMG_SIZE, 1))
        cam_vis = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())), cam_val))
        cam_vis = np.array(cam_vis).reshape(model.IMG_SIZE, model.IMG_SIZE)
        cam_patch = np.zeros([model.IMG_SIZE, model.IMG_SIZE])
        for dx in range(model.IMG_SIZE):
            for dy in range(model.IMG_SIZE):
                if cam_vis[dx][dy] >= PATCH_THRESH:
                    cam_patch[dx][dy] = cam_vis[dx][dy]
        #
        FIG_SIZE = (16, 8)
        pt = plt.figure(figsize=FIG_SIZE)
        #
        # Original
        pt = plt.subplot(1, 2, 1)
        pt = plt.title('Original', fontsize=9)
        pt = plt.imshow(img, cmap='gray')
        pt = plt.axis('off')
        #
        # Predict
        pt = plt.subplot(1, 2, 2)
        logit_val = ''
        for j in range(len(l_val[0])):
            logit_val = logit_val + ':%0.2f' % l_val[0][j]
        pt = plt.title('Label:{}, Pred:{} [{}]'.format(label_str, pred_str, logit_val), fontsize=9)
        pt = plt.imshow(img, cmap='gray')
        # ------------------------------------------------------
        pt = plt.imshow(cam_patch, cmap=plt.cm.jet, alpha=0.4, interpolation='nearest', vmin=0, vmax=1)
        pt = plt.axis('off')
        #
        save_file = CAM_PATH + '/L_{}_P_{}_{}.png'.format(label_str, pred_str, name)
        plt.savefig(save_file)
        plt.close()
    #
    print('acc=', '{0:0.3f}'.format(cnt_correct / min([NUM_VALIDATE, len(valid_data.images)])))





# ======================================================================================
# Day정보 반영 Model
class Model_StemCell_Days:
    def __init__(self, graph, device, model_name, img_size=512,
                 n_classes=3, c_layers=3, d_layers=2, learn_rate=1e-5, cost_wgt=[],
                 reverse=False, filter=8, kernel=3, pool=2, drop_out=0.7):
        self.Graph = graph
        self.IMG_SIZE = img_size
        self.Device = device
        self.ModelName = model_name
        print(model_name)
        #
        NUM_CLASSES = n_classes
        FILTER = filter
        KERNEL = kernel
        POOL = pool
        DROP_OUT = drop_out
        self.CONV_LAYERS = c_layers
        self.DENSE_LAYERS = d_layers
        self.DENSE_NODE = 8
        STRIDE = 2
        tf.set_random_seed(777)
        self.learning_rate = learn_rate
        #
        # ----------------------------------------------------------------------
        # Make Network
        self.X = tf.placeholder(tf.float32, (None, self.IMG_SIZE, self.IMG_SIZE, 1))
        self.D = tf.placeholder(tf.float32, [None, 20])   # day0, day0.5, day1, day1.5 ... day9.5
        self.Y = tf.placeholder(tf.int64, [None, NUM_CLASSES])
        self.IS_TRAIN = tf.placeholder(tf.bool)
        #
        net_cell = self.X
        if reverse:
            net_cell = -self.X  # REVERSE
        ker_2x = net_cell
        ker_4x = net_cell
        ffwd = net_cell
        for n in range(self.CONV_LAYERS):
            print('net_cell:', net_cell.shape)
            FILTER = FILTER * (n % 2 + 1)
            net_cell = tf.layers.conv2d(inputs=net_cell, filters=FILTER, kernel_size=KERNEL, padding='same', activation=tf.nn.relu)
            # --------------------------------------------------------------------------------
            ker_2x = tf.layers.conv2d(inputs=ker_2x, filters=FILTER, kernel_size=KERNEL * 2, padding='same', activation=tf.nn.relu)
            ker_2x = tf.layers.max_pooling2d(inputs=ker_2x, pool_size=POOL, strides=STRIDE, padding='same')
            ker_4x = tf.layers.conv2d(inputs=ker_4x, filters=FILTER, kernel_size=KERNEL * 4, padding='same', activation=tf.nn.relu)
            ker_4x = tf.layers.max_pooling2d(inputs=ker_4x, pool_size=POOL, strides=STRIDE, padding='same')
            # --------------------------------------------------------------------------------
            net_cell = tf.layers.max_pooling2d(inputs=net_cell, pool_size=POOL, strides=STRIDE, padding='same')
            net_cell = tf.layers.batch_normalization(inputs=net_cell, center=True, scale=True, training=self.IS_TRAIN)
            net_cell = tf.layers.dropout(inputs=net_cell, rate=DROP_OUT, training=self.IS_TRAIN)
            ffwd = tf.layers.max_pooling2d(inputs=ffwd, pool_size=POOL, strides=STRIDE, padding='same')
            net_cell = tf.concat([net_cell, ker_2x, ker_4x, ffwd], axis=3)
        #
        # Flatten and Concatenation
        self.net_cam = net_cell
        self.net_flat = tf.reshape(net_cell, [-1, net_cell.shape[1]._value * net_cell.shape[2]._value * net_cell.shape[3]._value])
        #
        for i in range(self.DENSE_LAYERS):
            self.net_flat = tf.layers.dense(self.net_flat, self.DENSE_NODE, activation=tf.nn.relu)
            if i == 0:
                self.net_flat = tf.concat([self.net_flat, self.D], axis=1)
            self.net_flat = tf.layers.dropout(self.net_flat, DROP_OUT)
            self.DENSE_NODE = self.DENSE_NODE * 2
        #
        self.logits = tf.layers.dense(self.net_flat, NUM_CLASSES, kernel_initializer=tf.contrib.layers.xavier_initializer())
        if cost_wgt != [] and len(cost_wgt) == n_classes:
            # weighted logits
            self.logits = tf.multiply(self.logits, tf.constant(cost_wgt))
            self.logits = tf.nn.softmax(self.logits)
        else:
            self.logits = tf.nn.softmax(self.logits)
        print('net_flat:', self.net_flat, ', net_cam=', self.net_cam)
        #
        # ----------------------------------------------------------------------
        # for CAM
        NET_DEPTH = self.net_cam.shape[3]._value
        self.gap = tf.reduce_mean(self.net_cam, (1, 2))
        self.gap_w = tf.get_variable('cam_w1', shape=[NET_DEPTH, NUM_CLASSES], initializer=tf.contrib.layers.xavier_initializer())
        self.cam = get_class_map(0, self.net_cam, img_size, self.gap_w)
        #
        # for Batch Normalization
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        #
        # ----------------------------------------------------------------------
        # Prediction and Accuracy
        self.predict = tf.argmax(self.logits, 1)
        self.correct_prediction = tf.equal(self.predict, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        #
        # ----------------------------------------------------------------------
        # Make Session
        if (self.Device != ''):
            self.config = tf.ConfigProto()
            self.config.gpu_options.visible_device_list = self.Device
            self.sess = tf.Session(graph=self.Graph, config=self.config)
        else:
            self.sess = tf.Session(graph=self.Graph)
        self.sess.run(tf.global_variables_initializer())
        if (model_name != ''):
            try:
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, model_name)
            except:
                print("Exception when loading:", model_name)
    #
    # ----------------------------------------------------------------------
    # Functions
    def train(self, x, d, y, is_train):
        if is_train:
            _ = self.sess.run(self.optimizer, feed_dict={self.X: x, self.D: d, self.Y: y, self.IS_TRAIN: is_train})
        l, p, c, a = self.sess.run([self.logits, self.predict, self.cost, self.accuracy],
                                   feed_dict={self.X: x, self.D: d, self.Y: y, self.IS_TRAIN: is_train})
        return c, l, p, a
    # ----------------------------------------------------------------------
    def test(self, x, d):
        l, p = self.sess.run([self.logits, self.predict], feed_dict={self.X: x, self.D: d, self.IS_TRAIN: False})
        return l, p
    # ----------------------------------------------------------------------
    def get_cam(self, x, d):
        cam_val, p_val = self.sess.run([self.cam, self.predict], feed_dict={self.X: x, self.D: d, self.IS_TRAIN: False})
        return cam_val, p_val
    # ----------------------------------------------------------------------
    def save(self):
        self.saver.save(self.sess, self.ModelName)
        return
    #
    def save(self, model_name):
        self.saver.save(self.sess, model_name)
        return



# ==============================================================================================
# Training model
def train_model_with_days(model, acc, cost, dataset, batch_size=10, document={}):
    best_acc = acc
    best_cost = cost
    print('-' * 25)
    # ----------------------------------------------------------------------
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            IS_TRAIN = True
        else:
            IS_TRAIN = False
        train_data = dataset.get(phase)
        if train_data is None:
            print('data is None')
            return model, best_acc, best_cost
        #
        # Training / Evaluation
        cost_sum = 0
        accuracy_sum = 0
        total_cnt = 0
        BATCH_CNT = int(len(train_data.images) / batch_size)
        img_size = model.IMG_SIZE
        for b in range(BATCH_CNT):
            batch_X = []
            batch_D = []
            batch_Y = []
            batch_names = train_data.names[batch_size * b: batch_size * (b + 1)]
            x_temp = train_data.images[batch_size * b: batch_size * (b + 1)].reshape(-1, img_size, img_size, 1)
            y_temp = train_data.labels[batch_size * b: batch_size * (b + 1)]
            # ---------------------------------------------------------
            for i in range(batch_size):
                name = batch_names[i]
                day_info = document[document.filename == name + '.jpg']
                #
                if len(day_info) >= 1:
                    batch_D.append([
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
                    batch_X.append(x_temp[i])
                    batch_Y.append(y_temp[i])
            # ---------------------------------------------------------
            c, l, p, a = model.train(batch_X, batch_D, batch_Y, IS_TRAIN)
            cost_sum += c
            accuracy_sum += a
            total_cnt += 1
        #
        # ----------------------------------------------------------------------
        # Statistics
        if total_cnt == 0:
            continue
        epoch_cost = cost_sum / total_cnt
        epoch_acc = accuracy_sum / total_cnt
        print(phase, ', cost={0:0.4f}'.format(epoch_cost), ', accuracy={0:0.4f}'.format(epoch_acc))
        #
        # ----------------------------------------------------------------------
        # Save Model
        if phase == 'val' and epoch_cost < best_cost:
            best_acc = epoch_acc
            best_cost = epoch_cost
            print('change best_model: {0:0.4f}'.format(epoch_cost))
            model.save(model.ModelName)
    # --------------------------------------------------------------------------
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Cost: {:4f}'.format(best_cost))
    return model, best_acc, best_cost




# ==============================================================================================
# Validation with CAM
def validate_cam_with_days(model, valid_data, CAM_PATH, NUM_VALIDATE=1000, PATCH_THRESH=0, LABEL=[1, 2], document={}):
    cnt = 0
    cnt_correct = 0
    for i in range(len(valid_data.images)):
        name = valid_data.names[i]
        img = valid_data.images[i]
        label = valid_data.labels[i]
        label_str = LABEL[np.argmax(label)]
        # -------------------------------------------------------------
        # Test model
        print(i, 'name=', name)
        day_one_hot = []
        day_info = document[document.filename == name + '.jpg']
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
            l_val, p_val = model.test(img.reshape(1, model.IMG_SIZE, model.IMG_SIZE, 1), day_one_hot)
            if p_val[0] == np.argmax(label):
                cnt_correct += 1
            pred_str = LABEL[p_val[0]]
            print('\tlabel=', label_str, ', pred=', pred_str)
            #
            # -------------------------------------------------------------
            # Draw CAM
            cam_val, cam_p = model.get_cam(img.reshape(1, model.IMG_SIZE, model.IMG_SIZE, 1), day_one_hot)
            cam_vis = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())), cam_val))
            cam_vis = np.array(cam_vis).reshape(model.IMG_SIZE, model.IMG_SIZE)
            cam_patch = np.zeros([model.IMG_SIZE, model.IMG_SIZE])
            for dx in range(model.IMG_SIZE):
                for dy in range(model.IMG_SIZE):
                    if cam_vis[dx][dy] >= PATCH_THRESH:
                        cam_patch[dx][dy] = cam_vis[dx][dy]
            #
            FIG_SIZE = (16, 8)
            pt = plt.figure(figsize=FIG_SIZE)
            #
            # Original
            pt = plt.subplot(1, 2, 1)
            pt = plt.title('Original', fontsize=9)
            pt = plt.imshow(img, cmap='gray')
            pt = plt.axis('off')
            #
            # Predict
            pt = plt.subplot(1, 2, 2)
            logit_val = ''
            for j in range(len(l_val[0])):
                logit_val = logit_val + ':%0.2f' % l_val[0][j]
            pt = plt.title('Label:{}, Pred:{} [{}]'.format(label_str, pred_str, logit_val), fontsize=9)
            pt = plt.imshow(img, cmap='gray')
            # ------------------------------------------------------
            pt = plt.imshow(cam_patch, cmap=plt.cm.jet, alpha=0.4, interpolation='nearest', vmin=0, vmax=1)
            pt = plt.axis('off')
            #
            save_file = CAM_PATH + '/L_{}_P_{}_{}.png'.format(label_str, pred_str, name)
            plt.savefig(save_file)
            plt.close()
        # --------------------------------------------------------------------------------
        cnt += 1
        if (cnt > NUM_VALIDATE):
           break
        # --------------------------------------------------------------------------------
    #
    print('acc=', '{0:0.3f}'.format(cnt_correct / min([NUM_VALIDATE, len(valid_data.images)])))




