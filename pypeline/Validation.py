# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: Validation 
# @Author: Henry Ng 
# @Date: 2021/12/06 18:40
import copy

import numpy as np
import tensorflow as tf
from config import Config
from Debug import Debug


class Validation(tf.keras.callbacks.Callback):

    def __init__(self, validation_gen, label: dict):
        super().__init__()
        self.val_loss = None
        self.val_acc = None
        self.generator = validation_gen
        self.label = label  # {'fake': 0, 'real': 1}

    def on_train_begin(self, logs=None):
        self.val_acc = []
        self.val_loss = []
        return

    def calc(self, img_ndarry: np.ndarray) -> int:
        """
        根据给定的一组图片计算综合的真伪（多数決）
        :param img_ndarry: 一组图片的数组
        :return:
        """
        # print("calc的label", self.label)
        tmp_list = {self.label[Config.name_fake]: 0, self.label[Config.name_real]: 0}  # 暂时存放每张图片预测的真伪
        for img in img_ndarry:
            print(type(img))
            print(img)
            print(self.model.summary())
            dealed_img = np.array([img])
            print(dealed_img)
            print('dealed_img.shape', dealed_img.shape)
            try:
                predict_result = self.model.predict(dealed_img)
                tmp_list[predict_result] += 1
            except Exception as e:
                print('predict:', e)
        if tmp_list[self.label[Config.name_fake]] > tmp_list[self.label[Config.name_real]]:
            return self.label[Config.name_fake]
        elif tmp_list[self.label[Config.name_fake]] < tmp_list[self.label[Config.name_real]]:
            return self.label[Config.name_real]
        else:
            return -1

    def on_epoch_end(self, epoch, logs=None):
        count = 0
        right = 0
        value = next(self.generator)
        # Debug.info('调用on_epoch_end')
        # print(value)
        while not value[2]:  # 没有完成一个循环
            # Debug.info('进入循环')
            for i in range(len(value[0])):
                if self.calc(value[0][i]) == value[1][i]:
                    right += 1
                count += 1
                # Debug.info('加一')
            value = next(self.generator)
        val_acc = right / count
        self.val_acc.append(val_acc)

        return

    def on_train_end(self, logs=None):
        history = self.model.history.history
        history['val_acc'] = self.val_acc
        history['val_loss'] = self.val_loss
        return
