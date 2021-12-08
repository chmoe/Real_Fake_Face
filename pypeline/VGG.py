# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: VGG16 
# @Author: Henry Ng 
# @Date: 2021/12/08 16:30

from tensorflow.keras.layers import Input, Dropout, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
from config import Config


class VGG(object):

    def __init__(self, shape: (int, int, int), frozen_layer: int = Config.frozen_layer_vgg16):
        self.shape = shape
        self.frozen_layer = frozen_layer

    def model(self) -> tf.keras.models.Model:
        input_tensor = Input(shape=self.shape)
        vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
        vgg16_model.summary()
        # %%
        # 构建全连接层
        top_model = Sequential()
        top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(units=1, activation='sigmoid'))
        model = Model(vgg16_model.input, top_model(vgg16_model.output))

        for layer in model.layers[: self.frozen_layer]:
            layer.trainable = False

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model

