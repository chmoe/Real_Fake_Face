# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: Resnet 
# @Author: Henry Ng 
# @Date: 2021/12/06 16:03

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from config import Config


class Resnet(object):

    def __init__(self,  shape: (int, int, int), frozen_layer: int = Config.frozen_layer_resnet):
        self.frozen_layer = frozen_layer
        self.shape = shape

    def model(self) -> tf.keras.models.Model:
        input_tensor = Input(shape=self.shape)
        resnet_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

        top_model = Sequential()
        top_model.add(Flatten(input_shape=resnet_model.output_shape[1:]))
        top_model.add(Dense(units=256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(units=1, activation='sigmoid'))

        model = Model(resnet_model.input, top_model(resnet_model.output))

        # 冻结层
        for layer in model.layers[:self.frozen_layer]:
            layer.trainable = False

        model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(learning_rate=1e-4, momentum=0.9),
            metrics=['accuracy']
        )
        return model

