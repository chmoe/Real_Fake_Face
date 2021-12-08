# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: XCeption 
# @Author: Henry Ng 
# @Date: 2021/12/08 20:18

from tensorflow.keras.layers import Input, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from config import Config


class XCeption(object):
    def __init__(self,  shape: (int, int, int), frozen_layer: int = Config.frozen_layer_xception):
        self.frozen_layer = frozen_layer
        self.shape = shape

    def model(self) -> tf.keras.models.Model:
        input_tensor = Input(self.shape)
        xception_model = Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)

        top_model = Sequential()
        top_model.add(Flatten(input_shape=xception_model.output_shape[1:]))
        top_model.add(Dense(units=256, activation='relu'))
        top_model.add(Dense(units=1, activation='sigmoid'))

        model = Model(xception_model.input, top_model(xception_model.output))

        for layer in model.layers[: self.frozen_layer]:
            layer.trainable = False

        model.compile(
            loss='binary_crossentropy',
            optimizer=SGD(learning_rate=1e-4, momentum=0.9),
            metrics=['accuracy']
        )
        return model

