# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: Net 
# @Author: Henry Ng 
# @Date: 2021/12/08 15:02

import os
from tensorflow.keras.preprocessing import image

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
import tensorflow as tf
from config import Config
import sys
sys.path.append('/')
from pypeline.Validation import Validation
from pypeline.VGG import VGG
from pypeline.XCeption import XCeption
from pypeline.Resnet import Resnet
from Debug import Debug


class Net(object):
    def __init__(self, k: int, frozen_layer: int = Config.frozen_layer_vgg16, net_name: str = Config.name_vgg16):
        self.K = k
        self.image_width = 300
        self.image_height = 300
        self.nb_epoch = 100
        self.batch_size = 32
        self.frozen_layer = frozen_layer
        self.label = {}
        self.net_name = net_name

    @staticmethod
    def save_history(history, result_file):
        loss = history.history['loss']
        acc = history.history['accuracy']
        # val_loss = history.history['val_loss']
        val_acc = history.history['val_accuracy']
        nb_epoch = len(acc)

        with open(result_file, "w") as fp:
            fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
            for i in range(nb_epoch):
                # fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[min(i, len(loss) - 1)], acc[min(i, len(acc) - 1)], val_loss[min(i, len(val_loss) - 1)], val_acc[min(i, len(val_acc) - 1)]))
                fp.write("%d\t%f\t%f\t%f\n" % (i, loss[min(i, len(loss) - 1)], acc[min(i, len(acc) - 1)], val_acc[min(i, len(val_acc) - 1)]))

    def create_model(self) -> tf.keras.models.Model:
        shape = (self.image_width, self.image_height, 3)
        model = Model()
        if Config.name_vgg16 == self.net_name:
            return VGG(shape=shape, frozen_layer=self.frozen_layer).model()
        elif Config.name_resnet == self.net_name:
            return Resnet(shape=shape, frozen_layer=self.frozen_layer).model()
        elif Config.name_xception == self.net_name:
            return XCeption(shape=shape, frozen_layer=self.frozen_layer).model()
        return model

    def generate_train(self, batch_size: int = 32, target_size: (int, int) = (256, 256)):
        head_data_path = Config.path(Config.slic_result_path, self.K, Config.name_train)

        train_datagen = image.ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        train_generator = train_datagen.flow_from_directory(
            head_data_path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='binary')
        self.label = train_generator.class_indices  # {'fake': 0, 'real': 1}
        return train_generator

    def generate_validation(self, batch_size: int = 32, target_size: (int, int) = (256, 256)):
        head_data_path = Config.path(Config.slic_result_path, self.K, Config.name_validation)
        real_path = Config.path(head_data_path, Config.name_real)
        fake_path = Config.path(head_data_path, Config.name_fake)
        real_list = Config.get_child_folder(real_path)  # 存储真图片的文件夹 每张图片一个文件夹
        fake_list = Config.get_child_folder(fake_path)
        image_list = real_list + fake_list  # 所有图片的子文件夹
        boundary = len(real_list)
        max_len = len(image_list)

        steps = 0
        finish_flag = False
        while True:
            x_list = []  # 存储图片的数组，每次返回后清零 （len/batch, 20, 300, 300, 3）
            y_list = []
            for i in range(steps, min(steps + batch_size, max_len)):
                tmp_list = []  # （20, 300, 300, 3)
                for pic in Config.get_image_file_list(image_list[i]):  # 遍历每一个子文件夹（20）
                    tmp_list.append(np.array(image.load_img(
                        path=pic,
                        target_size=target_size
                    )))
                x_list.append(np.array(tmp_list))  # （20, 300, 300, 3)
                y_list.append(self.label[Config.name_real] if i < boundary else self.label[Config.name_fake])  # 1

            steps += batch_size
            if steps >= max_len:
                steps = 0
                finish_flag = True
            # (n, 20, 300, 300, 3)  (n, 1)
            yield np.array(x_list), np.array(y_list), finish_flag
            finish_flag = False

    def fit(self, model: tf.keras.models.Model) -> tf.keras.models.Model:
        check = Config.path_exist(Config.path(Config.checkpoint_path, str(self.K), self.net_name))
        checkpoint_path = check + '/cp-{epoch:04d}.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)

        cp_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='accuracy',
            save_best_only=True,
            model='auto',
            save_weights_only=True,
            save_freq=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='accuracy',
            factor=0.1,
            patience=2,
        )
        train = self.generate_train(
                batch_size=self.batch_size,
                target_size=(self.image_width, self.image_height)
            )
        validation = Validation(
                    validation_gen=self.generate_validation(
                        batch_size=self.batch_size,
                        target_size=(self.image_width, self.image_height)
                    ),
                    label=self.label,  # {'fake': 0, 'real': 1}
                    status='{} slice with {} Net'.format(self.K, self.net_name)
                )
        initial_epoch = 0
        if os.path.exists(check):
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            if latest:
                model.load_weights(latest)
                tmp_len = len(checkpoint_dir) + 4  # 路径长度
                initial_epoch = int(latest[tmp_len:tmp_len + 4]) - 1
        model.fit_generator(
            generator=train,
            epochs=self.nb_epoch,
            # steps_per_epoch=nb_train_samples,
            # validation_data=validation_generator,
            # validation_steps=nb_validation_samples
            callbacks=[
                cp_callback,
                reduce_lr,
                validation
            ],
            initial_epoch=initial_epoch
        )
        return model

    def save(self, model: tf.keras.models.Model):
        model.save_weights(Config.path_exist(Config.model_path) + '{}_{}_fine-tuning.h5'.format(self.K, self.net_name))
        self.save_history(model.history, Config.path_exist(Config.history_path) + '{}_{}_history.txt'.format(self.K, self.net_name))

    def run(self):
        if not os.path.isfile(Config.path_exist(Config.model_path) + '{}_{}_fine-tuning.h5'.format(self.K, self.net_name)):
            Debug.info("Net_Processing: {} net is running with {} slices".format(self.net_name, self.K))
            model = self.create_model()
            self.save(self.fit(model))
        else:
            Debug.info('{} net exists, skipping...'.format(self.net_name))
