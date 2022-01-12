# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: main 
# @Author: Henry Ng 
# @Date: 2022/01/11 9:45

import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.training.checkpoint_management import latest_checkpoint

face_area_path = './face_area/'
train_face_area_path = face_area_path + 'train/'
validation_face_area_path = face_area_path + 'validation/'

fake_train_path = train_face_area_path + 'fake/'  # 训练用处理好的图像的fake
real_train_path = train_face_area_path + 'real/'  # 训练用处理好的图像的real
fake_validation_path = validation_face_area_path + 'fake/'  # 测试用处理好的图像的fake
real_validation_path = validation_face_area_path + 'real/'  # 测试用处理好的图像的real

target_size = (300, 300)
shape = (300, 300, 3)
batch_size = 32


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[min(i, len(loss) - 1)], acc[min(i, len(acc) - 1)], val_loss[min(i, len(val_loss) - 1)], val_acc[min(i, len(val_acc) - 1)]))


# 创建xception模型
input_tensor = Input(shape)
xception_model = Xception(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=xception_model.output_shape[1:]))
top_model.add(Dense(units=256, activation='relu'))
top_model.add(Dense(units=1, activation='sigmoid'))

model = Model(xception_model.input, top_model(xception_model.output))

for layer in model.layers[: 101]:
    layer.trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=SGD(learning_rate=1e-4, momentum=0.9),
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    train_face_area_path,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
)
validation_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    validation_face_area_path,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
)

check = './checkpoint/'
makedirs(check)
checkpoint_path = check + '/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='accuracy',
    save_best_only=True,
    save_weights_only=True,
    model='max',
    verbose=0,
    save_freq=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=0,
    mode='min',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0
)

initial_epoch = 0
if os.path.exists(check):
    latest = latest_checkpoint(checkpoint_dir)
    if latest:
        model.load_weights(latest)
        tmp_len = len(checkpoint_dir) + 4  # 路径长度
        initial_epoch = int(latest[tmp_len:tmp_len + 4]) - 1

model.fit_generator(
    generator=train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[
        cp_callback,
        reduce_lr,
    ],
    initial_epoch=initial_epoch,
)

model.save('./60_xception_fine-tuning.h5')
save_history(model.history, './history.txt')

"""
 TPTNFPFN：https://www.cnblogs.com/daniel-d/p/7889888.html
"""