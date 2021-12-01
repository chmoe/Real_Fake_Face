# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: ResNet 
# @Author: Henry Ng 
# @Date: 2021-12-01 21:27:12

# %%
import os
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf

# %%
def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['accuracy']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    nb_epoch = len(acc)

    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))


# %%
train_data_path = "./SLIC_result/"
validation_data_path = "./validation/"
image_width, image_height = 300, 300
batch_size = 32
nb_epoch = 50

result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
# %%
input_tensor = Input(shape=(image_width, image_height, 3))
resnet_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
# resnet_model = Model(resnet_model.input, GlobalAveragePooling2D()(resnet_model.output))
resnet_model.summary()
# %%
top_model = Sequential()
top_model.add(resnet_model)
# top_model.add(Flatten(input_shape=resnet_model.output_shape[1:]))
top_model.add(Flatten())
top_model.add(Dense(units=256, activation='relu'))
# top_model.add(Dropout(0.5))
top_model.add(Dense(units=1, activation='sigmoid'))
# top_model.load_weights(os.path.join(result_dir, 'bottleneck_fc_model.h5'))

# %%
# model = Model(resnet_model.input, top_model(resnet_model.output))
model = top_model
print('resnet_model: ', resnet_model)
print('top_model: ', top_model)
print('model', model)
model.summary()
# %%
for i in range(len(model.layers)):
    print(i, model.layers[i])
# %%
# 冻结前4个层（即前面的「2233」4个层（小层是0~14）
for layer in model.layers[:1]:
    layer.trainable = False
# %%
model.compile(
    loss='binary_crossentropy',
    optimizer=SGD(learning_rate=1e-4, momentum=0.9),
    metrics=['accuracy']
)
# %%
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(image_width, image_height),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(image_width, image_height),
    batch_size=32,
    class_mode='binary')

# %%
check = './checkpoint_res2'
if not os.path.exists(check):
    os.mkdir(check)
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
    monitor='loss',
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
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        model.load_weights(latest)
        tmp_len = len(check) + 4  # 路径长度
        initial_epoch = int(latest[tmp_len:tmp_len + 4])

# %%
history = model.fit_generator(
    generator=train_generator,
    epochs=nb_epoch,
    # steps_per_epoch=nb_train_samples,
    validation_data=validation_generator,
    # validation_steps=nb_validation_samples
    callbacks=[
        cp_callback,
        reduce_lr 
    ],
    initial_epoch=initial_epoch
)

model.save_weights(os.path.join(result_dir, 'res_finetuning2.h5'))
save_history(history, os.path.join(result_dir, 'res_history_finetuning2.txt'))
# %%

'''
参考链接：

http://pchun.work/resnet%E3%82%92fine-tuning%E3%81%97%E3%81%A6%E8%87%AA%E5%88%86%E3%81%8C%E7%94%A8%E6%84%8F%E3%81%97%E3%81%9F%E7%94%BB%E5%83%8F%E3%82%92%E5%AD%A6%E7%BF%92%E3%81%95%E3%81%9B%E3%82%8B/

https://www.cnblogs.com/zhengbiqing/p/12506331.html

https://www.cxybb.com/article/shizhengxin123/72473245
'''