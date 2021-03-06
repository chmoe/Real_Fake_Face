# %%
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
from skimage.segmentation import slic, mark_boundaries
from skimage import io, color
from tensorflow.keras.applications.xception import Xception
from tqdm import tqdm
from tensorflow.keras.optimizers import SGD
import math
# %%
shape = (300, 300, 3)

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


# 训练集生成器
def generate_train(batch_size: int = 32, target_size: (int, int) = (300, 300)):
    head_data_path = '../SLIC_result/60/train/'

    train_datagen = image.ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        head_data_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary')
    return train_generator
# %%
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def get_image_file_list(path, is_train_data: bool = False):
    all_file = [i for i in os.listdir(path) if not i.startswith('.')]
    if not is_train_data:
        all_file.sort(key=lambda x: int(x[:-4]))  # 4代表去掉'.jpg'之类的后缀名
    else:
        # print(all_file[0].split('_'))
        all_file.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1][:-4])))
        
    image_filenames = [os.path.join(path, x) for x in all_file if is_image_file(x)]
    return image_filenames
def path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
#%%
# 获取验证集的图片合集
path = './result/'
slic_path = '../SLIC_result/60/validation/'
fake_image_filenames = get_image_file_list(path + 'fake/')
real_image_filenames = get_image_file_list(path + 'real/')
fake_slic_path = slic_path + 'fake/'
real_slic_path = slic_path + 'real/'

# %%
# fake_start_count = math.ceil(0.8 * len(fake_image_filenames))  # fake起始
# real_start_count = math.ceil(0.8 * len(real_image_filenames))  # real起始
# print('正在运行图像处理：fake')
# start_count = 0
# train_processeed_list = get_image_file_list(path_exist('../SLIC_result/60/validation/fake/'), is_train_data=True)
# if len(train_processeed_list) != 0:
#         start_count = int(train_processeed_list[-1].split('/')[-1].split('_')[0])
# for i in tqdm(range(start_count, len(fake_image_filenames[fake_start_count:]))):
#     img = io.imread(fake_image_filenames[fake_start_count:][i])
#     segments = slic(img, 40, 10)
#     color_dictionary = {}
#     for row in range(segments.shape[0]):
#         for col in range(segments.shape[1]):
#             if segments[row][col] not in color_dictionary:
#                 color_dictionary[segments[row][col]] = [(row, col)]
#             else:
#                 color_dictionary[segments[row][col]].append((row, col))

#     for color in color_dictionary.keys():
#         path = path_exist(slic_path + 'fake/') + "{}_{}.jpg".format(i, color)
#         data_copy = img
#         for row in range(data_copy.shape[0]):
#             for col in range(data_copy.shape[1]):
#                 if (row, col) in color_dictionary[color]:
#                     continue
#                 else:
#                     data_copy[row][col] = (0, 0, 0)
#         io.imsave(path, data_copy, check_contrast=False)
# %%
# print('正在运行图像处理：real')
# start_count = 0
# train_processeed_list = get_image_file_list(path_exist('../SLIC_result/60/validation/real/'), True)
# if len(train_processeed_list) != 0:
#     start_count = int(train_processeed_list[-1].split('/')[-1].split('_')[0])
# for i in tqdm(range(start_count, len(real_image_filenames[real_start_count:]))):
#     img = io.imread(real_image_filenames[real_start_count:][i])
#     segments = slic(img, 40, 10)
#     color_dictionary = {}
#     for row in range(segments.shape[0]):
#         for col in range(segments.shape[1]):
#             if segments[row][col] not in color_dictionary:
#                 color_dictionary[segments[row][col]] = [(row, col)]
#             else:
#                 color_dictionary[segments[row][col]].append((row, col))

#     for color in color_dictionary.keys():
#         path = path_exist(slic_path + 'real/') + "{}_{}.jpg".format(i, color)
#         data_copy = img
#         for row in range(data_copy.shape[0]):
#             for col in range(data_copy.shape[1]):
#                 if (row, col) in color_dictionary[color]:
#                     continue
#                 else:
#                     data_copy[row][col] = (0, 0, 0)
#         io.imsave(path, data_copy, check_contrast=False)

# 生成验证集生成器
def generate_validation(batch_size: int = 32, target_size: (int, int) = (300, 300)):
    head_data_path = '../SLIC_result/60/validation/'

    train_datagen = image.ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_directory(
        head_data_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary')
    return train_generator

check = path_exist('../checkpoint/60/xception/')
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
train = generate_train(
        batch_size=32,
        target_size=(300, 300)
    )
validation = generate_validation(
        batch_size=32,
        target_size=(300, 300)
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
        epochs=200,
        # steps_per_epoch=nb_train_samples,
        validation_data=validation,
        # validation_steps=nb_validation_samples,
        callbacks=[
            cp_callback,
            reduce_lr,
            # validation
        ],
        initial_epoch=initial_epoch
    )
model.save('../models/60_xception_fine-tuning.h5')
save_history(model.history, '../history/60_xception_history.txt')