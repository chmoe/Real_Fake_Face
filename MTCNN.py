# -*- coding: UTF-8 -*-
# @Project: Real_Fake_Face 
# @File: haar_cascade 
# @Author: Henry Ng 
# @Date: 2021-11-29 14:02
# %%
import cv2  # conda install opencv
from mtcnn import MTCNN  # conda install mtcnn
import copy
# %%
train_file_path = "./archive/real_and_fake_face/training_"
# %%
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])
def get_image_file_list(path):
    image_filenames = [os.path.join(path, x) for x in os.listdir(path) if is_image_file(x)]
    return image_filenames

def path_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
# %%
fake_image_filenames = get_image_file_list(train_file_path + "fake")
real_image_filenames = get_image_file_list(train_file_path + "real")
# %%
fake_image_filenames
# %%
# detector
detector = MTCNN()

def get_face(list_name, name):
    path = path_exist('./result/' + name + '/')
    for i in range(851, len(list_name)):
        img = cv2.imread(list_name[i])
        
        face_frame = copy.deepcopy(img)

        # BGR2RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 检测出人脸的四个点
        dets = detector.detect_faces(img_rgb)

        for face in dets:
            # 座標と高幅を取得
            box_x, box_y, box_w, box_h = face['box']
            # 顔のトリミング
            face_image = face_frame[box_y:box_y + box_h, box_x:box_x + box_w]

            cv2.imwrite(path + str(i) + '.jpg', face_image)
        print("Processing: {}/{}".format(i, len(list_name)))
# %%
# get_face(fake_image_filenames, 'fake')
get_face(real_image_filenames, 'real')
# %%








#%%
# 参考内容
# Python 获取文件夹下的所有图片: https://blog.csdn.net/happyday_d/article/details/84899341