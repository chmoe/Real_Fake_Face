# -*- coding: UTF-8 -*-
# @Project: Progress 
# @File: config 
# @Author: Henry Ng 
# @Date: 2021/12/06 13:01
import os


class Config(object):
    name_fake = 'fake'
    name_real = 'real'
    name_train = 'train'
    name_validation = 'validation'
    name_vgg16 = 'vgg'
    name_resnet = 'resnet'
    name_xception = 'xception'
    frozen_layer_vgg16 = 15
    frozen_layer_resnet = 100
    frozen_layer_xception = 101
    original_face_path = './result/'  # 人脸区域原始图像
    slic_result_path = './SLIC_result/'  # 超像素处理后的位置
    checkpoint_path = './checkpoint/'  # 检查点的公共文件夹
    model_path = './models/'  # 存放模型的公共文件夹
    history_path = './history/'  # 存放历史记录的公共文件夹

    @staticmethod
    def path_exist(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

    @staticmethod
    def get_image_file_list(path):
        image_filenames = [os.path.join(path, x) for x in os.listdir(path) if Config.is_image_file(x)]
        return image_filenames

    @staticmethod
    def path(*args) -> str:
        import re
        """
        将给定的内容拼接成为路径，并且替换掉//.../
        :param args:
        :return:
        """
        tmp_str = "/"
        for i in args:
            tmp_str = tmp_str + str(i)
            tmp_str += '/'
        return re.sub(r'/{2,}', "/", tmp_str).replace('/.', '.')

    @staticmethod
    def get_child_folder(path_) -> list:
        """
        根据给定的路径返回所有子文件夹的路径，不包含内部文件
        :param path_: 输入的路径
        :return: 返回路径的list
        """
        return [Config.path(path_, i) for i in os.listdir(path_) if not i.startswith('.')]
